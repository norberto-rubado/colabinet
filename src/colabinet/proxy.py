"""Per-session metering proxy (spec §7): pass any request through unchanged, stream the response back as it arrives,
and record one row per request from a Tap on the side."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable
from urllib.parse import quote, unquote, urlsplit

import httpx

from .activity import Activity, is_inference
from .net import QuietServer, start_server, stop_server
from .stats import RequestRecord, Tap

log = logging.getLogger("colabinet.proxy")

HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailers",
              b"transfer-encoding", b"upgrade", b"host"}
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


@dataclass
class Route:
    kind: str  # forward | unavailable (503) | down (502)
    upstream_port: int | None = None
    message: str = ""
    run_id: int | None = None
    inject_usage: bool = False


def with_usage_option(body: bytes) -> bytes:
    """Ask for streamed usage, but only if the client asked for a stream and didn't set stream_options itself."""
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(obj, dict) or not obj.get("stream") or "stream_options" in obj:
        return body
    obj["stream_options"] = {"include_usage": True}
    return json.dumps(obj).encode()


def host_ok(headers: dict[bytes, bytes], port: int) -> bool:
    host = headers.get(b"host", b"").decode("latin-1")
    return host in (f"127.0.0.1:{port}", f"localhost:{port}")


def origin_ok(headers: dict[bytes, bytes]) -> bool:
    origin = headers.get(b"origin")
    if origin is None:
        return True  # non-browser clients (pi, curl, SDKs) send no Origin
    return urlsplit(origin.decode("latin-1")).hostname in LOCAL_HOSTS


MAX_BODY_BYTES = 64 * 1024 * 1024  # JSON / media requests (a 32 MiB base64 image is ~43 MB of JSON)
POLL_CLIENT = "colabinet-poll"  # User-Agent of MediaService's /history, /queue, /models/loras polling
# Set by MediaService on its requests (R3-02): recorded as usual, never counted as inference activity, because a
# POST /prompt answered 200 only queued the job. The public gateway strips it (DROP_REQUEST), so only a local client
# can send it, and this listener is loopback-only anyway.
MEDIA_HEADER = b"x-colabinet-media"
MAX_UPLOAD_BYTES = 128 * 1024 * 1024  # raw ComfyUI uploads


class BodyTooLarge(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f"请求体过大（上限 {limit // (1024 * 1024)} MiB）")
        self.limit = limit


async def read_body(receive, limit: int, declared: bytes | None = None) -> bytes | None:
    """The whole request body, or None on disconnect. Raises BodyTooLarge instead of buffering past `limit`
    bytes; `declared` (the Content-Length header) lets an oversized request fail before anything is read."""
    with contextlib.suppress(ValueError):
        if declared is not None and int(declared) > limit:
            raise BodyTooLarge(limit)
    body = bytearray()
    while True:
        m = await receive()
        if m["type"] == "http.disconnect":
            return None
        chunk = m.get("body", b"")
        if len(body) + len(chunk) > limit:
            raise BodyTooLarge(limit)
        if not body and not m.get("more_body"):
            return bytes(chunk)  # single-message body: no copy
        body += chunk
        if not m.get("more_body"):
            return bytes(body)


# Characters kept as they are when re-quoting a request path: RFC 3986 pchar plus '/'. With '%' in the set,
# existing escapes (%2F, %3F, ...) travel on unchanged; without it, a literal '%' becomes %25.
PATH_SAFE = "/:@!$&'()*+,;=-._~"


def bad_path(path: str) -> bool:
    """`.`/`..` segments (httpx would resolve them and escape /v1) or backslashes in the ASGI-decoded path."""
    return "\\" in path or any(seg in (".", "..") for seg in path.split("/"))


def wire_path(scope) -> str | None:
    """The request path as it arrived (still percent-encoded), to forward upstream unchanged; None if the request
    must be refused (R2-01). Forwarding the decoded scope["path"] would let the next hop decode it again:
    /v1/%252e%252e/props became /v1/../props there, and a decoded %3f/%23 turned into a query or fragment.
    Refused: an encoded '%' (%25, i.e. double encoding) or a decoded path that would decode further, dot segments
    or backslashes in the decoded path, and a raw path that does not decode to scope["path"] (what is validated must
    be what is forwarded)."""
    path = scope["path"]
    raw = scope.get("raw_path")
    # ASGI scopes built by tests or other servers may lack raw_path: use the encoded form of the decoded path.
    wire = quote(raw, safe=PATH_SAFE + "%") if raw is not None else quote(path, safe=PATH_SAFE)
    if "%25" in wire or unquote(path) != path or bad_path(path) or unquote(wire) != path:
        return None
    return wire


async def _send_json(send, status: int, message: str, kind: str) -> None:
    data = json.dumps({"error": {"message": message, "type": kind}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data})


class SessionProxy:
    """ASGI app for one session's local port."""

    _ids = itertools.count(1)

    def __init__(self, port: int, route: Callable[[], Route],
                 record: Callable[[RequestRecord, int], Awaitable[None]], activity: Activity,
                 client: httpx.AsyncClient, now: Callable[[], float],
                 on_upstream_error: Callable[[], None] | None = None) -> None:
        self.port = port
        self.route = route
        self.record = record
        self.activity = activity
        self.client = client
        self.now = now
        self.on_upstream_error = on_upstream_error or (lambda: None)

    async def _record(self, rec: RequestRecord, run_id: int | None) -> None:
        if run_id is None:
            return
        if rec.client == POLL_CLIENT:  # internal media status polling: activity-neutral, never a request row (R1-14)
            return
        try:
            await self.record(rec, run_id)
        except Exception:  # a stats failure must never affect the proxied response
            log.exception("could not record request")

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return
        headers = dict(scope["headers"])
        if not host_ok(headers, self.port) or not origin_ok(headers):
            return await _send_json(send, 403, "colabinet: request rejected (Host/Origin)", "forbidden")
        method, path = scope["method"], scope["path"]
        wire = wire_path(scope)
        if wire is None:  # R2-01: also here, the gateway is not the only client of this port
            return await _send_json(send, 400, "colabinet: invalid path", "invalid_request_error")
        # Body limit and is_inference() use the decoded path validated above; upstream gets the raw `wire` form.
        limit = MAX_UPLOAD_BYTES if path.startswith(("/upload/", "/api/upload/")) else MAX_BODY_BYTES
        try:
            body = await read_body(receive, limit, headers.get(b"content-length"))
        except BodyTooLarge as e:
            return await _send_json(send, 413, f"colabinet: {e}", "request_too_large")
        if body is None:
            return
        t0 = self.now()
        tap = Tap(method, path, t0, headers.get(b"user-agent", b"").decode("latin-1"), body)
        route = self.route()
        if route.kind != "forward":
            status = 503 if route.kind == "unavailable" else 502
            await _send_json(send, status, f"colabinet: {route.message}", "unavailable")
            tap.rec.error_short = route.message[:200]
            return await self._record(tap.finish(self.now(), status), route.run_id)

        client = scope.get("client")
        media = headers.get(MEDIA_HEADER) == b"1" and (client is None or client[0] in ("127.0.0.1", "::1"))
        inference = is_inference(method, path) and not media
        key = next(self._ids)
        if inference:
            self.activity.begin(key, t0)
        query = scope.get("query_string", b"").decode("latin-1")
        url = f"http://127.0.0.1:{route.upstream_port}{wire}" + (f"?{query}" if query else "")
        if route.inject_usage and method == "POST" and path.rstrip("/") in ("/v1/chat/completions", "/v1/completions"):
            body = with_usage_option(body)
        fwd_headers = [(k, v) for k, v in scope["headers"]
                       if k.lower() not in HOP_BY_HOP and k.lower() not in (b"content-length", MEDIA_HEADER)]
        try:
            resp = await self.client.send(self.client.build_request(method, url, headers=fwd_headers, content=body),
                                          stream=True)
        except httpx.TransportError as e:
            self.activity.end(key, self.now(), useful=False)
            self.on_upstream_error()
            await _send_json(send, 502, "colabinet: engine unreachable through the tunnel", "bad_gateway")
            tap.rec.error_short = f"upstream: {type(e).__name__}"
            return await self._record(tap.finish(self.now(), 502), route.run_id)

        tap.start(resp.status_code, resp.headers.get("content-type"))
        disconnected = False

        async def watch() -> None:
            while True:
                if (await receive())["type"] == "http.disconnect":
                    return

        async def pump() -> None:
            await send({"type": "http.response.start", "status": resp.status_code,
                        "headers": [(k, v) for k, v in resp.headers.raw if k.lower() not in HOP_BY_HOP]})
            async for chunk in resp.aiter_raw():
                t = self.now()
                tap.feed(chunk, t)
                self.activity.touch(key, t)
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        watcher, pumper = asyncio.create_task(watch()), asyncio.create_task(pump())
        try:
            done, _ = await asyncio.wait({watcher, pumper}, return_when=asyncio.FIRST_COMPLETED)
            if pumper in done:
                watcher.cancel()
                exc = pumper.exception()
                if exc is not None:
                    tap.rec.error_short = f"upstream: {type(exc).__name__}"
            else:
                disconnected = True
                pumper.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pumper
        finally:
            await resp.aclose()
        t_end = self.now()
        # An error answer (e.g. a misrouted chat request hitting ComfyUI's 404) is not work the user wanted done.
        self.activity.end(key, t_end, useful=inference and resp.status_code < 400)
        await self._record(tap.finish(t_end, 499 if disconnected else None), route.run_id)


class ProxyManager:
    """Starts and stops one listener per session port."""

    def __init__(self) -> None:
        self._servers: dict[int, tuple[QuietServer, asyncio.Task, int]] = {}

    async def bind(self, sid: int, port: int, app) -> None:
        if sid in self._servers:
            return
        server, task = await start_server(app, port)
        self._servers[sid] = (server, task, port)

    async def unbind(self, sid: int) -> None:
        entry = self._servers.pop(sid, None)
        if entry:
            await stop_server(entry[0], entry[1])

    def bound(self, sid: int) -> bool:
        return sid in self._servers

    def ports(self) -> set[int]:
        return {p for _, _, p in self._servers.values()}

    async def close(self) -> None:
        for sid in list(self._servers):
            await self.unbind(sid)
