"""One authenticated OpenAI-compatible gateway in front of every ready session.

Remote clients (behind a reverse proxy, e.g. Cloudflare + OpenResty) send `Authorization: Bearer KEY` or
`x-api-key: KEY`. The gateway picks the ready session whose run alias (or colab name) equals `body.model` and forwards
the request to that session's metering proxy on 127.0.0.1:PROXY_PORT, so per-request stats keep being recorded.

Cloudflare drops a request whose origin has not answered within 100 s. When the upstream has not returned headers
within `grace` seconds, the gateway sends 200 early and then keep-alive bytes every `interval` seconds: SSE comments
(`: keepalive`) for stream requests, single spaces for JSON (parsers skip leading whitespace). SSE streams also get a
comment whenever the upstream is silent for `interval` seconds between events.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

log = logging.getLogger("colabinet.gateway")

GRACE_S = 20.0
INTERVAL_S = 15.0
CACHE_S = 2.0
SSE_KEEPALIVE = b": keepalive\n\n"
JSON_KEEPALIVE = b" "

HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailers",
              b"transfer-encoding", b"upgrade", b"host"}
# Never forwarded: our own credential, the browser Origin (the session proxy rejects foreign origins), and
# content-length (httpx sets it for the possibly rewritten body).
DROP_REQUEST = HOP_BY_HOP | {b"authorization", b"x-api-key", b"origin", b"referer", b"content-length", b"cookie"}


# --- key ---------------------------------------------------------------------------------------------------------
def load_or_create_key(path: Path) -> str:
    """Read the gateway key, creating it (mode 0600) on first use. It survives restarts."""
    path = Path(path)
    if path.exists():
        key = path.read_text().strip()
        if key:
            os.chmod(path, 0o600)
            return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)
    os.chmod(path, 0o600)
    return key


def read_key(path: Path) -> str | None:
    try:
        return Path(path).read_text().strip() or None
    except OSError:
        return None


def mask(key: str | None) -> str:
    if not key:
        return ""
    return key[:4] + "•" * 8 + key[-4:] if len(key) > 12 else "•" * len(key)


# --- session lookup ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Target:
    sid: int
    name: str
    alias: str | None
    port: int
    created: int

    @property
    def model_id(self) -> str:
        return self.alias or self.name


async def ready_targets(store) -> list[Target]:
    """Ready sessions with a proxy port, newest first, read from the store (supervisor.py stays untouched)."""
    out = []
    for row in await store.live_sessions():
        if row.get("state") != "ready" or not row.get("proxy_port"):
            continue
        run = await store.current_run(row["id"])
        out.append(Target(sid=row["id"], name=row.get("colab_name") or f"session-{row['id']}",
                          alias=(run or {}).get("alias") or None, port=int(row["proxy_port"]),
                          created=int(row.get("ready_at") or row.get("created_at") or 0)))
    out.sort(key=lambda t: t.sid, reverse=True)
    return out


def pick(targets: list[Target], model: Any) -> tuple[Target | None, bool]:
    """(target, exact): exact alias match, then colab-name match, then the only ready session."""
    if isinstance(model, str) and model:
        for t in targets:
            if t.alias and t.alias == model:
                return t, True
        for t in targets:
            if t.name == model:
                return t, False
    if len(targets) == 1:
        return targets[0], False
    return None, False


# --- ASGI helpers ------------------------------------------------------------------------------------------------
async def _read_body(receive) -> bytes | None:
    body = b""
    while True:
        m = await receive()
        if m["type"] == "http.disconnect":
            return None
        body += m.get("body", b"")
        if not m.get("more_body"):
            return body


def _error_body(message: str, kind: str, **extra: Any) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind, **extra}}).encode()


async def _send_json(send, status: int, data: bytes, extra: tuple[tuple[bytes, bytes], ...] = ()) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode()),
                            *extra]})
    await send({"type": "http.response.body", "body": data})


class Gateway:
    """ASGI app: auth, /v1/models, and forwarding of every other /v1/* request to a session proxy."""

    def __init__(self, sup, key: str, *, client: httpx.AsyncClient | None = None, grace: float = GRACE_S,
                 interval: float = INTERVAL_S, cache_s: float = CACHE_S,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.sup = sup
        self.key = key.encode()
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(10, read=None), trust_env=False,
                                                  limits=httpx.Limits(max_keepalive_connections=0))
        self._own_client = client is None
        self.grace = grace
        self.interval = interval
        self.cache_s = cache_s
        self.now = now
        self._cache: tuple[float, list[Target]] | None = None

    async def aclose(self) -> None:
        if self._own_client:
            await self.client.aclose()

    async def targets(self) -> list[Target]:
        t = self.now()
        if self._cache and t - self._cache[0] < self.cache_s:
            return self._cache[1]
        targets = await ready_targets(self.sup.store)
        self._cache = (t, targets)
        return targets

    def authorized(self, headers: dict[bytes, bytes]) -> bool:
        candidates = []
        auth = headers.get(b"authorization", b"")
        if auth[:7].lower() == b"bearer ":
            candidates.append(auth[7:].strip())
        if b"x-api-key" in headers:
            candidates.append(headers[b"x-api-key"].strip())
        ok = False
        for c in candidates:  # no short-circuit: every candidate is compared in constant time
            ok |= hmac.compare_digest(c, self.key)
        return ok

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                m = await receive()
                if m["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif m["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = {k.lower(): v for k, v in scope["headers"]}
        if not self.authorized(headers):
            return await _send_json(send, 401, _error_body(
                "colabinet gateway: missing or invalid API key (use 'Authorization: Bearer KEY' or 'x-api-key')",
                "authentication_error"), ((b"www-authenticate", b"Bearer"),))
        path, method = scope["path"], scope["method"]
        if path != "/v1" and not path.startswith("/v1/"):
            return await _send_json(send, 404, _error_body(f"no route {method} {path}", "not_found"))
        body = await _read_body(receive)
        if body is None:
            return
        targets = await self.targets()
        if method == "GET" and path.rstrip("/") == "/v1/models":
            data = json.dumps({"object": "list", "data": [
                {"id": t.model_id, "object": "model", "created": t.created, "owned_by": "colabinet",
                 "session": t.name} for t in targets]}).encode()
            return await _send_json(send, 200, data)

        obj = None
        if body:
            with contextlib.suppress(ValueError, UnicodeDecodeError):
                obj = json.loads(body)
        model = obj.get("model") if isinstance(obj, dict) else None
        target, exact = pick(targets, model)
        if target is None:
            available = [t.model_id for t in targets]
            msg = (f"colabinet gateway: model {model!r} not found; available: {', '.join(available)}"
                   if available else "colabinet gateway: no ready session")
            return await _send_json(send, 404, _error_body(msg, "model_not_found", available=available))
        if not exact and target.alias and isinstance(obj, dict) and model != target.alias:
            # Matched by colab name or as the only session: send the engine its own model id.
            obj["model"] = target.alias
            body = json.dumps(obj).encode()
        stream = isinstance(obj, dict) and obj.get("stream") is True
        await self.forward(scope, receive, send, target, body, stream)

    async def forward(self, scope, receive, send, target: Target, body: bytes, stream: bool) -> None:
        host = f"127.0.0.1:{target.port}"
        query = scope.get("query_string", b"").decode("latin-1")
        url = f"http://{host}{scope['path']}" + (f"?{query}" if query else "")
        fwd = [(k, v) for k, v in scope["headers"] if k.lower() not in DROP_REQUEST]
        fwd.append((b"host", host.encode()))
        request = self.client.build_request(scope["method"], url, headers=fwd, content=body)

        async def watch() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass

        watcher = asyncio.create_task(watch())
        upstream = asyncio.create_task(self.client.send(request, stream=True))
        keepalive = SSE_KEEPALIVE if stream else JSON_KEEPALIVE
        early = False  # 200 already sent with our own headers
        resp: httpx.Response | None = None
        try:
            # Phase 1: wait for upstream headers, keeping the client connection alive.
            wait = self.grace
            while True:
                done, _ = await asyncio.wait({upstream, watcher}, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
                if upstream in done:
                    break
                if watcher in done:
                    return  # finally cancels the upstream request
                if not early:
                    ctype = b"text/event-stream; charset=utf-8" if stream else b"application/json"
                    await send({"type": "http.response.start", "status": 200,
                                "headers": [(b"content-type", ctype), (b"cache-control", b"no-cache"),
                                            (b"x-accel-buffering", b"no")]})
                    early = True
                await send({"type": "http.response.body", "body": keepalive, "more_body": True})
                wait = self.interval
            try:
                resp = upstream.result()
            except httpx.HTTPError as e:
                log.warning("upstream %s unreachable: %s", host, type(e).__name__)
                data = _error_body(f"colabinet gateway: session {target.name} unreachable ({type(e).__name__})",
                                   "bad_gateway")
                if not early:
                    return await _send_json(send, 502, data)
                await send({"type": "http.response.body", "body": b"data: " + data + b"\n\n" if stream else data,
                            "more_body": False})
                return

            # Phase 2: stream the upstream body.
            ctype = resp.headers.get("content-type", "")
            sse = stream if early else "text/event-stream" in ctype
            error_in_sse = early and stream and resp.status_code >= 400 and "text/event-stream" not in ctype
            if early:
                chunks = resp.aiter_bytes()  # our headers carry no content-encoding: decode
            else:
                out_headers = [(k, v) for k, v in resp.headers.raw if k.lower() not in HOP_BY_HOP]
                if sse and not any(k.lower() == b"x-accel-buffering" for k, _ in out_headers):
                    out_headers.append((b"x-accel-buffering", b"no"))
                await send({"type": "http.response.start", "status": resp.status_code, "headers": out_headers})
                chunks = resp.aiter_raw()
            if error_in_sse:
                data = await resp.aread()
                await send({"type": "http.response.body", "body": b"data: " + data.strip() + b"\n\n",
                            "more_body": True})
            else:
                it = chunks.__aiter__()
                tail = b"\n\n"  # keep-alive comments only go between complete SSE events
                while True:
                    nxt = asyncio.ensure_future(it.__anext__())
                    try:
                        while True:
                            done, _ = await asyncio.wait({nxt, watcher}, timeout=self.interval if sse else None,
                                                         return_when=asyncio.FIRST_COMPLETED)
                            if nxt in done or watcher in done:
                                break
                            if tail.endswith(b"\n\n") or tail.endswith(b"\r\n\r\n"):
                                await send({"type": "http.response.body", "body": SSE_KEEPALIVE, "more_body": True})
                        if nxt not in done:
                            return  # client went away
                    finally:
                        if not nxt.done():
                            nxt.cancel()
                            with contextlib.suppress(BaseException):
                                await nxt
                    try:
                        chunk = nxt.result()
                    except StopAsyncIteration:
                        break
                    except httpx.HTTPError as e:
                        log.warning("upstream %s stream broke: %s", host, type(e).__name__)
                        break
                    if chunk:
                        tail = (tail + chunk)[-4:]
                        await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            watcher.cancel()
            if not upstream.done():
                upstream.cancel()
                with contextlib.suppress(BaseException):
                    await upstream
            elif resp is None and not upstream.cancelled() and upstream.exception() is None:
                resp = upstream.result()
            if resp is not None:
                with contextlib.suppress(Exception):
                    await resp.aclose()


async def ui_context(settings, store=None) -> dict[str, Any]:
    """Settings-page data for the 「对外 API」 card."""
    port = settings.gateway_port
    base = settings.public_api_base.rstrip("/")
    enabled = bool(port)
    key = read_key(settings.gateway_key_path) if enabled else None
    model = "<模型 ID>"
    if store is not None and enabled:
        with contextlib.suppress(Exception):
            targets = await ready_targets(store)
            if targets:
                model = targets[0].model_id
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "你好"}]}, ensure_ascii=False)

    def curl(k: str) -> str:
        return (f"curl {base}/chat/completions \\\n  -H 'Authorization: Bearer {k}' \\\n"
                f"  -H 'Content-Type: application/json' \\\n  -d '{body}'")

    return {"enabled": enabled, "port": port, "listen": f"127.0.0.1:{port}" if enabled else "",
            "base": base, "key": key or "", "key_masked": mask(key), "key_path": str(settings.gateway_key_path),
            "curl": curl(mask(key) if key else "<密钥>"), "curl_full": curl(key or "<密钥>")}
