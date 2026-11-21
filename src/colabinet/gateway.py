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
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import httpx

from .proxy import MAX_BODY_BYTES, MAX_UPLOAD_BYTES, BodyTooLarge, bad_path, read_body, wire_path  # noqa: F401 (bad_path re-exported)

log = logging.getLogger("colabinet.gateway")
COMFY_UPLOAD_PREFIXES = ("/upload/", "/api/upload/")  # upstream ComfyUI paths that get the 128 MiB upload limit

GRACE_S = 20.0
INTERVAL_S = 15.0
CACHE_S = 2.0
SSE_KEEPALIVE = b": keepalive\n\n"
JSON_KEEPALIVE = b" "

HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailers",
              b"transfer-encoding", b"upgrade", b"host"}
# Never forwarded: our own credential, the browser Origin (the session proxy rejects foreign origins),
# content-length (httpx sets it for the possibly rewritten body), and MediaService's internal marker (R3-02: a public
# client must not make its own POST /prompt activity-neutral, nor pose as MediaService).
DROP_REQUEST = HOP_BY_HOP | {b"authorization", b"x-api-key", b"origin", b"referer", b"content-length", b"cookie",
                             b"x-colabinet-media"}


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
    engine: str = "llamacpp"
    kind: str = "text"

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
        spec = json.loads((run or {}).get("spec_json") or "{}")
        out.append(Target(sid=row["id"], name=row.get("colab_name") or f"session-{row['id']}",
                          alias=(run or {}).get("alias") or None, port=int(row["proxy_port"]),
                          created=int(row.get("ready_at") or row.get("created_at") or 0),
                          engine=(run or {}).get("engine", "llamacpp"), kind=spec.get("kind", "text")))
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


def text_targets(targets: list[Target]) -> list[Target]:
    """Sessions that can answer LLM requests: ComfyUI media sessions only speak /comfy and the media routes."""
    return [t for t in targets if t.engine != "comfyui" and t.kind == "text"]


class ComfyPathError(Exception):
    """A /comfy/ request path that cannot be split into model and upstream path unambiguously (R3-03)."""


def comfy_split(path: str, wire: str) -> tuple[str, str, str]:
    """(model, decoded upstream path, raw upstream path) of a /comfy/{model}/{rest} request, split on the decoded
    path and checked against the raw one, so what is validated (routing, '//', body limit) is what is forwarded.
    The raw path must start with the literal '/comfy/' and the model segment must not contain an encoded '/': model
    ids containing '/' are not supported on /comfy (R3-03)."""
    parts = path.split("/", 3)  # ['', 'comfy', model, rest]
    raw = wire.split("/", 3)
    if not wire.startswith("/comfy/"):  # e.g. /comfy%2F<model>/...: the raw and decoded splits would disagree
        raise ComfyPathError("invalid ComfyUI path: use /comfy/{model}/{path}")
    if "%2f" in raw[2].lower():
        raise ComfyPathError("invalid ComfyUI path: model ids containing '/' are not supported on /comfy; "
                             "use the session name instead")
    if len(raw) != len(parts) or unquote(raw[2]) != parts[2]:
        raise ComfyPathError("invalid ComfyUI path: use /comfy/{model}/{path}")
    model = parts[2]
    upstream, raw_upstream = "/" + (parts[3] if len(parts) > 3 else ""), "/" + (raw[3] if len(raw) > 3 else "")
    if not model or upstream.startswith("//") or raw_upstream.startswith("//") or unquote(raw_upstream) != upstream:
        raise ComfyPathError("invalid ComfyUI path: use /comfy/{model}/{path}")
    return model, upstream, raw_upstream


def body_limit(comfy_upstream: str | None) -> int:
    """128 MiB only for a validated /comfy/{model}/upload/... (or api/upload) upstream path, as split by
    comfy_split(); 64 MiB for everything else (R3-03: the limit follows the path that is forwarded)."""
    if comfy_upstream is not None and comfy_upstream.startswith(COMFY_UPLOAD_PREFIXES):
        return MAX_UPLOAD_BYTES
    return MAX_BODY_BYTES


# --- ASGI helpers ------------------------------------------------------------------------------------------------
class StreamAborted(Exception):
    """The upstream body broke after our headers went out and the response cannot be ended honestly: raising
    makes the server drop the connection so the client sees an incomplete body instead of a clean end."""


def _aborted(exc: BaseException | None) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, StreamAborted):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


class StreamAbortedFilter(logging.Filter):
    """Drops uvicorn's ERROR "Exception in ASGI application" traceback (~55 lines) for StreamAborted: it is the
    intended way to drop the connection and forward() already logged one WARNING line for it (R2-05)."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info
        if isinstance(exc, tuple):
            exc = exc[1]
        return not (isinstance(exc, BaseException) and _aborted(exc))


def install_log_filter(logger: logging.Logger | None = None) -> None:
    """Idempotent: one StreamAbortedFilter on uvicorn's error logger (it logs unhandled ASGI exceptions)."""
    logger = logger or logging.getLogger("uvicorn.error")
    if not any(isinstance(f, StreamAbortedFilter) for f in logger.filters):
        logger.addFilter(StreamAbortedFilter())


def _sse_data(data: bytes) -> bytes:
    """One SSE event carrying `data`, every line prefixed (unprefixed continuation lines are dropped by clients)."""
    lines = data.strip().splitlines() or [b""]
    return b"".join(b"data: " + line + b"\n" for line in lines) + b"\n"


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
        install_log_filter()  # before any uvicorn server serves this app (R2-05)

    async def aclose(self) -> None:
        if getattr(self.sup, "_media", None):
            await self.sup._media.close()
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
        # Before routing: httpx would resolve /v1/../x to /x upstream, and a double-encoded /v1/%252e%252e/x would be
        # decoded again by the session proxy (R2-01). `path` (decoded, validated) drives routing, the body limit and
        # media routes; upstream gets the raw `wire` path so %2F, %3F and %23 are never reinterpreted.
        wire = wire_path(scope)
        if wire is None:  # wire_path() also applies bad_path() to the decoded path
            return await _send_json(send, 400, _error_body("invalid path", "invalid_request_error"))
        if path != "/v1" and not path.startswith(("/v1/", "/comfy/")):
            return await _send_json(send, 404, _error_body(f"no route {method} {path}", "not_found"))
        comfy: tuple[str, str, str] | None = None
        if path.startswith("/comfy/"):
            try:
                comfy = comfy_split(path, wire)
            except ComfyPathError as exc:  # before the body: a shifted split must not borrow the upload limit
                return await _send_json(send, 400, _error_body(str(exc), "invalid_request_error"))
        try:
            body = await read_body(receive, body_limit(comfy[1] if comfy else None), headers.get(b"content-length"))
        except BodyTooLarge as e:
            return await _send_json(send, 413, _error_body(f"colabinet gateway: {e}", "request_too_large"))
        if body is None:
            return
        targets = await self.targets()
        if method == "GET" and path.rstrip("/") == "/v1/models":
            data = json.dumps({"object": "list", "data": [
                {"id": t.model_id, "object": "model", "created": t.created, "owned_by": "colabinet",
                 "session": t.name, "kind": t.kind} for t in targets]}).encode()
            return await _send_json(send, 200, data)

        if path.startswith(("/v1/videos/", "/v1/files/")) or path in ("/v1/images/generations", "/v1/videos"):
            return await self.media_route(scope, receive, send, targets, body)
        if comfy is not None:
            # Split once on the decoded path and checked against the raw one (comfy_split, R3-03).
            model, upstream_path, wire_upstream = comfy
            target = next((t for t in targets if t.engine == "comfyui" and model in (t.model_id, t.name)), None)
            if target is None:
                return await _send_json(send, 404, _error_body("ComfyUI session not found", "model_not_found"))
            raw_scope = {**scope, "path": upstream_path, "comfy_raw": True}
            return await self.forward(raw_scope, receive, send, target, body, False, wire_upstream)

        obj = None
        if body:
            with contextlib.suppress(ValueError, UnicodeDecodeError):
                obj = json.loads(body)
        model = obj.get("model") if isinstance(obj, dict) else None
        llm = text_targets(targets)
        target, exact = pick(llm, model)
        if target is None:
            available = [t.model_id for t in llm]
            msg = (f"colabinet gateway: model {model!r} not found; available: {', '.join(available)}"
                   if available else "colabinet gateway: no ready text session")
            return await _send_json(send, 404, _error_body(msg, "model_not_found", available=available))
        if not exact and target.alias and isinstance(obj, dict) and model != target.alias:
            # Matched by colab name or as the only session: send the engine its own model id.
            obj["model"] = target.alias
            body = json.dumps(obj).encode()
        stream = isinstance(obj, dict) and obj.get("stream") is True
        await self.forward(scope, receive, send, target, body, stream, wire)

    async def media_route(self, scope, receive, send, targets, raw):
        from pydantic import ValidationError
        from starlette.responses import FileResponse

        from .comfy_graphs import ImageRequest, VideoRequest
        from .media import MediaError, media_service

        path, method = scope["path"], scope["method"]
        try:
            service = media_service(self.sup, self.client)
            if method == "GET" and path.startswith("/v1/files/"):
                parts = path.split("/")
                if len(parts) != 5 or parts[4] != "content":
                    raise MediaError("未找到文件", 404)
                file, ctype = service.files.content(parts[3])
                return await FileResponse(file, media_type=ctype, headers={"Cache-Control": "private, no-store"})(
                    scope, receive, send)
            if method == "GET" and path.startswith("/v1/videos/"):
                parts = path.split("/")
                if len(parts) == 4:
                    return await _send_json(send, 200, json.dumps(service.video_status(parts[3])).encode())
                if len(parts) != 5 or parts[4] != "content" or not parts[3].startswith("video_"):
                    raise MediaError("未找到视频", 404)
                file, ctype = service.files.content(parts[3])
                return await FileResponse(file, media_type=ctype, headers={"Cache-Control": "private, no-store"})(
                    scope, receive, send)
            if method != "POST" or path not in ("/v1/images/generations", "/v1/videos"):
                raise MediaError("未找到媒体路由", 404)
            schema = ImageRequest if path == "/v1/images/generations" else VideoRequest
            body = schema.model_validate_json(raw)
            kind = "image" if schema is ImageRequest else "video"
            # Media routing is exact: a typo must not silently dispatch to the only LLM session.
            target = next((t for t in targets if t.engine == "comfyui" and t.kind == kind
                           and body.model in (t.model_id, t.name)), None)
            if target is None:
                raise MediaError("没有匹配的已就绪 ComfyUI 媒体会话", 404)
            if isinstance(body, VideoRequest):
                result = await service.create_video(target, body)
                return await _send_json(send, 200, json.dumps(result).encode())
            # Cold model load can exceed Cloudflare's 100s. Valid leading JSON whitespace keeps it alive.
            # An owned task preserves activity tracking even if the HTTP client disconnects.
            task = service.spawn(service.images(target, body))
            return await self.media_json(send, task)
        except ValidationError as exc:
            return await _send_json(send, 400, _error_body(
                json.dumps(exc.errors(include_input=False, include_url=False), default=str), "invalid_request_error"))
        except MediaError as exc:
            return await _send_json(send, exc.status, _error_body(str(exc), "media_error"))

    async def media_json(self, send, task):
        from .media import MediaError

        early = False
        wait = self.grace
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=wait)
                if done:
                    break
                if not early:
                    await send({"type": "http.response.start", "status": 200, "headers": [
                        (b"content-type", b"application/json"), (b"cache-control", b"no-store"),
                        (b"x-accel-buffering", b"no")]})
                    early = True
                await send({"type": "http.response.body", "body": b" ", "more_body": True})
                wait = self.interval
            result = task.result()
            # URLs use the configured public base, not an untrusted Host header.
            base = self.sup.s.public_api_base.rstrip("/")
            for item in result["data"]:
                if "url" in item:
                    item["url"] = base + item["url"].removeprefix("/v1")
            data, status = json.dumps(result).encode(), 200
        except asyncio.CancelledError:
            task_now = asyncio.current_task()
            if task_now is not None and task_now.cancelling():
                # This request itself is being cancelled: end an already started body, then keep cancelling.
                if early:
                    with contextlib.suppress(Exception):
                        await send({"type": "http.response.body", "more_body": False,
                                    "body": _error_body("图像任务已取消", "media_error")})
                raise
            # Only the job was cancelled (service shutdown): answer like any other failed job.
            data, status = _error_body("图像任务已取消", "media_error"), 503
        except (MediaError, ValueError, KeyError) as exc:
            data, status = _error_body(str(exc), "media_error"), getattr(exc, "status", 502)
        except OSError as exc:  # e.g. ENOSPC while saving a response_format=url image
            log.warning("image result could not be saved: %s", exc)
            data, status = _error_body(f"保存生成结果失败：{exc.strerror or type(exc).__name__}", "media_error"), 500
        if not early:
            return await _send_json(send, status, data)
        await send({"type": "http.response.body", "body": data, "more_body": False})

    async def forward(self, scope, receive, send, target: Target, body: bytes, stream: bool, wire: str) -> None:
        """`wire`: the still percent-encoded upstream path (see wire_path), never the decoded scope["path"]."""
        host = f"127.0.0.1:{target.port}"
        query = scope.get("query_string", b"").decode("latin-1")
        url = f"http://{host}{wire}" + (f"?{query}" if query else "")
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
                if scope.get("comfy_raw"):
                    wait = self.interval
                    continue  # raw binary /view and uploads must stay byte-for-byte intact
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
                await send({"type": "http.response.body", "body": _sse_data(data) if stream else data,
                            "more_body": False})
                return

            # Phase 2: stream the upstream body.
            ctype = resp.headers.get("content-type", "")
            sse = stream if early else "text/event-stream" in ctype
            # Our early 200 promised an event stream: any other upstream body (an error, or plain JSON from an
            # endpoint that ignores "stream": true) has to travel as one SSE event or parsers drop it.
            wrap_in_sse = early and stream and "text/event-stream" not in ctype
            if early:
                chunks = resp.aiter_bytes()  # our headers carry no content-encoding: decode
            else:
                out_headers = [(k, v) for k, v in resp.headers.raw if k.lower() not in HOP_BY_HOP]
                if sse and not any(k.lower() == b"x-accel-buffering" for k, _ in out_headers):
                    out_headers.append((b"x-accel-buffering", b"no"))
                await send({"type": "http.response.start", "status": resp.status_code, "headers": out_headers})
                chunks = resp.aiter_raw()
            if wrap_in_sse:
                try:
                    data = await resp.aread()
                except httpx.HTTPError as e:
                    log.warning("upstream %s body broke: %s", host, type(e).__name__)
                    data = _error_body("upstream stream interrupted", "bad_gateway")
                await send({"type": "http.response.body", "body": _sse_data(data), "more_body": True})
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
                        if not sse or (not early and "content-length" in resp.headers):
                            # A clean end would pass the truncated body off as complete (and break a passed-through
                            # Content-Length): drop the connection instead. This WARNING is the only log line:
                            # StreamAbortedFilter hides uvicorn's traceback for it.
                            log.warning("upstream %s stream broke: %s; dropping the client connection",
                                        host, type(e).__name__)
                            raise StreamAborted(f"upstream {host} stream broke: {type(e).__name__}") from e
                        log.warning("upstream %s stream broke: %s", host, type(e).__name__)
                        # Close any half-sent event, then tell the client the stream did not finish.
                        done_event = tail.endswith((b"\n\n", b"\r\n\r\n"))
                        sep = b"" if done_event else b"\n" if tail.endswith(b"\n") else b"\n\n"
                        await send({"type": "http.response.body", "more_body": True, "body": sep + _sse_data(
                            _error_body("upstream stream interrupted", "bad_gateway"))})
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
            targets = text_targets(await ready_targets(store))  # the example calls /chat/completions
            if targets:
                model = targets[0].model_id
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "你好"}]}, ensure_ascii=False)

    def curl(k: str) -> str:
        # shlex.quote: an alias or key containing ' must not end the shell string early.
        return (f"curl {shlex.quote(base + '/chat/completions')} \\\n"
                f"  -H {shlex.quote('Authorization: Bearer ' + k)} \\\n"
                f"  -H 'Content-Type: application/json' \\\n  -d {shlex.quote(body)}")

    return {"enabled": enabled, "port": port, "listen": f"127.0.0.1:{port}" if enabled else "",
            "base": base, "key": key or "", "key_masked": mask(key), "key_path": str(settings.gateway_key_path),
            "curl": curl(mask(key) if key else "<密钥>"), "curl_full": curl(key or "<密钥>")}
