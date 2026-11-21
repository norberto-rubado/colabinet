"""Regression tests for review round 1 gateway / proxy issues (R1-02, R1-04, R1-05, R1-08 item 3, R1-09,
R1-14 item 4)."""

import asyncio
import errno
import json
import shlex
import shutil
import subprocess
import time
from types import SimpleNamespace

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.config import Settings
from colabinet.gateway import Gateway, StreamAborted, load_or_create_key, ui_context
from colabinet.net import first_free, start_server, stop_server
from colabinet.proxy import Route, SessionProxy

KEY = "test-gateway-key-r1-0123456789"
MiB = 1024 * 1024


class Store:
    """rows: (row, run) pairs; run carries alias/engine/spec_json like store.current_run()."""

    def __init__(self, rows):
        self.rows = rows

    async def live_sessions(self):
        return [row for row, _ in self.rows]

    async def current_run(self, sid):
        return next((run for row, run in self.rows if row["id"] == sid), None)


def llm(sid=1, alias="qwen-a", port=3081):
    return ({"id": sid, "colab_name": f"llm-{sid}", "proxy_port": port, "state": "ready", "ready_at": 100 + sid},
            {"alias": alias, "engine": "llamacpp", "spec_json": json.dumps({"kind": "text"})})


def comfy(sid=2, alias="comfy-image-uncensored", port=3080):
    return ({"id": sid, "colab_name": f"img-{sid}", "proxy_port": port, "state": "ready", "ready_at": 100 + sid},
            {"alias": alias, "engine": "comfyui", "spec_json": json.dumps({"kind": "image"})})


def auth():
    return {"Authorization": f"Bearer {KEY}"}


class Recorder:
    """MockTransport handler for the session proxy behind the gateway."""

    def __init__(self, respond=None):
        self.requests: list[httpx.Request] = []
        self.respond = respond or (lambda req: httpx.Response(
            200, stream=httpx.ByteStream(json.dumps({"ok": True}).encode()),
            headers={"content-type": "application/json"}))

    async def __call__(self, req):
        await req.aread()
        self.requests.append(req)
        out = self.respond(req)
        return await out if asyncio.iscoroutine(out) else out


def gateway(rows, handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gw = Gateway(SimpleNamespace(store=Store(rows)), KEY, client=client, cache_s=0, **kw)
    return gw, client


def asgi(gw):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="https://llm.example.com")


async def call(gw, path, chunks, method="POST", headers=()):
    """Drive the ASGI app directly; returns (sent messages, number of receive() calls that delivered body)."""
    sent, state = [], {"i": 0}
    done = asyncio.Event()

    async def receive():
        if state["i"] < len(chunks):
            state["i"] += 1
            return {"type": "http.request", "body": chunks[state["i"] - 1], "more_body": state["i"] < len(chunks)}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)
        if msg["type"] == "http.response.body" and not msg.get("more_body", False):
            done.set()

    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(b"authorization", f"Bearer {KEY}".encode()), *headers]}
    try:
        await gw(scope, receive, send)
    finally:
        done.set()
    return sent, state["i"]


def body_of(sent):
    return b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


# --- R1-02 -------------------------------------------------------------------------------------------------------
async def test_llm_request_with_only_comfy_ready_is_not_forwarded():
    """R1-02: with only a ComfyUI session ready, /v1/chat/completions is a 404 model_not_found, never sent to it."""
    rec = Recorder()
    gw, client = gateway([comfy()], rec)
    async with asgi(gw) as c:
        r1 = await c.post("/v1/chat/completions", headers=auth(), json={"model": "qwen-x", "messages": []})
        r2 = await c.post("/v1/chat/completions", headers=auth(),
                          json={"model": "comfy-image-uncensored", "messages": []})
        r3 = await c.post("/v1/embeddings", headers=auth(), json={"input": "hi"})
    for r in (r1, r2, r3):
        assert r.status_code == 404 and r.json()["error"]["type"] == "model_not_found"
        assert r.json()["error"]["available"] == []
    assert rec.requests == []
    await gw.aclose()
    await client.aclose()


async def test_llm_fallback_ignores_coexisting_comfy_session():
    """R1-02: one LLM + one ComfyUI session: the single-LLM fallback still works, media aliases are not LLM models,
    and the 404 'available' list only names text models."""
    rec = Recorder()
    gw, client = gateway([llm(port=3081), comfy(port=3080)], rec)
    async with asgi(gw) as c:
        r1 = await c.post("/v1/chat/completions", headers=auth(), json={"model": "gpt-4o", "messages": []})
        r2 = await c.post("/v1/chat/completions", headers=auth(),
                          json={"model": "comfy-image-uncensored", "messages": []})
        models = await c.get("/v1/models", headers=auth())
    # The only text session answers anything; a media alias falls back to it with the engine's own model id.
    assert r1.status_code == 200 and r2.status_code == 200
    assert [req.url.port for req in rec.requests] == [3081, 3081]
    assert [json.loads(req.content)["model"] for req in rec.requests] == ["qwen-a", "qwen-a"]
    assert {m["id"] for m in models.json()["data"]} == {"qwen-a", "comfy-image-uncensored"}  # listing unchanged
    await gw.aclose()
    await client.aclose()

    rec = Recorder()
    gw, client = gateway([llm(1, "qwen-a", 3081), llm(3, "qwen-b", 3082), comfy(port=3080)], rec)
    async with asgi(gw) as c:
        r = await c.post("/v1/chat/completions", headers=auth(),
                         json={"model": "comfy-image-uncensored", "messages": []})
    assert r.status_code == 404 and set(r.json()["error"]["available"]) == {"qwen-a", "qwen-b"}
    assert "comfy-image-uncensored" not in r.json()["error"]["message"].split("available:")[1]
    assert rec.requests == []
    await gw.aclose()
    await client.aclose()


async def test_ui_context_example_uses_text_session(tmp_path):
    """R1-02: the settings-page /chat/completions example names a text model, or the placeholder if none."""
    s = Settings(data_dir=tmp_path)
    load_or_create_key(s.gateway_key_path)
    ctx = await ui_context(s, Store([llm(1, "qwen-a"), comfy(2)]))  # comfy is the newest ready session
    assert "qwen-a" in ctx["curl"] and "comfy-image-uncensored" not in ctx["curl"]
    ctx = await ui_context(s, Store([comfy(2)]))
    assert "<模型 ID>" in ctx["curl"] and "comfy" not in ctx["curl"]


@pytest.mark.parametrize("status, useful", [(404, False), (500, False), (200, True)])
async def test_session_proxy_error_answer_is_not_activity(status, useful):
    """R1-02: a 4xx/5xx upstream answer to an inference path (e.g. ComfyUI's 404 for a chat request) does not reset
    the idle clock; a successful one still does."""
    def vm(req):
        return httpx.Response(status, stream=httpx.ByteStream(b"404: Not Found" if status == 404 else b"{}"),
                              headers={"content-type": "text/plain"})

    activity = Activity(1000.0)
    records = []

    async def record(rec, run_id):
        records.append(rec)

    async with httpx.AsyncClient(transport=httpx.MockTransport(vm)) as vm_http:
        proxy = SessionProxy(3080, lambda: Route("forward", 8188, run_id=1), record, activity, vm_http,
                             lambda: 5000.0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="http://127.0.0.1:3080") as c:
            r = await c.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == status and records[0].status == status
    assert activity.last == (5000.0 if useful else 1000.0) and not activity._inflight


async def test_session_proxy_does_not_record_media_poll_rows():
    """R1-14 (lead follow-up): MediaService status polling (UA colabinet-poll) is forwarded but never stored as a
    request row, so an 80 s video no longer writes ~160 rows that push real requests out of the session page."""
    def vm(req):
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"), headers={"content-type": "application/json"})

    records = []

    async def record(rec, run_id):
        records.append(rec)

    async with httpx.AsyncClient(transport=httpx.MockTransport(vm)) as vm_http:
        proxy = SessionProxy(3080, lambda: Route("forward", 8188, run_id=1), record, Activity(1000.0), vm_http,
                             lambda: 5000.0)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy), base_url="http://127.0.0.1:3080") as c:
            polled = await c.get("/history/abc", headers={"User-Agent": "colabinet-poll"})
            real = await c.post("/prompt", json={}, headers={"User-Agent": "colabinet-gateway"})
    assert polled.status_code == real.status_code == 200
    assert [r.path for r in records] == ["/prompt"]


# --- R1-04 -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", ["/v1/../props", "/v1/../slots", "/v1/./chat/completions", "/v1/a/../../metrics",
                                  "/v1/..", "/v1/models/.", "/comfy/comfy-image-uncensored/../props",
                                  "/comfy/comfy-image-uncensored/./queue", "/comfy/../v1/models",
                                  "/v1/a\\..\\props", "/comfy/comfy-image-uncensored/view\\x"])
async def test_dot_segments_and_backslashes_rejected(path):
    """R1-04: a decoded path with '.'/'..' segments or a backslash is a 400 before routing; nothing goes upstream."""
    rec = Recorder()
    gw, client = gateway([llm(), comfy()], rec)
    sent, _ = await call(gw, path, [b""], method="GET")
    assert sent[0]["status"] == 400
    err = json.loads(body_of(sent))["error"]
    assert err["type"] == "invalid_request_error" and err["message"] == "invalid path"
    assert rec.requests == []
    await gw.aclose()
    await client.aclose()


async def test_dot_check_keeps_normal_paths_and_dots_inside_segments(tmp_path):
    """R1-04: dots inside a segment (model ids, file names) are fine; only whole '.'/'..' segments are refused."""
    rec = Recorder()
    gw, client = gateway([llm(alias="qwen3.6-flash"), comfy()], rec)
    gw.sup.s, gw.sup._activity = Settings(data_dir=tmp_path), {}
    sent1, _ = await call(gw, "/comfy/comfy-image-uncensored/view", [b""], method="GET")
    sent2, _ = await call(gw, "/v1/chat/completions", [b'{"model": "qwen3.6-flash"}'])
    sent3, _ = await call(gw, "/v1/files/..file/content", [b""], method="GET")
    assert sent1[0]["status"] == 200 and sent2[0]["status"] == 200 and sent3[0]["status"] == 404
    assert [req.url.path for req in rec.requests] == ["/view", "/v1/chat/completions"]
    await gw.aclose()
    await client.aclose()


# --- R1-05 -------------------------------------------------------------------------------------------------------
async def test_gateway_body_over_64_mib_is_413_and_stops_reading():
    """R1-05: a JSON route stops reading once the body passes 64 MiB and answers 413 with a JSON error."""
    rec = Recorder()
    gw, client = gateway([comfy()], rec)
    chunk = b"A" * MiB
    sent, reads = await call(gw, "/v1/images/generations", [chunk] * 100,
                             headers=[(b"content-type", b"application/json")])
    assert sent[0]["status"] == 413 and reads == 65  # the 65th MiB crosses the limit; the rest is never read
    err = json.loads(body_of(sent))["error"]
    assert err["type"] == "request_too_large" and "64 MiB" in err["message"]
    # Content-Length announcing too much is refused before reading anything.
    sent, reads = await call(gw, "/v1/chat/completions", [b"{}"],
                             headers=[(b"content-length", str(64 * MiB + 1).encode())])
    assert sent[0]["status"] == 413 and reads == 0
    assert rec.requests == []
    await gw.aclose()
    await client.aclose()


async def test_gateway_comfy_upload_limit_is_128_mib():
    """R1-05: /comfy/*/upload/* may carry up to 128 MiB (more than the JSON limit) and is refused above it."""
    rec = Recorder()
    gw, client = gateway([comfy()], rec)
    chunk = b"B" * MiB
    sent, _ = await call(gw, "/comfy/comfy-image-uncensored/upload/image", [chunk] * 70)
    assert sent[0]["status"] == 200 and len(rec.requests[0].content) == 70 * MiB
    rec.requests.clear()
    sent, reads = await call(gw, "/comfy/comfy-image-uncensored/upload/image", [chunk] * 140)
    assert sent[0]["status"] == 413 and reads == 129 and "128 MiB" in json.loads(body_of(sent))["error"]["message"]
    # Other raw ComfyUI routes keep the JSON limit.
    sent, reads = await call(gw, "/comfy/comfy-image-uncensored/prompt", [chunk] * 70)
    assert sent[0]["status"] == 413 and reads == 65
    assert rec.requests == []
    await gw.aclose()
    await client.aclose()


async def test_session_proxy_body_limit():
    """R1-05: the per-session proxy also refuses bodies past 64 MiB (128 MiB for /upload/*) with a 413."""
    hits = []

    def vm(req):
        hits.append(req.url.path)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(vm)) as vm_http:
        async def record(rec, run_id):
            pass

        proxy = SessionProxy(3080, lambda: Route("forward", 8188, run_id=1), record, Activity(), vm_http, time.time)
        host = [(b"host", b"127.0.0.1:3080")]
        sent, reads = await call(proxy, "/prompt", [b"C" * MiB] * 80, headers=host)
        assert sent[0]["status"] == 413 and reads == 65
        assert json.loads(body_of(sent))["error"]["type"] == "request_too_large"
        sent, _ = await call(proxy, "/upload/image", [b"C" * MiB] * 80, headers=host)
        assert sent[0]["status"] == 200
        sent, reads = await call(proxy, "/upload/image", [b"C" * MiB] * 130, headers=host)
        assert sent[0]["status"] == 413 and reads == 129
    assert hits == ["/upload/image"]


# --- R1-09 -------------------------------------------------------------------------------------------------------
class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, first: bytes):
        self.first = first

    async def __aiter__(self):
        if self.first:
            yield self.first
        raise httpx.ReadError("tunnel dropped")


def llm_gateway(respond):
    return gateway([llm()], Recorder(respond), grace=0.01, interval=0.01)


async def slow(response, delay=0.05):
    await asyncio.sleep(delay)
    return response


async def test_mid_body_break_with_content_length_aborts_connection():
    """R1-09: a non-SSE body breaking mid-way (upstream Content-Length passed through) is not ended cleanly: the
    gateway raises so the server drops the connection."""
    gw, client = llm_gateway(lambda req: httpx.Response(200, stream=BrokenStream(b'{"id":"1","choices":['),
                                                        headers={"content-type": "application/json",
                                                                 "content-length": "100"}))
    with pytest.raises(StreamAborted):
        await call(gw, "/v1/chat/completions", [b'{"model": "qwen-a"}'])
    await gw.aclose()
    await client.aclose()

    # Early 200 (no Content-Length of ours) for a non-stream request: also aborted, not a clean truncated JSON.
    gw, client = llm_gateway(lambda req: slow(httpx.Response(200, stream=BrokenStream(b'{"id":'),
                                                            headers={"content-type": "application/json"})))
    with pytest.raises(StreamAborted):
        await call(gw, "/v1/chat/completions", [b'{"model": "qwen-a"}'])
    await gw.aclose()
    await client.aclose()


async def test_mid_body_break_resets_real_connection():
    """R1-09: through uvicorn, a chunked (early 200) body that breaks mid-way reaches the client as a broken
    response instead of a clean, truncated 200."""
    gw, client = llm_gateway(lambda req: slow(httpx.Response(200, stream=BrokenStream(b'{"id":"1","choices":['),
                                                            headers={"content-type": "application/json"})))
    port = first_free(28600, 28899)
    server, task = await start_server(gw, port)
    try:
        async with httpx.AsyncClient() as c:
            with pytest.raises(httpx.HTTPError):
                await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers=auth(), json={"model": "qwen-a"})
    finally:
        await stop_server(server, task)
        await gw.aclose()
        await client.aclose()


@pytest.mark.parametrize("delay", [0, 0.05])  # passthrough SSE headers, and our own early 200 SSE headers
async def test_sse_mid_stream_break_sends_error_event(delay):
    """R1-09: an SSE stream breaking mid-way ends with a final `data:` error event (after closing the half-sent
    event) instead of looking like a complete stream."""
    first = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: {"choi'
    gw, client = llm_gateway(lambda req: slow(httpx.Response(200, stream=BrokenStream(first),
                                                            headers={"content-type": "text/event-stream"}), delay))
    sent, _ = await call(gw, "/v1/chat/completions", [b'{"model": "qwen-a", "stream": true}'])
    text = body_of(sent).decode()
    assert sent[-1]["more_body"] is False
    last = [e for e in text.split("\n\n") if e.strip()][-1]
    assert last.startswith("data: ") and json.loads(last[6:])["error"] == {
        "message": "upstream stream interrupted", "type": "bad_gateway"}
    assert '{"choi\n\ndata: {"error"' in text  # the partial event is terminated before ours
    await gw.aclose()
    await client.aclose()


async def test_early_sse_error_body_read_failure_is_caught():
    """R1-09: when reading the upstream error body after an early SSE 200 fails, the client gets an SSE error event."""
    gw, client = llm_gateway(lambda req: slow(httpx.Response(500, stream=BrokenStream(b""),
                                                            headers={"content-type": "application/json"})))
    sent, _ = await call(gw, "/v1/chat/completions", [b'{"model": "qwen-a", "stream": true}'])
    text = body_of(sent).decode()
    assert sent[0]["status"] == 200 and sent[-1]["more_body"] is False
    event = text.replace(": keepalive\n\n", "")
    assert event.startswith("data: ") and json.loads(event[6:])["error"]["type"] == "bad_gateway"
    await gw.aclose()
    await client.aclose()


async def test_early_sse_multiline_error_is_framed_per_line():
    """R1-09: a multi-line upstream error after an early SSE 200 gets `data: ` on every line."""
    gw, client = llm_gateway(lambda req: slow(httpx.Response(500, stream=httpx.ByteStream(b'{\n  "error": "boom"\n}'),
                                                            headers={"content-type": "application/json"})))
    sent, _ = await call(gw, "/v1/chat/completions", [b'{"model": "qwen-a", "stream": true}'])
    event = body_of(sent).decode().replace(": keepalive\n\n", "")
    assert event == 'data: {\ndata:   "error": "boom"\ndata: }\n\n'
    data = "\n".join(line[6:] for line in event.strip().split("\n"))  # what an SSE parser reassembles
    assert json.loads(data) == {"error": "boom"}
    await gw.aclose()
    await client.aclose()


async def test_early_sse_wraps_non_sse_success_body():
    """R1-09: a 200 application/json answer to a "stream": true request after early SSE headers is sent as an SSE
    event, not as raw JSON inside text/event-stream."""
    plain = httpx.ByteStream(b'{"object":"list","data":[]}')
    gw, client = llm_gateway(lambda req: slow(httpx.Response(200, stream=plain,
                                                            headers={"content-type": "application/json"})))
    sent, _ = await call(gw, "/v1/embeddings", [b'{"model": "qwen-a", "stream": true}'])
    assert b"text/event-stream" in dict(sent[0]["headers"])[b"content-type"]
    event = body_of(sent).decode().replace(": keepalive\n\n", "")
    assert event == 'data: {"object":"list","data":[]}\n\n'
    await gw.aclose()
    await client.aclose()


# --- R1-08 (item 3) ----------------------------------------------------------------------------------------------
def media_gateway(tmp_path):
    sup = SimpleNamespace(store=Store([]), s=Settings(data_dir=tmp_path), _activity={})
    return Gateway(sup, KEY, client=httpx.AsyncClient(), grace=0.01, interval=0.01)


async def failing(delay, exc):
    await asyncio.sleep(delay)
    raise exc


@pytest.mark.parametrize("delay", [0, 0.05])
async def test_media_json_oserror_becomes_json_error(tmp_path, delay):
    """R1-08: an OSError (ENOSPC saving a url image) becomes a JSON error object; after an early 200 the body is
    still closed."""
    gw = media_gateway(tmp_path)
    sent = []

    async def send(msg):
        sent.append(msg)

    task = asyncio.create_task(failing(delay, OSError(errno.ENOSPC, "No space left on device")))
    await asyncio.sleep(0)
    await gw.media_json(send, task)
    body = body_of(sent)
    err = json.loads(body)["error"]
    assert err["type"] == "media_error" and "No space left on device" in err["message"]
    assert sent[0]["status"] == (200 if delay else 500)
    assert sent[-1]["type"] == "http.response.body" and sent[-1].get("more_body", False) is False
    if delay:
        assert body.startswith(b" ")  # keep-alive whitespace came first
    await gw.client.aclose()


async def test_media_json_cancelled_job_closes_body(tmp_path):
    """R1-08: if only the image job is cancelled (service shutdown) after an early 200, the body ends with an error."""
    gw = media_gateway(tmp_path)
    sent = []

    async def send(msg):
        sent.append(msg)

    task = asyncio.create_task(asyncio.sleep(10))
    asyncio.get_running_loop().call_later(0.05, task.cancel)
    await gw.media_json(send, task)
    assert sent[0]["status"] == 200 and sent[-1]["more_body"] is False
    assert json.loads(body_of(sent))["error"]["type"] == "media_error"
    await gw.client.aclose()


async def test_media_json_request_cancelled_closes_body_then_reraises(tmp_path):
    """R1-08: when the request handler itself is cancelled after an early 200, the body is closed and the
    CancelledError propagates."""
    gw = media_gateway(tmp_path)
    sent = []

    async def send(msg):
        sent.append(msg)

    job = asyncio.create_task(asyncio.sleep(10))
    handler = asyncio.create_task(gw.media_json(send, job))
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert sent[0]["status"] == 200 and sent[-1]["more_body"] is False
    assert json.loads(body_of(sent))["error"]["type"] == "media_error"
    job.cancel()
    await gw.client.aclose()


# --- R1-14 (item 4) ----------------------------------------------------------------------------------------------
async def test_ui_context_curl_is_shell_safe(tmp_path):
    """R1-14: an alias containing a single quote cannot break the copied curl command."""
    s = Settings(data_dir=tmp_path)
    key = load_or_create_key(s.gateway_key_path)
    ctx = await ui_context(s, Store([llm(1, "it's-a-'model'")]))
    for cmd, k in ((ctx["curl"], None), (ctx["curl_full"], key)):
        argv = shlex.split(cmd.replace("\\\n", " "))
        assert argv[0] == "curl" and argv[1] == "https://llm.111649.xyz/v1/chat/completions"
        payload = json.loads(argv[argv.index("-d") + 1])
        assert payload["model"] == "it's-a-'model'" and payload["messages"][0]["content"] == "你好"
        if k:
            assert argv[argv.index("-H") + 1] == f"Authorization: Bearer {k}"
    if shutil.which("sh"):  # the real shell agrees
        out = subprocess.run(["sh", "-c", "curl() { printf '%s\\n' \"$@\"; }\n" + ctx["curl_full"]],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        assert json.loads(out[out.index("-d") + 1])["model"] == "it's-a-'model'"
