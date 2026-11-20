import asyncio
import contextlib
import json
import os
import stat
import time
from types import SimpleNamespace

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.config import Settings
from colabinet.gateway import Gateway, load_or_create_key, mask, ui_context
from colabinet.net import first_free, start_server, stop_server
from colabinet.proxy import ProxyManager, Route, SessionProxy
from colabinet.web.app import make_templates
from tests.fakes import fake_engine

KEY = "test-gateway-key-0123456789"


class FakeStore:
    def __init__(self, sessions: list[tuple[dict, str | None]]):
        self.rows = sessions  # list of (row, alias)

    async def live_sessions(self):
        return [row for row, _ in self.rows]

    async def current_run(self, sid):
        for row, alias in self.rows:
            if row["id"] == sid:
                return {"alias": alias} if alias is not None else None
        return None


def session(sid, name, port, alias, state="ready"):
    return ({"id": sid, "colab_name": name, "proxy_port": port, "state": state, "ready_at": 1_800_000_000 + sid},
            alias)


class Upstream:
    """Scriptable fake engine. Body knobs: x_wait (s before headers), x_gap (s between SSE events), x_n, x_status."""

    def __init__(self):
        self.hosts, self.paths, self.bodies, self.auth = [], [], [], []
        self.disconnected = asyncio.Event()
        self.port = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        headers = dict(scope["headers"])
        self.hosts.append(headers.get(b"host", b"").decode())
        self.auth.append((headers.get(b"authorization"), headers.get(b"x-api-key")))
        self.paths.append(scope["path"])
        raw = b""
        while True:
            m = await receive()
            raw += m.get("body", b"")
            if not m.get("more_body"):
                break
        req = json.loads(raw or b"{}")
        self.bodies.append(req)

        async def watch():
            while (await receive())["type"] != "http.disconnect":
                pass
            self.disconnected.set()

        async def sleep(seconds):
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.disconnected.wait(), seconds)
            if self.disconnected.is_set():
                raise asyncio.CancelledError

        watcher = asyncio.create_task(watch())
        try:
            await sleep(float(req.get("x_wait", 0)))
            status = int(req.get("x_status", 200))
            if status >= 400:
                data = json.dumps({"error": {"message": "boom", "type": "server_error"}}).encode()
                await send({"type": "http.response.start", "status": status,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": data})
                return
            if not req.get("stream"):
                data = json.dumps({"id": "c1", "object": "chat.completion", "model": req.get("model"),
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]})
                await send({"type": "http.response.start", "status": 200,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": data.encode()})
                return
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/event-stream")]})
            for i in range(int(req.get("x_n", 3))):
                if i:
                    await sleep(float(req.get("x_gap", 0)))
                chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i}"}}]}
                await send({"type": "http.response.body", "body": f"data: {json.dumps(chunk)}\n\n".encode(),
                            "more_body": True})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
        finally:
            watcher.cancel()


@pytest.fixture
async def upstream():
    app = Upstream()
    port = first_free(27000, 27499)
    server, task = await start_server(app, port)
    app.port = port
    yield app
    await stop_server(server, task)


async def serve_gateway(gw):
    port = first_free(27500, 27999)
    server, task = await start_server(gw, port)
    return port, server, task


@pytest.fixture
async def gateway(upstream):
    store = FakeStore([session(1, "llm-a", upstream.port, "qwen-a")])
    gw = Gateway(SimpleNamespace(store=store), KEY, grace=0.3, interval=0.2, cache_s=0)
    port, server, task = await serve_gateway(gw)
    yield SimpleNamespace(url=f"http://127.0.0.1:{port}", store=store, app=gw)
    await stop_server(server, task)
    await gw.aclose()


def asgi_client(gw, **kw):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="https://llm.example.com", **kw)


def auth(key=KEY):
    return {"Authorization": f"Bearer {key}"}


# --- key file ----------------------------------------------------------------------------------------------------
def test_key_file_created_0600_and_kept(tmp_path):
    path = tmp_path / "gateway.key"
    key = load_or_create_key(path)
    assert len(key) >= 40 and stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert load_or_create_key(path) == key
    assert mask(key).startswith(key[:4]) and key not in mask(key)


# --- auth --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"x-api-key": "nope"},
                                     {"Authorization": KEY}, {"Authorization": "Basic " + KEY}])
async def test_auth_rejected(headers):
    gw = Gateway(SimpleNamespace(store=FakeStore([])), KEY, cache_s=0)
    async with asgi_client(gw) as c:
        r = await c.get("/v1/models", headers=headers)
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    await gw.aclose()


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {KEY}"}, {"authorization": f"bearer {KEY}"},
                                     {"x-api-key": KEY}])
async def test_auth_accepted_any_host_and_origin(headers):
    gw = Gateway(SimpleNamespace(store=FakeStore([])), KEY, cache_s=0)
    async with asgi_client(gw) as c:
        r = await c.get("/v1/models", headers={**headers, "Origin": "https://evil.example", "Host": "llm.111649.xyz"})
    assert r.status_code == 200 and r.json() == {"object": "list", "data": []}
    await gw.aclose()


# --- models and routing ------------------------------------------------------------------------------------------
async def test_models_lists_ready_sessions_only():
    store = FakeStore([session(1, "llm-a", 3080, "qwen-a"), session(2, "llm-b", 3081, None),
                       session(3, "llm-c", 3082, "boot", state="setting_up"), session(4, "llm-d", None, "x")])
    gw = Gateway(SimpleNamespace(store=store), KEY, cache_s=0)
    async with asgi_client(gw) as c:
        r = await c.get("/v1/models", headers=auth())
    data = r.json()["data"]
    assert [m["id"] for m in data] == ["llm-b", "qwen-a"]
    assert all(m["object"] == "model" and m["owned_by"] == "colabinet" for m in data)
    await gw.aclose()


async def test_model_list_is_cached():
    store = FakeStore([session(1, "llm-a", 3080, "qwen-a")])
    clock = SimpleNamespace(t=0.0)
    gw = Gateway(SimpleNamespace(store=store), KEY, cache_s=2, now=lambda: clock.t)
    async with asgi_client(gw) as c:
        assert len((await c.get("/v1/models", headers=auth())).json()["data"]) == 1
        store.rows.append(session(2, "llm-b", 3081, "qwen-b"))
        clock.t = 1.0
        assert len((await c.get("/v1/models", headers=auth())).json()["data"]) == 1
        clock.t = 2.5
        assert len((await c.get("/v1/models", headers=auth())).json()["data"]) == 2
    await gw.aclose()


async def two_upstreams():
    a, b = Upstream(), Upstream()
    servers = []
    for app in (a, b):
        app.port = first_free(27000, 27499)
        servers.append(await start_server(app, app.port))
    return a, b, servers


async def test_routes_by_alias_and_by_name():
    a, b, servers = await two_upstreams()
    store = FakeStore([session(1, "llm-a", a.port, "qwen-a"), session(2, "llm-b", b.port, "qwen-b")])
    gw = Gateway(SimpleNamespace(store=store), KEY, cache_s=0)
    try:
        async with asgi_client(gw) as c:
            r1 = await c.post("/v1/chat/completions", headers=auth(), json={"model": "qwen-b", "messages": []})
            r2 = await c.post("/v1/chat/completions", headers={"x-api-key": KEY},
                              json={"model": "llm-a", "messages": []})
            r3 = await c.post("/v1/chat/completions", headers=auth(), json={"model": "nope", "messages": []})
            r4 = await c.post("/v1/chat/completions", headers=auth(), json={"messages": []})
        assert r1.status_code == 200 and b.bodies[0]["model"] == "qwen-b" and len(a.bodies) == 1
        # Matched by colab name: the engine still gets its own alias.
        assert r2.status_code == 200 and a.bodies[0]["model"] == "qwen-a"
        assert r3.status_code == 404 and r4.status_code == 404
        err = r3.json()["error"]
        assert err["type"] == "model_not_found" and set(err["available"]) == {"qwen-a", "qwen-b"}
        assert "qwen-a" in err["message"] and "qwen-b" in err["message"]
        assert len(a.bodies) == 1 and len(b.bodies) == 1
        # Our credential never reaches the engine.
        assert all(h == (None, None) for h in a.auth + b.auth)
    finally:
        for s, t in servers:
            await stop_server(s, t)
        await gw.aclose()


async def test_single_session_fallback_and_host_rewrite(gateway, upstream):
    async with asgi_client(gateway.app) as c:
        r1 = await c.post("/v1/chat/completions", headers={**auth(), "Host": "llm.111649.xyz"},
                          json={"model": "whatever", "messages": []})
        r2 = await c.post("/v1/embeddings?x=1", headers=auth(), json={"input": "hi"})
    assert r1.status_code == 200 and upstream.bodies[0]["model"] == "qwen-a"
    assert r2.status_code == 200  # the fake answers any path
    assert upstream.hosts == [f"127.0.0.1:{upstream.port}"] * 2
    assert upstream.paths == ["/v1/chat/completions", "/v1/embeddings"]


async def test_no_ready_session_is_404():
    gw = Gateway(SimpleNamespace(store=FakeStore([session(1, "a", 3080, "m", state="setting_up")])), KEY, cache_s=0)
    async with asgi_client(gw) as c:
        r = await c.post("/v1/chat/completions", headers=auth(), json={"model": "m"})
    assert r.status_code == 404 and r.json()["error"]["available"] == []
    await gw.aclose()


async def test_unreachable_session_is_502():
    port = first_free(26000, 26999)
    gw = Gateway(SimpleNamespace(store=FakeStore([session(1, "a", port, "m")])), KEY, cache_s=0)
    async with asgi_client(gw) as c:
        r = await c.post("/v1/chat/completions", headers=auth(), json={"model": "m"})
    assert r.status_code == 502 and r.json()["error"]["type"] == "bad_gateway"
    await gw.aclose()


# --- streaming ---------------------------------------------------------------------------------------------------
async def test_stream_passthrough_chunk_by_chunk(gateway, upstream):
    seen = []
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", gateway.url + "/v1/chat/completions", headers=auth(),
                            json={"model": "qwen-a", "stream": True, "x_n": 3, "x_gap": 0.05}) as r:
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
            async for chunk in r.aiter_raw():
                seen.append(chunk)
    text = b"".join(seen).decode()
    assert text.count("data: ") == 4 and text.endswith("data: [DONE]\n\n") and ": keepalive" not in text
    assert len(seen) >= 3  # not buffered into one body


async def test_early_200_keepalive_non_stream(gateway, upstream):
    async with httpx.AsyncClient() as c:
        r = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                         json={"model": "qwen-a", "x_wait": 0.9})
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.text.startswith(" ") and r.text.count(" ") >= 2
    assert json.loads(r.text)["object"] == "chat.completion"  # leading whitespace is valid JSON


async def test_early_200_keepalive_stream(gateway, upstream):
    async with httpx.AsyncClient() as c:
        r = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                         json={"model": "qwen-a", "stream": True, "x_wait": 0.9})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    head, _, rest = r.text.partition("data: ")
    assert head.count(": keepalive\n\n") >= 2 and head.replace(": keepalive\n\n", "") == ""
    assert rest.strip().endswith("[DONE]")


async def test_early_200_then_upstream_error_goes_in_body(gateway, upstream):
    async with httpx.AsyncClient() as c:
        r1 = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                          json={"model": "qwen-a", "x_wait": 0.5, "x_status": 500})
        r2 = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                          json={"model": "qwen-a", "x_wait": 0.5, "x_status": 500, "stream": True})
    assert r1.status_code == 200 and json.loads(r1.text)["error"]["message"] == "boom"
    assert r2.status_code == 200 and "data: " in r2.text
    assert json.loads(r2.text.split("data: ", 1)[1])["error"]["message"] == "boom"


async def test_fast_upstream_error_keeps_status(gateway, upstream):
    async with httpx.AsyncClient() as c:
        r = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                         json={"model": "qwen-a", "x_status": 500})
    assert r.status_code == 500 and r.json()["error"]["message"] == "boom"


async def test_mid_stream_sse_keepalive(gateway, upstream):
    async with httpx.AsyncClient() as c:
        r = await c.post(gateway.url + "/v1/chat/completions", headers=auth(),
                         json={"model": "qwen-a", "stream": True, "x_n": 2, "x_gap": 0.7})
    events = r.text.split("\n\n")
    assert events[0].startswith("data: ") and ": keepalive" in events[1]
    assert r.text.count(": keepalive\n\n") >= 2 and r.text.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize("knobs", [{"x_n": 50, "x_gap": 0.1}, {"x_wait": 5}])
async def test_client_disconnect_cancels_upstream(gateway, upstream, knobs):
    async with httpx.AsyncClient() as c:
        async with c.stream("POST", gateway.url + "/v1/chat/completions", headers=auth(),
                            json={"model": "qwen-a", "stream": True, **knobs}) as r:
            async for _ in r.aiter_raw():
                break
    await asyncio.wait_for(upstream.disconnected.wait(), 3)


async def test_through_real_session_proxy_records_stats():
    """Gateway -> SessionProxy (Host check + metering) -> engine: the request is recorded."""
    engine_port = first_free(24000, 24999)
    engine = await start_server(fake_engine.app, engine_port)
    records = []

    async def record(rec, run_id):
        records.append((rec, run_id))

    proxy_port = first_free(25000, 25999)
    http = httpx.AsyncClient(timeout=httpx.Timeout(5, read=None))
    manager = ProxyManager()
    await manager.bind(1, proxy_port, SessionProxy(proxy_port, lambda: Route("forward", engine_port, run_id=9),
                                                   record, Activity(), http, time.time))
    gw = Gateway(SimpleNamespace(store=FakeStore([session(1, "llm-a", proxy_port, "fake-model")])), KEY, cache_s=0)
    gport, server, task = await serve_gateway(gw)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{gport}/v1/chat/completions",
                             headers={**auth(), "Host": "llm.111649.xyz", "Origin": "https://cherry.example"},
                             json={"model": "fake-model", "stream": True, "x_tokens": 3})
            assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]")
            models = await c.get(f"http://127.0.0.1:{gport}/v1/models", headers={"x-api-key": KEY})
        assert models.json()["data"][0]["id"] == "fake-model"
        await asyncio.sleep(0.05)
        assert records and records[0][1] == 9 and records[0][0].completion_tokens == 3
    finally:
        await stop_server(server, task)
        await gw.aclose()
        await manager.close()
        await http.aclose()
        await stop_server(*engine)


# --- settings card -----------------------------------------------------------------------------------------------
def render_settings(gateway_ctx):
    st = {"config": {"daily_cap_cu": 0, "low_balance_cu": 10, "idle_min_default": 30, "notify": False},
          "gpu_types": [], "eligible_gpus": set(), "rates": [], "cli_version": None, "colab_auth": "oauth2",
          "colab_bin": "colab", "ssh_key": "k", "ssh_key_ok": True, "hf_token": False, "db_path": "d", "db_mb": 0,
          "stop_on_exit": False, "setup_limit_s": 1, "grace_s": 1, "proxy_port_start": 3080,
          "hidden_port_range": (1, 2)}
    return make_templates().env.get_template("pages/settings.html").render(
        st=st, csrf="c", active="settings", gateway=gateway_ctx,
        account={"state": "idle", "mode": "oauth2", "busy": False, "accounts": []},
        base={"balance": None, "burn_rate": 0, "live_count": 0, "alert_count": 0, "command_sessions": [],
              "daily": {"spend": 0, "cap": 10}})


async def test_settings_card_enabled(tmp_path):
    s = Settings(data_dir=tmp_path)
    key = load_or_create_key(s.gateway_key_path)
    ctx = await ui_context(s, FakeStore([session(1, "llm-a", 3080, "qwen-a")]))
    page = render_settings(ctx)
    assert "对外 API" in page and "https://llm.111649.xyz/v1" in page and "127.0.0.1:3079" in page
    assert mask(key) in page and "cbCopy" in page and "curl https://llm.111649.xyz/v1/chat/completions" in page
    assert "qwen-a" in ctx["curl"] and key not in ctx["curl"] and key in ctx["curl_full"]
    assert str(s.gateway_key_path) in page and "已关闭" not in page


async def test_settings_card_disabled(tmp_path):
    s = Settings(data_dir=tmp_path, gateway_port=0)
    page = render_settings(await ui_context(s))
    assert "对外 API" in page and "已关闭" in page and "gw-curl" not in page and "gw-key" not in page


def test_cli_flags(monkeypatch):
    from colabinet import __main__ as cli

    seen = {}

    def fake_settings(opts):
        seen["opts"] = opts
        raise SystemExit(0)

    monkeypatch.setattr(cli, "_settings", fake_settings)
    with pytest.raises(SystemExit):
        cli.main(["serve", "--gateway-port", "0", "--public-api-base", "https://x.example/v1"])
    assert seen["opts"].gateway_port == 0 and seen["opts"].public_api_base == "https://x.example/v1"
    with pytest.raises(SystemExit):
        cli.main(["serve"])
    assert seen["opts"].gateway_port == 3079 and seen["opts"].public_api_base == "https://llm.111649.xyz/v1"
