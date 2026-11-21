"""Regression tests for review round 2: R2-01 .. R2-05."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.alerts import Alerts, NullNotifier
from colabinet.cli_parse import Listing, Runtime, Usage
from colabinet.colab_cli import CmdResult
from colabinet.comfy_graphs import ImageRequest, VideoRequest
from colabinet.config import SEED_RATES, Settings
from colabinet.gateway import Gateway, StreamAborted, StreamAbortedFilter, install_log_filter
from colabinet.media import MediaError, MediaService
from colabinet.net import first_free, start_server, stop_server
from colabinet.presets import PresetError
from colabinet.proxy import ProxyManager, Route, SessionProxy, wire_path
from colabinet.supervisor import QUEUE_BUSY_MAX_S, QUEUE_GRACE_S, Supervisor
from colabinet.web.app import make_templates
from tests.unit.test_review_r1_gateway import KEY, BrokenStream, Store, auth, comfy, llm, slow
from tests.unit.test_review_r1_lifecycle import comfy_guard, run_until_stop, tunnel_down
from tests.unit.test_review_r1_media import DONE, IMAGE, VIDEO, Comfy, enospc, wait_video, world  # noqa: F401


# ------------------------------------------------------------------------------------------------------- R2-01
class Hop(httpx.AsyncBaseTransport):
    """One uvicorn hop without a socket: the raw request target is split and percent-decoded exactly like
    uvicorn's h11_impl (raw_path kept, path = unquote(raw_path))."""

    def __init__(self, app):
        self.app = app

    async def handle_async_request(self, request):
        raw_path, _, query = request.url.raw_path.partition(b"?")
        body = await request.aread()
        sent, delivered, done = [], False, asyncio.Event()

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await done.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            sent.append(msg)
            if msg["type"] == "http.response.body" and not msg.get("more_body", False):
                done.set()

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": request.method,
                 "scheme": "http", "path": unquote(raw_path.decode("ascii")), "raw_path": raw_path,
                 "query_string": query, "headers": [(k.lower(), v) for k, v in request.headers.raw]}
        try:
            await self.app(scope, receive, send)
        finally:
            done.set()
        return httpx.Response(sent[0]["status"], headers=sent[0]["headers"],
                              stream=httpx.ByteStream(b"".join(m.get("body", b"") for m in sent[1:])))


class Chain:
    """public client -> Gateway -> SessionProxy (one per session port) -> VM, every hop decoding like uvicorn.
    `vm` records what each VM finally received: (port, method, raw path, path, query)."""

    def __init__(self, rows):
        self.vm: list[tuple[int, str, bytes, str, str]] = []
        self.activity = {3081: Activity(100.0), 3080: Activity(100.0)}
        vms = {3081: 8091, 3080: 8090}

        def vm_handler(req):
            self.vm.append((req.url.port, req.method, req.url.raw_path.partition(b"?")[0], req.url.path,
                            req.url.query.decode()))
            return httpx.Response(200, stream=httpx.ByteStream(b'{"ok": true}'),
                                  headers={"content-type": "application/json"})

        self.vm_http = httpx.AsyncClient(transport=httpx.MockTransport(vm_handler))
        self.proxies = {port: SessionProxy(port, lambda vm_port=vm_port: Route("forward", vm_port, run_id=1),
                                           AsyncMock(), self.activity[port], self.vm_http, lambda: 500.0)
                        for port, vm_port in vms.items()}
        hops = {port: Hop(proxy) for port, proxy in self.proxies.items()}

        class Router(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return await hops[request.url.port].handle_async_request(request)

        self.gw_http = httpx.AsyncClient(transport=Router())
        self.gw = Gateway(SimpleNamespace(store=Store(rows)), KEY, client=self.gw_http, cache_s=0)
        self.public = httpx.AsyncClient(transport=Hop(self.gw), base_url="https://llm.example.com")

    async def close(self):
        await self.public.aclose()
        await self.gw.aclose()
        await self.gw_http.aclose()
        await self.vm_http.aclose()


@pytest.fixture
async def chain():
    c = Chain([llm(port=3081), comfy(port=3080)])
    yield c
    await c.close()


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/%252e%252e/props"), ("POST", "/v1/%252e%252e/slots?id_slot=0&action=erase"),
    ("GET", "/v1/%252E%252E/metrics"), ("POST", "/v1/%25252e%25252e/completion"),
    ("POST", "/comfy/comfy-image-uncensored/upload/%252e%252e/prompt"),
    ("GET", "/comfy/comfy-image-uncensored/%252e%252e/system_stats"),
    ("GET", "/comfy/comfy-image-uncensored/view%253ffilename=a.png"),
])
async def test_r2_01_double_encoding_is_refused_before_any_hop(chain, method, path):
    """R2-01: an encoded '%' (%25, double encoding) on /v1 and /comfy is a 400 invalid_request_error at the
    gateway; nothing reaches a session proxy or a VM."""
    r = await chain.public.request(method, path, headers=auth(), json={"model": "qwen-a"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert chain.vm == []
    assert chain.activity[3080].last == chain.activity[3081].last == 100.0


@pytest.mark.parametrize("path", ["/v1/%252e%252e/props", "/upload/%252e%252e/prompt", "/v1/%2e%2e/props",
                                  "/upload/%2E%2E/prompt", "/v1/a%5c..%5cprops", "/v1/%25"])
async def test_r2_01_session_proxy_refuses_bad_paths_itself(path):
    """R2-01: the session proxy port is reachable without the gateway: it applies the same refusal (%25 in the
    raw path, '.'/'..' segments or backslashes in the decoded path)."""
    hits = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: hits.append(req) or httpx.Response(200))
                                 ) as vm_http:
        proxy = SessionProxy(3080, lambda: Route("forward", 8188, run_id=1), AsyncMock(), Activity(100.0), vm_http,
                             lambda: 500.0)
        async with httpx.AsyncClient(transport=Hop(proxy), base_url="http://127.0.0.1:3080") as c:
            r = await c.post(path, json={})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert hits == []


async def test_r2_01_encoded_delimiters_stay_in_the_path(chain):
    """R2-01: a %3f or %23 inside the path reaches the VM still encoded: it never becomes a query or fragment."""
    r1 = await chain.public.get("/comfy/comfy-image-uncensored/view%3ffilename=injected.png&type=input",
                                headers=auth())
    r2 = await chain.public.post("/v1/chat/completions%23frag", headers=auth(), json={"model": "qwen-a"})
    r3 = await chain.public.get("/comfy/comfy-image-uncensored/history/a%3Fb", headers=auth())
    assert r1.status_code == r2.status_code == r3.status_code == 200
    assert chain.vm == [
        (8090, "GET", b"/view%3ffilename=injected.png&type=input", "/view?filename=injected.png&type=input", ""),
        (8091, "POST", b"/v1/chat/completions%23frag", "/v1/chat/completions#frag", ""),
        (8090, "GET", b"/history/a%3Fb", "/history/a?b", ""),
    ]
    # Not an inference path any more once %23 stays encoded: no activity credit for a made-up path.
    assert chain.activity[3081].last == 100.0


async def test_r2_01_normal_comfy_and_dotted_model_paths_still_work(tmp_path):
    """R2-01: the regular ComfyUI raw API (/view?filename=a.png, /prompt) and a model id with dots keep working,
    with the query string passed through and POST /prompt still counted as activity."""
    c = Chain([llm(alias="qwen3.6-flash", port=3081), comfy(alias="comfy.image-v1.5", port=3080)])
    try:
        r1 = await c.public.get("/comfy/comfy.image-v1.5/view?filename=a.png&subfolder=x&type=output",
                                headers=auth())
        r2 = await c.public.post("/comfy/comfy.image-v1.5/prompt", headers=auth(), json={"prompt": {}})
        r3 = await c.public.post("/v1/chat/completions", headers=auth(), json={"model": "qwen3.6-flash"})
        r4 = await c.public.get("/comfy/img-2/object_info", headers=auth())  # by colab name
        assert [r.status_code for r in (r1, r2, r3, r4)] == [200] * 4
        assert [(port, raw, query) for port, _, raw, _, query in c.vm] == [
            (8090, b"/view", "filename=a.png&subfolder=x&type=output"), (8090, b"/prompt", ""),
            (8091, b"/v1/chat/completions", ""), (8090, b"/object_info", "")]
        assert c.activity[3080].last == c.activity[3081].last == 500.0
    finally:
        await c.close()


async def test_r2_01_encoded_slash_is_forwarded_as_encoded_slash(chain):
    """R2-01: a legitimate %2F inside a /comfy path segment (e.g. a subfolder in a file name) reaches ComfyUI as
    %2F, not as a path separator; an encoded '/' in the model segment cannot shift the prefix split."""
    r = await chain.public.get("/comfy/comfy-image-uncensored/view/sub%2Fa.png", headers=auth())
    assert r.status_code == 200
    assert chain.vm == [(8090, "GET", b"/view/sub%2Fa.png", "/view/sub/a.png", "")]
    r = await chain.public.get("/comfy/comfy-image-uncensored%2Fview/x", headers=auth())
    assert r.status_code in (400, 404) and len(chain.vm) == 1


async def test_r2_01_upload_limit_follows_the_validated_path(chain):
    """R2-01: body limit and is_inference() use the decoded, validated path: a >64 MiB JSON body can no longer
    borrow the 128 MiB upload limit through /upload/%252e%252e/prompt, and a real /upload/image still can."""
    big = b" " * (64 * 1024 * 1024 + 1024) + b"{}"
    r = await chain.public.post("/comfy/comfy-image-uncensored/upload/%252e%252e/prompt", headers=auth(),
                                content=big)
    assert r.status_code == 400 and chain.vm == []
    r = await chain.public.post("/comfy/comfy-image-uncensored/prompt", headers=auth(), content=big)
    assert r.status_code == 413 and chain.vm == []
    r = await chain.public.post("/comfy/comfy-image-uncensored/upload/image", headers=auth(), content=big)
    assert r.status_code == 200 and [hit[2] for hit in chain.vm] == [b"/upload/image"]


@pytest.mark.parametrize("scope,wire", [
    ({"path": "/v1/chat/completions"}, "/v1/chat/completions"),  # test-built scope without raw_path
    ({"path": "/view/a b.png"}, "/view/a%20b.png"),
    ({"path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions"}, "/v1/chat/completions"),
    ({"path": "/view/sub/a.png", "raw_path": b"/view/sub%2Fa.png"}, "/view/sub%2Fa.png"),
    ({"path": "/v1/%2e%2e/props"}, None),  # decoded path that would decode again
    ({"path": "/v1/../props"}, None),
    ({"path": "/v1/x", "raw_path": b"/v1/%252e"}, None),
    ({"path": "/v1/x", "raw_path": b"/v1/y"}, None),  # raw and decoded disagree
])
def test_r2_01_wire_path_with_and_without_raw_path(scope, wire):
    """R2-01: wire_path() keeps raw escapes, falls back to quoting scope['path'] when raw_path is absent (ASGI
    scopes built by tests), and refuses double encoding, dot segments and raw/decoded mismatches."""
    assert wire_path(scope) == wire


# ------------------------------------------------------------------------------------------------------- R2-02
def failing_comfy(status=None, exc=None, prompt_status=200):
    def handler(req):
        if exc is not None:
            raise exc(req)
        if req.url.path == "/prompt":
            return httpx.Response(prompt_status, json={"error": "invalid prompt"})
        return httpx.Response(status or 200, json={})
    return handler


@pytest.mark.parametrize("comfy_handler,body,status", [
    (Comfy(), {"workflow": "h3edit", "image": "not-base64!"}, 400),  # local validation error
    (failing_comfy(exc=lambda req: httpx.ConnectError("tunnel down", request=req)), {}, 502),  # broken tunnel
    (failing_comfy(prompt_status=400), {}, 502),  # ComfyUI refused the workflow
    (Comfy(history=lambda n: {"p1": {"status": {"status_str": "error", "messages": []}}}), {}, 502),  # render error
    (Comfy(history=lambda n: {}), {}, 502),  # prompt vanished
])
async def test_r2_02_failed_image_jobs_do_not_restart_the_idle_clock(world, comfy_handler, body, status):  # noqa: F811
    """R2-02: images() ends its activity with useful=False on every MediaError (400, 502, render error, vanished
    prompt): retrying a failing request against a broken session no longer keeps the paid VM alive."""
    w = await world(comfy_handler)
    act = w.service.activity(IMAGE)
    act.last = 5.0
    with pytest.raises(MediaError) as caught:
        await w.service.images(IMAGE, ImageRequest(model="m", prompt="x", **body))
    assert caught.value.status == status
    assert act.last == 5.0 and not act._inflight


async def test_r2_02_image_save_failure_and_cancellation_are_not_activity(world):  # noqa: F811
    """R2-02: an OSError while saving a url result and a cancelled job also end with useful=False."""
    w = await world(Comfy())
    act = w.service.activity(IMAGE)
    act.last = 5.0
    w.service.files.save = enospc
    with pytest.raises(OSError):
        await w.service.images(IMAGE, ImageRequest(model="m", prompt="x", response_format="url"))
    assert act.last == 5.0 and not act._inflight

    w2 = await world(Comfy(history=lambda n: {}, queue=lambda n: {"queue_running": [[0, "p1"]]}))
    act2 = w2.service.activity(VIDEO)  # another sid: its own clock
    act2.last = 5.0
    task = w2.service.spawn(w2.service.images(VIDEO, ImageRequest(model="m", prompt="x")))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert act2.last == 5.0 and not act2._inflight


async def test_r2_02_successful_image_still_counts(world):  # noqa: F811
    """R2-02: a delivered image still restarts the idle clock."""
    w = await world(Comfy())
    act = w.service.activity(IMAGE)
    act.last = 5.0
    result = await w.service.images(IMAGE, ImageRequest(model="m", prompt="x"))
    assert result["data"] and act.last == 1000.0 and not act._inflight


async def test_r2_02_failed_video_is_not_activity_and_completed_one_is(world):  # noqa: F811
    """R2-02: _video() only counts a completed, saved video; a render error or a failed save do not."""
    error = {"p1": {"status": {"status_str": "error", "messages": ["boom"]}}}
    for comfy_handler, save_fails, status, useful in [(Comfy(history=lambda n: error), False, "failed", False),
                                                      (Comfy(), True, "failed", False),
                                                      (Comfy(), False, "completed", True)]:
        w = await world(comfy_handler)
        act = w.service.activity(VIDEO)
        act.last = 5.0
        if save_fails:
            w.service.files.save = enospc
        job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))[
            "id"])
        assert job["status"] == status
        assert act.last == (1000.0 if useful else 5.0) and not act._inflight


async def test_r2_02_cancelled_video_is_not_activity(world):  # noqa: F811
    """R2-02: a video job cancelled by a service shutdown does not move the idle clock."""
    w = await world(Comfy(history=lambda n: {}, queue=lambda n: {"queue_running": [[0, "p1"]]}))
    act = w.service.activity(VIDEO)
    act.last = 5.0
    await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x"))
    await asyncio.sleep(0.05)
    await w.service.close()
    assert act.last == 5.0 and not act._inflight


async def test_r2_02_broken_tunnel_image_retries_let_the_session_idle_out(tmp_path):
    """R2-02: with the tunnel down, a client retrying /v1/images/generations every 15 min no longer keeps a
    ComfyUI session alive: it idles out after idle_min (the R1-01 grace is the only deferral)."""
    sup, row, clock, client = comfy_guard(tmp_path, tunnel_down)
    async with client:
        service = MediaService(sup, client=client, poll_s=0.01)
        target = SimpleNamespace(sid=1, port=3080)
        calls = 0
        try:
            for step in range(1200):  # 10 h of 30 s guard ticks
                if sup.stop.called:
                    break
                clock.t += 30.0
                if step % 30 == 0:  # a retry every 15 min
                    calls += 1
                    with pytest.raises(MediaError) as caught:
                        await service.images(target, ImageRequest(model="m", prompt="cat"))
                    assert caught.value.status == 502
                await sup.guards_once()
        finally:
            await service.close()
    assert calls >= 2 and sup.stop.called and sup.stop.call_args.args == (1, "idle")
    assert clock.t - 1000.0 <= 30 * 60 + 60 and row["last_activity"] == 1000.0


# ------------------------------------------------------------------------------------------------------- R2-03
class OutageCLI:
    """`colab exec` and `colab stop` both fail (e.g. expired OAuth) until `fixed`."""

    enabled = True

    def __init__(self, clock):
        self.clock = clock
        self.runtimes = [Runtime("ext-g4", "gpu-g4-ext-1", "G4", "Standard", "GPU")]
        self.fixed = False
        self.stops = 0

    def command(self):
        return ["colab"]

    async def listing(self):
        return Listing(list(self.runtimes), Usage(500.0, SEED_RATES["G4"][0], len(self.runtimes)), True,
                       self.clock.now())

    async def exec_code(self, name, code, timeout=None):
        return CmdResult(0, "1\n") if self.fixed else CmdResult(1, "OAuth token expired")

    async def stop(self, name):
        self.stops += 1
        if not self.fixed:
            return CmdResult(1, "OAuth token expired")
        self.runtimes = []
        return CmdResult(0, "[colab] Session terminated.")


@pytest.fixture
def outage(store, clock, tmp_path):
    cli = OutageCLI(clock)
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=cli, tunnels=None,
                     proxies=ProxyManager(), alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=PresetError("unused"), http=object())
    return sup, cli


def alert(sup, key):
    return next((a for a in sup.alerts.list() if a.key == key), None)


async def test_r2_03_autostop_failure_after_heartbeat_failures_keeps_its_stop_button(outage, clock, caplog):
    """R2-03: when heartbeats already failed (external_keepalive alert), a failing auto-stop still raises its own
    external_autostop alert, logged, with the stop-external action and target, rendered as 「停止 ext-g4」."""
    sup, cli = outage
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-ext-1", True, stop_after_min=5)
    for _ in range(3):
        await sup.keepalive_externals_once()
        clock.advance(60)
    beat = alert(sup, "external_keepalive:gpu-g4-ext-1")
    assert beat and "保活心跳连续 3 次失败" in beat.message and beat.actions == []
    clock.advance(130)
    with caplog.at_level(logging.WARNING, logger="colabinet.alerts"):
        await sup.keepalive_externals_once()
    stop = alert(sup, "external_autostop:gpu-g4-ext-1")
    assert stop is not None and "保活计时已到，但自动停止失败" in stop.message
    assert stop.actions == ["stop-external"] and stop.target == "ext-g4" and stop.level == "critical"
    assert any("ALERT external_autostop" in r.getMessage() for r in caplog.records)
    assert alert(sup, "external_keepalive:gpu-g4-ext-1") is beat  # the heartbeat alert is untouched
    html = make_templates().env.get_template("partials/overview_attention.html").render(
        overview_data={"alerts": [a.__dict__ for a in sup.alerts.list()], "account_id": "default"})
    assert "停止 ext-g4" in html and "外部运行时自动停止失败" in html


async def test_r2_03_autostop_alert_clears_when_the_stop_succeeds(outage, clock):
    """R2-03: the external_autostop alert goes away once a retry stops the runtime."""
    sup, cli = outage
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-ext-1", True, stop_after_min=1)
    clock.advance(61)
    await sup.keepalive_externals_once()
    assert alert(sup, "external_autostop:gpu-g4-ext-1") is not None
    cli.fixed = True
    clock.advance(15)
    await sup.keepalive_externals_once()
    assert cli.stops == 2
    assert alert(sup, "external_autostop:gpu-g4-ext-1") is None
    assert alert(sup, "external_keepalive:gpu-g4-ext-1") is None


async def test_r2_03_autostop_alert_clears_when_the_runtime_disappears(outage, clock):
    """R2-03: a runtime that vanished from the listing (stopped elsewhere) also clears its external_autostop."""
    sup, cli = outage
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-ext-1", True, stop_after_min=1)
    clock.advance(61)
    await sup.keepalive_externals_once()
    assert alert(sup, "external_autostop:gpu-g4-ext-1") is not None
    cli.runtimes = []
    await sup.reconcile_once()
    assert alert(sup, "external_autostop:gpu-g4-ext-1") is None


# ------------------------------------------------------------------------------------------------------- R2-04
def stuck_queue(request):
    return httpx.Response(200, json={"queue_running": [[0, "stuck"]], "queue_pending": []})


async def test_r2_04_busy_queue_is_never_persisted_as_activity(tmp_path):
    """R2-04: a non-empty /queue defers the idle stop but never writes sessions.last_activity."""
    sup, row, clock, client = comfy_guard(tmp_path, stuck_queue)
    async with client:
        for _ in range(120):
            clock.t += 60
            await sup.guards_once()
    assert not sup.stop.called
    assert row["last_activity"] == 1000.0 and sup._activity[1].last == 1000.0
    assert not hasattr(sup, "_queue_busy_since")


async def test_r2_04_busy_cap_survives_engine_down_flaps_and_restarts(tmp_path):
    """R2-04: the 6 h busy cap counts from the persisted last real activity, so an engine_down flap and a service
    restart (fresh Supervisor, Activity loaded from the row) do not grant another 6 h."""
    sup, row, clock, client = comfy_guard(tmp_path, stuck_queue)
    async with client:
        for _ in range(350):  # 5 h 50 min busy
            clock.t += 60
            await sup.guards_once()
        row["state"] = "engine_down"
        clock.t += 60
        await sup.guards_once()
        row["state"] = "ready"
        assert not sup.stop.called
        # restart: a new supervisor with the persisted row
        sup2, _, _, _ = comfy_guard(tmp_path, stuck_queue)
        sup2.http, sup2.clock, sup2.store = client, clock, sup.store
        sup2._activity[1] = Activity(row["last_activity"])
        while not sup2.stop.called and clock.t - 1000.0 < 24 * 3600:
            clock.t += 60
            await sup2.guards_once()
    assert sup2.stop.called and sup2.stop.call_args.args == (1, "idle")
    assert QUEUE_BUSY_MAX_S <= clock.t - 1000.0 <= QUEUE_BUSY_MAX_S + 30 * 60 + 60


async def test_r2_04_busy_queue_without_activity_record_counts_from_ready_at(tmp_path):
    """R2-04: a row with no last_activity yet falls back to ready_at for the busy cap."""
    sup, row, clock, client = comfy_guard(tmp_path, stuck_queue, start=1000.0, ready_at=1000.0)
    row["last_activity"] = None
    sup._activity[1] = Activity()
    async with client:
        elapsed = await run_until_stop(sup, clock, 24 * 3600, step=60.0)
    assert sup.stop.called and QUEUE_BUSY_MAX_S <= elapsed <= QUEUE_BUSY_MAX_S + 30 * 60 + 60


async def test_r2_04_real_activity_extends_the_busy_cap(tmp_path):
    """R2-04: a real request (e.g. a proxied POST /prompt) while the queue is busy restarts the cap from it."""
    sup, row, clock, client = comfy_guard(tmp_path, stuck_queue)
    async with client:
        clock.t += 5 * 3600
        await sup.guards_once()
        sup._activity[1].end(7, clock.t, useful=True)
        real = clock.t
        await sup.guards_once()
        assert row["last_activity"] == real
        elapsed = await run_until_stop(sup, clock, 24 * 3600, step=60.0)
    assert sup.stop.called
    assert QUEUE_BUSY_MAX_S <= clock.t - real <= QUEUE_BUSY_MAX_S + 30 * 60 + 60 and elapsed > QUEUE_BUSY_MAX_S


async def test_r2_04_failed_probe_grace_unchanged(tmp_path):
    """R2-04 keeps R1-01: a failing /queue probe defers the idle stop only QUEUE_GRACE_S from ready_at."""
    sup, row, clock, client = comfy_guard(tmp_path, tunnel_down, idle_min=1)
    async with client:
        elapsed = await run_until_stop(sup, clock, 3600)
    assert sup.stop.called and elapsed == pytest.approx(QUEUE_GRACE_S, abs=31)
    assert row["last_activity"] == 1000.0


# ------------------------------------------------------------------------------------------------------- R2-05
def test_r2_05_filter_drops_only_stream_aborted_chains():
    """R2-05: the uvicorn.error filter drops records whose exception chain holds StreamAborted, keeps the rest;
    installing it twice adds one filter."""
    def record(exc):
        return logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1, "Exception in ASGI application\n", (),
                                 (type(exc), exc, None) if exc else None)

    try:
        try:
            raise httpx.ReadError("tunnel dropped")
        except httpx.ReadError as inner:
            raise StreamAborted("upstream broke") from inner
    except StreamAborted as e:
        aborted = e
    try:
        try:
            raise aborted
        except StreamAborted:
            raise RuntimeError("wrapped by a middleware")  # noqa: B904 (implicit __context__)
    except RuntimeError as e:
        wrapped = e
    f = StreamAbortedFilter()
    assert not f.filter(record(aborted)) and not f.filter(record(wrapped))
    assert f.filter(record(ValueError("real bug"))) and f.filter(record(None))
    logger = logging.getLogger("colabinet.test.r2_05")
    install_log_filter(logger)
    install_log_filter(logger)
    assert sum(isinstance(x, StreamAbortedFilter) for x in logger.filters) == 1
    install_log_filter()
    install_log_filter()
    assert sum(isinstance(x, StreamAbortedFilter) for x in logging.getLogger("uvicorn.error").filters) == 1


@pytest.mark.parametrize("early", [False, True])  # passthrough Content-Length, and our own early 200
async def test_r2_05_real_uvicorn_drops_connection_without_traceback(caplog, early):
    """R2-05: through a real uvicorn server a mid-body upstream break still drops the client connection, but
    uvicorn logs no ERROR traceback; the gateway logs one WARNING line instead."""
    def respond(req):
        headers = {"content-type": "application/json"}
        if not early:
            headers["content-length"] = "100"
        response = httpx.Response(200, stream=BrokenStream(b'{"id":"1","choices":['), headers=headers)
        return slow(response) if early else response

    async def handler(req):
        await req.aread()
        out = respond(req)
        return await out if asyncio.iscoroutine(out) else out

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gw = Gateway(SimpleNamespace(store=Store([llm()])), KEY, client=client, cache_s=0, grace=0.01, interval=0.01)
    port = first_free(28900, 29199)
    server, task = await start_server(gw, port)
    try:
        with caplog.at_level(logging.WARNING):
            async with httpx.AsyncClient() as c:
                with pytest.raises(httpx.HTTPError):
                    await c.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers=auth(),
                                 json={"model": "qwen-a"})
            await asyncio.sleep(0.05)
    finally:
        await stop_server(server, task)
        await gw.aclose()
        await client.aclose()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and r.name.startswith("uvicorn")]
    assert not [r for r in errors if r.exc_info], [r.getMessage() for r in errors]
    warnings = [r.getMessage() for r in caplog.records if r.name == "colabinet.gateway"]
    assert len(warnings) == 1 and "stream broke: ReadError; dropping the client connection" in warnings[0]
