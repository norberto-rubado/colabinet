"""Regression tests for review round 3: R3-01 .. R3-05."""

import asyncio
import dataclasses
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.alerts import Alerts, NullNotifier
from colabinet.api import create_app
from colabinet.cli_parse import Listing, Runtime, Usage
from colabinet.colab_cli import CmdResult
from colabinet.comfy_graphs import ImageRequest, VideoRequest
from colabinet.config import REPO_ROOT, SEED_RATES, Settings
from colabinet.media import MediaError, MediaService
from colabinet.presets import PresetError, load_presets
from colabinet.proxy import MAX_BODY_BYTES, ProxyManager, Route, SessionProxy
from colabinet.supervisor import QUEUE_BUSY_MAX_S, Supervisor
from colabinet.web import views
from colabinet.web.app import make_templates
from tests.unit.test_review_r1_gateway import auth, comfy
from tests.unit.test_review_r1_lifecycle import comfy_guard
from tests.unit.test_review_r1_media import Comfy, enospc, wait_video
from tests.unit.test_review_r2 import Chain, Hop

IDLE_S = 30 * 60


# ------------------------------------------------------------------------------------------------------- R3-01
class Batch:
    """ComfyUI VM: POST /prompt queues a long batch that runs until `done` is set."""

    def __init__(self):
        self.busy = False

    def __call__(self, req):
        if req.url.path == "/prompt":
            self.busy = True
            return httpx.Response(200, json={"prompt_id": "batch"})
        if req.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, "batch"]] if self.busy else [],
                                             "queue_pending": []})
        return httpx.Response(200, json={})


def ui_idle_left(sup, row):
    """views.session_rows() over the guard's row (stats and run lookups stubbed)."""
    full = dict(row, proxy_port=3080, account_id="default", stop_unconfirmed=0)
    sup.store.db = SimpleNamespace(fetchall=AsyncMock(side_effect=lambda *a, **k: [dict(full)]),
                                   fetchone=AsyncMock(return_value={"n": 0, "out": 0, "tps": None}))
    sup.store.current_run = AsyncMock(return_value={"engine": "comfyui", "spec_json": "{}"})
    return views.session_rows(sup, "1=1")


async def submit_batch(sup, vm, clock):
    """One real POST /prompt through the session proxy (it is inference activity: last_activity = submit time)."""
    proxy = SessionProxy(3080, lambda: Route("forward", 23001, run_id=1), AsyncMock(), sup._activity[1], sup.http,
                         clock.now)
    async with httpx.AsyncClient(transport=Hop(proxy), base_url="http://127.0.0.1:3080") as c:
        assert (await c.post("/prompt", json={"prompt": {}})).status_code == 200
    assert vm.busy


async def test_r3_01_long_render_gets_the_full_idle_window_after_the_queue_drains(tmp_path):
    """R3-01: a 4 h batch submitted with a single POST /prompt is not stopped when the queue drains: the session
    survives about idle_min after the drain, gets an idle_warn first, then idles out; the panel's countdown follows
    the busy watermark during and after the render."""
    vm = Batch()
    sup, row, clock, client = comfy_guard(tmp_path, vm)
    async with client:
        await submit_batch(sup, vm, clock)
        for _ in range(480):  # 4 h of 30 s guard ticks
            clock.t += 30
            await sup.guards_once()
        assert not sup.stop.called and row["last_activity"] == 1000.0  # busy never persisted
        during = (await ui_idle_left(sup, row))[0]["idle_left_s"]
        assert during == IDLE_S  # not 0 while the render runs
        vm.busy = False
        drained = clock.t
        warned_at = None
        while not sup.stop.called and clock.t - drained < 2 * 3600:
            clock.t += 30
            await sup.guards_once()
            if warned_at is None and any(a.kind == "idle_warn" for a in sup.alerts.list()):
                warned_at = clock.t
            if clock.t - drained == 600:  # the watermark is the last tick that saw the queue busy (= drained)
                assert (await ui_idle_left(sup, row))[0]["idle_left_s"] == IDLE_S - 600
    assert sup.stop.called and sup.stop.call_args.args == (1, "idle")
    assert IDLE_S - 30 <= clock.t - drained <= IDLE_S + 30
    assert warned_at is not None and clock.t - warned_at >= 4 * 60
    assert row["last_activity"] == 1000.0


async def test_r3_01_watermark_never_extends_the_six_hour_cap(tmp_path):
    """R3-01: a queue busy until just before the cap and then drained still stops by real + 6 h + idle_min."""
    vm = Batch()
    sup, row, clock, client = comfy_guard(tmp_path, vm)
    async with client:
        await submit_batch(sup, vm, clock)
        while clock.t - 1000.0 < QUEUE_BUSY_MAX_S + 3600:  # busy past the cap
            clock.t += 60
            await sup.guards_once()
            if sup.stop.called:
                break
            if clock.t - 1000.0 == QUEUE_BUSY_MAX_S + 10 * 60:
                vm.busy = False  # drains after the cap: the watermark stays at real + 6 h
        assert sup._queue_busy_floor[1] <= 1000.0 + QUEUE_BUSY_MAX_S
    assert sup.stop.called and clock.t - 1000.0 <= QUEUE_BUSY_MAX_S + IDLE_S + 60


async def test_r3_01_restart_loses_the_watermark_but_not_the_cap(tmp_path):
    """R3-01: the watermark is in memory only. After a restart (fresh Supervisor, Activity from the persisted row) a
    drained session stops no later than the persisted clock allows, and a still busy one by real + 6 h + idle_min."""
    for still_busy in (False, True):
        vm = Batch()
        sup, row, clock, client = comfy_guard(tmp_path, vm)
        async with client:
            await submit_batch(sup, vm, clock)
            for _ in range(5 * 60):  # 5 h busy
                clock.t += 60
                await sup.guards_once()
            assert not sup.stop.called and sup._queue_busy_floor[1] == clock.t
            vm.busy = still_busy
            sup2, _, _, _ = comfy_guard(tmp_path, vm)  # restart: nothing in memory survives
            sup2.http, sup2.clock, sup2.store = client, clock, sup.store
            sup2._activity[1] = Activity(row["last_activity"])
            restart = clock.t
            while not sup2.stop.called and clock.t - 1000.0 < 24 * 3600:
                clock.t += 60
                await sup2.guards_once()
        assert sup2.stop.called and sup2.stop.call_args.args == (1, "idle")
        assert clock.t - 1000.0 <= QUEUE_BUSY_MAX_S + IDLE_S + 60
        if not still_busy:  # the persisted clock (1000) is long past idle_min: the first tick stops it
            assert clock.t - restart <= 60
        else:  # the cap from the persisted last activity
            assert clock.t - 1000.0 >= QUEUE_BUSY_MAX_S


async def test_r3_01_new_activity_still_restarts_the_window(tmp_path):
    """R3-01: the watermark is only a floor: real activity after the drain still counts from itself."""
    vm = Batch()
    sup, row, clock, client = comfy_guard(tmp_path, vm)
    async with client:
        await submit_batch(sup, vm, clock)
        clock.t += 3600
        await sup.guards_once()
        vm.busy = False
        clock.t += 20 * 60
        sup._activity[1].end(9, clock.t, useful=True)
        real = clock.t
        while not sup.stop.called:
            clock.t += 30
            await sup.guards_once()
    assert IDLE_S - 30 <= clock.t - real <= IDLE_S + 30


# ------------------------------------------------------------------------------------------------------- R3-02
def streamed(handler):
    """MockTransport responses as a real byte stream: SessionProxy relays them with aiter_raw()."""
    def handle(req):
        resp = handler(req)
        return httpx.Response(resp.status_code, headers=resp.headers, stream=httpx.ByteStream(resp.content))
    return handle


@pytest.fixture
async def media_chain(tmp_path):
    """MediaService -> the real SessionProxy (uvicorn-like hop) -> scriptable ComfyUI, sharing the supervisor's
    Activity, with guards_once() persisting last_activity into `row` like the real store."""
    made = []

    async def make(vm):
        sup, row, clock, client = comfy_guard(tmp_path, streamed(vm))
        clock.t = 2000.0
        records = []

        async def record(rec, run_id):
            records.append(rec)

        proxy = SessionProxy(3080, lambda: Route("forward", 23001, run_id=1), record, sup._activity[1], client,
                             clock.now)
        proxy_client = httpx.AsyncClient(transport=Hop(proxy))
        service = MediaService(sup, client=proxy_client, poll_s=0.002, timeout_s=5, failure_window_s=1.0,
                               backoff_max_s=0.01, fetch_backoff_s=0.001)
        made.append((service, proxy_client, client))
        return SimpleNamespace(sup=sup, row=row, clock=clock, service=service, act=sup._activity[1],
                               records=records, target=SimpleNamespace(sid=1, port=3080))

    yield make
    for service, proxy_client, client in made:
        await service.close()
        await proxy_client.aclose()
        await client.aclose()


async def assert_not_activity(w):
    assert w.act.last == 1000.0 and not w.act._inflight
    await w.sup.guards_once()
    assert w.row["last_activity"] == 1000.0
    prompts = [r for r in w.records if r.path == "/prompt"]
    assert prompts and prompts[0].status == 200  # queued (200) and still a request row for the stats
    assert all(r.status < 400 for r in w.records)  # every hop worked: the failure is the job's own


@pytest.mark.parametrize("vm,status", [
    (Comfy(history=lambda n: {"p1": {"status": {"status_str": "error", "messages": ["CUDA out of memory"]}}}), 502),
    (Comfy(history=lambda n: {}), 502),  # PromptVanished
])
async def test_r3_02_failed_image_after_a_queued_prompt_is_not_activity(media_chain, vm, status):
    """R3-02: through the real SessionProxy, an image job whose POST /prompt was answered 200 but then failed
    (render error, vanished prompt) refreshes neither Activity.last nor the persisted last_activity."""
    w = await media_chain(vm)
    with pytest.raises(MediaError) as caught:
        await w.service.images(w.target, ImageRequest(model="m", prompt="x"))
    assert caught.value.status == status
    assert "渲染失败" in str(caught.value) or "均未找到该任务" in str(caught.value)  # failed after queueing
    await assert_not_activity(w)


async def test_r3_02_image_enospc_and_cancellation_are_not_activity(media_chain):
    """R3-02: a failed save (ENOSPC) and a cancelled image job, both after a 200 POST /prompt, are not activity."""
    w = await media_chain(Comfy())
    w.service.files.save = enospc
    with pytest.raises(OSError):
        await w.service.images(w.target, ImageRequest(model="m", prompt="x", response_format="url"))
    await assert_not_activity(w)

    w = await media_chain(Comfy(history=lambda n: {}, queue=lambda n: {"queue_running": [[0, "p1"]]}))
    task = w.service.spawn(w.service.images(w.target, ImageRequest(model="m", prompt="x")))
    for _ in range(200):
        if any(r.path == "/prompt" for r in w.records):
            break
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await assert_not_activity(w)


async def test_r3_02_failed_and_cancelled_videos_are_not_activity(media_chain):
    """R3-02: a video job that fails after a 200 POST /prompt (render error, ENOSPC on save) or is cancelled by a
    service shutdown does not move the idle clock through the proxy."""
    error = {"p1": {"status": {"status_str": "error", "messages": ["boom"]}}}
    for vm, save_fails in ((Comfy(history=lambda n: error), False), (Comfy(), True)):
        w = await media_chain(vm)
        if save_fails:
            w.service.files.save = enospc
        job = await wait_video(w.service, (await w.service.create_video(w.target, VideoRequest(model="m",
                                                                                               prompt="x")))["id"])
        assert job["status"] == "failed"
        await assert_not_activity(w)
    w = await media_chain(Comfy(history=lambda n: {}, queue=lambda n: {"queue_running": [[0, "p1"]]}))
    await w.service.create_video(w.target, VideoRequest(model="m", prompt="x"))
    await asyncio.sleep(0.05)
    await w.service.close()
    await assert_not_activity(w)


async def test_r3_02_successful_jobs_still_count(media_chain):
    """R3-02: a delivered image and a completed video still refresh the idle clock (and it is persisted)."""
    w = await media_chain(Comfy())
    result = await w.service.images(w.target, ImageRequest(model="m", prompt="x"))
    assert result["data"] and w.act.last == 2000.0
    await w.sup.guards_once()
    assert w.row["last_activity"] == 2000.0
    w = await media_chain(Comfy())
    job = await wait_video(w.service, (await w.service.create_video(w.target, VideoRequest(model="m",
                                                                                           prompt="x")))["id"])
    assert job["status"] == "completed" and w.act.last == 2000.0


async def test_r3_02_retried_failing_job_no_longer_keeps_the_vm_alive(media_chain):
    """R3-02: a client retrying a render that always fails after queueing (CUDA OOM) every 15 min no longer keeps
    the session alive: it idles out after idle_min."""
    w = await media_chain(Comfy(history=lambda n: {"p1": {"status": {"status_str": "error", "messages": []}}}))
    w.clock.t = 1000.0
    for step in range(1200):
        if w.sup.stop.called:
            break
        w.clock.t += 30
        if step % 30 == 0:
            with pytest.raises(MediaError):
                await w.service.images(w.target, ImageRequest(model="m", prompt="x"))
        await w.sup.guards_once()
    assert w.sup.stop.called and w.clock.t - 1000.0 <= IDLE_S + 60 and w.row["last_activity"] == 1000.0


async def test_r3_02_media_header_is_stripped_by_the_gateway_and_not_forwarded():
    """R3-02: a public client cannot forge x-colabinet-media through the gateway: its POST /prompt still counts as
    activity, and the marker never reaches the VM (the session proxy drops it too)."""
    seen = []
    c = Chain([comfy(port=3080)])
    real = c.vm_http._transport.handler

    def vm(req):
        seen.append(req.headers.get("x-colabinet-media"))
        return real(req)

    c.vm_http._transport.handler = vm
    try:
        r = await c.public.post("/comfy/comfy-image-uncensored/prompt", headers={**auth(), "x-colabinet-media": "1"},
                                json={"prompt": {}})
        assert r.status_code == 200 and c.activity[3080].last == 500.0 and seen == [None]
    finally:
        await c.close()


async def test_r3_02_session_proxy_honours_the_marker_only_from_loopback():
    """R3-02: the marker only makes a POST /prompt activity-neutral for a loopback client."""
    async def call(client_addr):
        act = Activity(100.0)
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, json={"prompt_id": "p"}))) as vm_http:
            proxy = SessionProxy(3080, lambda: Route("forward", 8188, run_id=1), AsyncMock(), act, vm_http,
                                 lambda: 500.0)
            sent, messages = [], [{"type": "http.request", "body": b"{}", "more_body": False}]

            async def receive():
                if messages:
                    return messages.pop()
                await asyncio.sleep(3600)

            async def send(msg):
                sent.append(msg)

            await proxy({"type": "http", "method": "POST", "path": "/prompt", "raw_path": b"/prompt",
                         "query_string": b"", "client": client_addr,
                         "headers": [(b"host", b"127.0.0.1:3080"), (b"x-colabinet-media", b"1")]}, receive, send)
        assert sent[0]["status"] == 200
        return act.last

    assert await call(("127.0.0.1", 50000)) == 100.0
    assert await call(("203.0.113.9", 50000)) == 500.0


# ------------------------------------------------------------------------------------------------------- R3-03
@pytest.fixture
async def upload_chain():
    row, run = comfy(sid=2, alias="comfy/image-v1", port=3080)
    row["colab_name"] = "upload"
    c = Chain([(row, run)])
    yield c
    await c.close()


@pytest.mark.parametrize("path", ["/comfy%2Fupload/upload//object_info", "/comfy%2fupload/upload/object_info",
                                  "/comfy%2Fupload/upload/prompt"])
async def test_r3_03_encoded_slash_after_comfy_is_refused(upload_chain, path):
    """R3-03: the raw path must start with the literal '/comfy/': an encoded '/' after /comfy (which shifted the
    raw split and sent //object_info upstream) is a 400 at the gateway, and nothing reaches a VM."""
    r = await upload_chain.public.post(path, headers=auth(), json={})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    assert upload_chain.vm == []


async def test_r3_03_shifted_upload_path_does_not_borrow_the_upload_limit(upload_chain):
    """R3-03: a >64 MiB body on /comfy%2Fupload/upload/prompt is refused by the gateway itself (400 before the body
    is read), /comfy/upload/prompt gets the gateway's 413, a real /comfy/upload/upload/image keeps 128 MiB."""
    big = b" " * (MAX_BODY_BYTES + 1024 * 1024) + b"{}"
    r = await upload_chain.public.post("/comfy%2Fupload/upload/prompt", headers=auth(), content=big)
    assert r.status_code == 400 and upload_chain.vm == []
    r = await upload_chain.public.post("/comfy/upload/prompt", headers=auth(), content=big)
    assert r.status_code == 413 and r.json()["error"]["message"].startswith("colabinet gateway:")
    r = await upload_chain.public.post("/comfy/upload/upload/image", headers=auth(), content=big)
    assert r.status_code == 200 and [hit[2] for hit in upload_chain.vm] == [b"/upload/image"]


async def test_r3_03_model_ids_with_a_slash_get_a_clear_400(upload_chain):
    """R3-03: model ids containing '/' are not supported on /comfy: the encoded form is a 400 that says so (not a
    misleading 404), and the session name keeps working."""
    r = await upload_chain.public.get("/comfy/comfy%2Fimage-v1/object_info", headers=auth())
    assert r.status_code == 400 and "model ids containing '/' are not supported" in r.json()["error"]["message"]
    r = await upload_chain.public.get("/comfy/upload/object_info", headers=auth())
    assert r.status_code == 200 and upload_chain.vm[-1][2] == b"/object_info"


@pytest.mark.parametrize("path,wire,expected", [
    ("/comfy/m/view", "/comfy/m/view", ("m", "/view", "/view")),
    ("/comfy/m", "/comfy/m", ("m", "/", "/")),
    ("/comfy/a b/view/sub/x.png", "/comfy/a%20b/view/sub%2Fx.png", ("a b", "/view/sub/x.png", "/view/sub%2Fx.png")),
    ("/comfy/m/upload/image", "/comfy/m/upload/image", ("m", "/upload/image", "/upload/image")),
    ("/comfy/m/m//x", "/comfy%2Fm/m//x", None),
    ("/comfy/a/b/x", "/comfy/a%2Fb/x", None),
    ("/comfy/m//x", "/comfy/m//x", None),
    ("/comfy//x", "/comfy//x", None),
])
def test_r3_03_comfy_split(path, wire, expected):
    """R3-03: comfy_split() splits the decoded path and requires the raw one to agree segment by segment."""
    from colabinet.gateway import ComfyPathError, comfy_split

    if expected is None:
        with pytest.raises(ComfyPathError):
            comfy_split(path, wire)
    else:
        assert comfy_split(path, wire) == expected


# ------------------------------------------------------------------------------------------------------- R3-04
class ListedCLI:
    enabled = True

    def __init__(self, clock):
        self.clock = clock
        self.runtimes = [Runtime("comfy-g4", "gpu-g4-1", "G4", "Standard", "GPU")]

    def command(self):
        return ["colab"]

    async def listing(self):
        return Listing(list(self.runtimes), Usage(500.0, SEED_RATES["G4"][0], len(self.runtimes)), True,
                       self.clock.now())

    async def stop(self, name):
        self.runtimes = [r for r in self.runtimes if r.name != name]
        return CmdResult(0, "[colab] Session terminated.")


@pytest.fixture
async def hung_comfy(store, clock, tmp_path):
    """An engine_down ComfyUI session idle far past idle_min whose /queue probe hangs until released."""
    started, release = asyncio.Event(), asyncio.Event()

    async def hung(req):
        started.set()
        await release.wait()
        raise httpx.ReadTimeout("ComfyUI hung", request=req)

    client = httpx.AsyncClient(transport=httpx.MockTransport(hung))
    sup = Supervisor(settings=Settings(data_dir=tmp_path, daily_cap_cu=0, stop_retry_s=0.001), store=store,
                     cli=ListedCLI(clock), tunnels=Mock(alive=Mock(return_value=False), stop=AsyncMock()),
                     proxies=ProxyManager(), alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=load_presets(REPO_ROOT / "presets.toml"), http=client)
    t0 = clock.t - 4 * 3600
    sid = await store.insert_session(
        account_id="default", colab_name="comfy-g4", endpoint="gpu-g4-1", gpu_requested="G4", gpu_class="G4",
        proxy_port=3080, hidden_port=23001, vm_port=8090, state="engine_down", state_since=clock.t - 60,
        created_at=t0, ready_at=t0, last_activity=t0, rate_cu_h=8.9, rate_source="measured", idle_min=30)
    spec = sup._run_spec(SimpleNamespace(refusal_projection=None, preset_id="comfy-image-uncensored", gpu_class="G4",
                                         ctx=None, kv=None, parallel=None, extra_args=()))
    await store.insert_run(session_id=sid, engine="comfyui", preset_id="comfy-image-uncensored", repo=spec.repo,
                           alias=spec.alias, ctx=spec.ctx, kv=spec.kv, parallel=1, started_at=t0, status="ready",
                           spec_json=json.dumps(dataclasses.asdict(spec)))
    sup._activity[sid] = Activity(t0)
    sup._start_driver = Mock()
    yield SimpleNamespace(sup=sup, sid=sid, started=started, release=release)
    release.set()
    await client.aclose()


@pytest.mark.parametrize("action", ["retry", "reconfigure"])
async def test_r3_04_recovery_clicked_during_the_queue_probe_is_not_idle_stopped(hung_comfy, store, action):
    """R3-04: a retry-setup or reconfigure clicked while guards_once() waits on the /queue probe of an engine_down
    session is not stopped with the stale engine_down snapshot; the session is skipped for this tick."""
    h = hung_comfy
    guard = asyncio.create_task(h.sup.guards_once())
    await h.started.wait()
    if action == "retry":
        await h.sup.decide(h.sid, "retry")
    else:
        await h.sup.reconfigure(h.sid)
    during = (await store.session(h.sid))["state"]
    assert during in ("setting_up", "reconfiguring")
    h.release.set()
    await guard
    row = await store.session(h.sid)
    assert row["state"] == during and row["stop_reason"] is None
    assert await store.db.fetchall("SELECT * FROM events WHERE kind = 'guard'") == []


async def test_r3_04_unchanged_session_is_still_idle_stopped(hung_comfy, store):
    """R3-04: without a concurrent change the idle stop still happens after the (failed) probe."""
    h = hung_comfy
    h.release.set()
    await h.sup.guards_once()
    row = await store.session(h.sid)
    assert row["state"] == "stopped" and row["stop_reason"] == "idle"


async def test_r3_04_idle_stop_refuses_a_session_that_left_the_idle_states(hung_comfy, store):
    """R3-04: stop(sid, "idle") itself refuses a session that is no longer idle-guarded (setting_up)."""
    h = hung_comfy
    await h.sup.decide(h.sid, "retry")
    await h.sup.stop(h.sid, "idle")
    assert (await store.session(h.sid))["state"] == "setting_up"


async def test_r3_01_watermark_is_cleared_by_reconfigure_and_stop(hung_comfy, store):
    """R3-01: the busy watermark belongs to the current run: reconfigure and stop clear it."""
    h = hung_comfy
    h.release.set()
    h.sup._queue_busy_floor[h.sid] = 123.0
    await h.sup.reconfigure(h.sid)
    assert h.sid not in h.sup._queue_busy_floor
    await store.update_session(h.sid, state="ready")
    h.sup._queue_busy_floor[h.sid] = 123.0
    await h.sup.stop(h.sid, "user")
    assert h.sid not in h.sup._queue_busy_floor and (await store.session(h.sid))["state"] == "stopped"


# ------------------------------------------------------------------------------------------------------- R3-05
class ExtCLI:
    """Scriptable external runtimes; `stop_fails` makes `colab stop` fail (optionally removing the runtime)."""

    enabled = True

    def __init__(self, clock):
        self.clock = clock
        self.runtimes: list[Runtime] = []
        self.stopped: list[str] = []
        self.stop_fails = False
        self.gone_on_failed_stop = False

    def command(self):
        return ["colab"]

    async def listing(self):
        return Listing(list(self.runtimes), Usage(500.0, SEED_RATES["G4"][0], len(self.runtimes)), True,
                       self.clock.now())

    async def exec_code(self, name, code, timeout=None):
        return CmdResult(0, "1\n")

    async def stop(self, name):
        if self.stop_fails:
            if self.gone_on_failed_stop:
                self.runtimes = [r for r in self.runtimes if r.name != name]
            return CmdResult(1, f"Session {name} not found")
        self.stopped.append(name)
        self.runtimes = [r for r in self.runtimes if r.name != name]
        return CmdResult(0, "[colab] Session terminated.")


@pytest.fixture
async def ext(store, clock, tmp_path):
    cli = ExtCLI(clock)
    notifier = NullNotifier()
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=cli, tunnels=None,
                     proxies=ProxyManager(), alerts=Alerts(notifier, clock.now), clock=clock,
                     presets=PresetError("unused"), http=object())
    app = create_app(sup, "tok", 7860, csrf="csrf-1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as web:
        yield SimpleNamespace(sup=sup, cli=cli, notifier=notifier, web=web)


def g4(name, endpoint):
    return Runtime(name, endpoint, "G4", "Standard", "GPU")


def find(sup, key):
    return next((a for a in sup.alerts.list() if a.key == key), None)


def attention(sup):
    return make_templates().env.get_template("partials/overview_attention.html").render(
        overview_data={"alerts": [dataclasses.asdict(a) for a in sup.alerts.list()], "account_id": "default"})


def stop_button(html):
    m = re.search(r'hx-post="(/ui/actions/external/[^"]+)"\s+hx-vals=\'([^\']+)\'', html)
    assert m, html
    return m.group(1), json.loads(m.group(2).replace("&#34;", '"'))


async def events(store, kind):
    return await store.db.fetchall("SELECT * FROM events WHERE kind = ? ORDER BY id", (kind,))


async def test_r3_05_alert_stop_button_carries_the_endpoint(ext, clock):
    """R3-05: the external_autostop alert's 「停止」 posts the endpoint (and a URL-encoded name), so a click after the
    runtime was re-created under the same name is refused instead of stopping the new VM."""
    e = ext
    e.cli.runtimes = [g4("ext g4", "gpu-ep1")]
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-ep1", True, stop_after_min=5)
    e.cli.stop_fails = True  # e.g. an OAuth outage: the auto-stop fails while the VM keeps billing
    clock.advance(301)
    await e.sup.keepalive_externals_once()
    stop = find(e.sup, "external_autostop:gpu-ep1")
    assert stop is not None and stop.endpoint == "gpu-ep1" and stop.level == "critical"
    url, vals = stop_button(attention(e.sup).split("外部运行时自动停止失败")[1])
    assert url == "/ui/actions/external/ext%20g4/stop" and vals["endpoint"] == "gpu-ep1"
    assert find(e.sup, "external:gpu-ep1").endpoint == "gpu-ep1"  # the external_runtime button too
    e.cli.stop_fails = False
    e.cli.runtimes = [g4("ext g4", "gpu-ep2")]  # stopped and re-created under the same name
    r = await e.web.post(url, headers={"X-CSRF-Token": "csrf-1"}, data={**vals, "account_id": "default"})
    assert r.status_code == 409 and e.cli.stopped == []


async def test_r3_05_already_stopped_runtime_is_not_a_critical_alert(ext, store, clock):
    """R3-05: a kept-alive runtime stopped elsewhere before the next reconcile is treated as stopped when its timer
    expires: no CRITICAL alert or notification, keepalive cleared, external_stopped(reason=already_gone)."""
    e = ext
    e.cli.runtimes = [g4("done-g4", "gpu-done")]
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-done", True, stop_after_min=5)
    e.notifier.sent.clear()
    e.cli.runtimes = []
    clock.advance(301)
    await e.sup.keepalive_externals_once()
    # only the plain external_runtime card alert is left, for the next reconcile (as after any external stop)
    assert [a.key for a in e.sup.alerts.list() if a.key.endswith(":gpu-done")] == ["external:gpu-done"]
    await e.sup.reconcile_once()
    assert not [a for a in e.sup.alerts.list() if a.key.endswith(":gpu-done")]
    assert e.notifier.sent == [] and e.cli.stopped == []
    assert not (await store.external("gpu-done"))["keepalive"]
    stopped = await events(store, "external_stopped")
    assert len(stopped) == 1 and json.loads(stopped[0]["detail_json"])["reason"] == "already_gone"


async def test_r3_05_failed_stop_of_a_vanished_runtime_counts_as_stopped(ext, store, clock):
    """R3-05: `colab stop` failing because the runtime is gone (absent from the next listing) is a success."""
    e = ext
    e.cli.runtimes = [g4("ext-g4", "gpu-ep1")]
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-ep1", True, stop_after_min=5)
    e.cli.stop_fails = e.cli.gone_on_failed_stop = True
    clock.advance(301)
    await e.sup.keepalive_externals_once()
    assert find(e.sup, "external_autostop:gpu-ep1") is None
    assert not (await store.external("gpu-ep1"))["keepalive"]
    assert json.loads((await events(store, "external_stopped"))[0]["detail_json"])["reason"] == "already_gone"
    # but a failure while the VM is still listed keeps the R1-03 / R2-03 behaviour (alert, retry)
    e.cli.runtimes = [g4("ext-2", "gpu-ep2")]
    e.cli.gone_on_failed_stop = False
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-ep2", True, stop_after_min=5)
    clock.advance(301)
    await e.sup.keepalive_externals_once()
    assert find(e.sup, "external_autostop:gpu-ep2").level == "critical"
    assert (await store.external("gpu-ep2"))["keepalive"]


async def test_r3_05_runtime_without_a_name_gets_a_warning_without_a_dead_button(ext, store, clock):
    """R3-05: a kept-alive runtime that lost its CLI name gets no stop button that can only fail: a non-critical
    external_unnamed warning explains it, offers 「取消接管保活」 (also on the card), and turning keepalive off
    clears it; it also clears when the runtime disappears."""
    e = ext
    e.cli.runtimes = [g4("lost-g4", "gpu-lost")]
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-lost", True, stop_after_min=5)
    e.notifier.sent.clear()
    e.cli.runtimes = [g4(None, "gpu-lost")]
    clock.advance(301)
    await e.sup.keepalive_externals_once()  # before the reconcile sees the lost name
    await e.sup.reconcile_once()
    await e.sup.keepalive_externals_once()
    assert find(e.sup, "external_autostop:gpu-lost") is None and e.cli.stopped == []
    warn = find(e.sup, "external_unnamed:gpu-lost")
    assert warn is not None and warn.level == "warning" and "没有本地 CLI 名称" in warn.message
    assert warn.actions == ["keepalive-off"] and warn.endpoint == "gpu-lost"
    assert not any("lost-g4" in msg for _, msg in e.notifier.sent)
    html = attention(e.sup)
    assert "外部运行时已失去名称" in html and "取消接管保活" in html and "停止 lost-g4" not in html
    card = (await e.web.get("/ui/external")).text
    assert "取消接管保活" in card
    r = await e.web.post("/ui/actions/external-keepalive", headers={"X-CSRF-Token": "csrf-1"},
                         data={"endpoint": "gpu-lost", "enabled": "0", "account_id": "default"})
    assert r.status_code == 200 and find(e.sup, "external_unnamed:gpu-lost") is None
    await e.sup.keepalive_externals_once()
    assert find(e.sup, "external_unnamed:gpu-lost") is None  # not raised again once keepalive is off
    # and a warning left for a runtime that then disappears goes away with the reconcile
    e.cli.runtimes = [g4("x", "gpu-x")]
    await e.sup.reconcile_once()
    await e.sup.external_keepalive("gpu-x", True, stop_after_min=0)
    e.cli.runtimes = [g4(None, "gpu-x")]
    await e.sup.reconcile_once()
    await e.sup.keepalive_externals_once()
    assert find(e.sup, "external_unnamed:gpu-x") is not None
    e.cli.runtimes = []
    await e.sup.reconcile_once()
    assert find(e.sup, "external_unnamed:gpu-x") is None
