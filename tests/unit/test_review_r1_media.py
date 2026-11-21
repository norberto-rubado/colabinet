"""Regression tests for review round 1: media polling, activity release, negative prompts and request stats."""
import asyncio
import base64
import errno
import json
import sqlite3
from types import SimpleNamespace

import httpx
import pytest

from colabinet.api import create_app
from colabinet.comfy_graphs import NEGATIVE_WORKFLOWS, WORKFLOWS, ImageRequest, VideoRequest, image_graph, template
from colabinet.config import Settings
from colabinet.db import SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, Database
from colabinet.gateway import Target
from colabinet.media import MediaError, MediaService
from tests.unit.test_media_gateway import AUTH, MP4, PNG, media  # noqa: F401

VIDEO = Target(2, "vid", "comfy-video-h3", 3080, 1, "comfyui", "video")
IMAGE = Target(1, "img", "comfy-image", 3081, 1, "comfyui", "image")
DONE = {"status": {"completed": True, "status_str": "success"},
        "outputs": {"save": {"images": [{"filename": "out.mp4", "subfolder": "", "type": "output"},
                                        {"filename": "out.png", "subfolder": "", "type": "output"}]}}}


class Comfy:
    """Scriptable ComfyUI: history/queue/view behaviour per test, every request recorded."""

    def __init__(self, history=None, queue=None, view=None):
        self.calls = []
        self.history = history or (lambda n: {"p1": DONE})
        self.queue = queue or (lambda n: {"queue_running": [], "queue_pending": []})
        self.view = view or (lambda n: None)
        self.counts = {}

    def count(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def __call__(self, req):
        self.calls.append(req)
        path = req.url.path
        if path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1", "node_errors": {}})
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "in.png", "subfolder": "", "type": "input"})
        if path == "/models/loras":
            return httpx.Response(200, json=["extra.safetensors"])
        if path == "/history/p1":
            result = self.history(self.count("history"))
            return result if isinstance(result, httpx.Response) else httpx.Response(200, json=result)
        if path == "/queue":
            return httpx.Response(200, json=self.queue(self.count("queue")))
        if path == "/view":
            failed = self.view(self.count("view"))
            if failed:
                return failed
            is_mp4 = req.url.params["filename"].endswith(".mp4")
            return httpx.Response(200, content=MP4 if is_mp4 else PNG,
                                  headers={"content-type": "video/mp4" if is_mp4 else "image/png"})
        raise AssertionError(path)


@pytest.fixture
async def world(tmp_path):
    made = []

    async def make(comfy, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(comfy))
        sup = SimpleNamespace(s=Settings(data_dir=tmp_path), _activity={}, now=lambda: 1000.0, http=client)
        options = {"poll_s": 0.002, "timeout_s": 5, "failure_window_s": 1.0, "backoff_max_s": 0.01,
                   "fetch_backoff_s": 0.001, **kwargs}
        service = MediaService(sup, client=client, **options)
        made.append((service, client))
        return SimpleNamespace(sup=sup, service=service, comfy=comfy)

    yield make
    for service, client in made:
        await service.close()
        await client.aclose()


async def wait_video(service, ident, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        job = service.video_status(ident)
        if job["status"] in ("completed", "failed"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"video job still {job}")


# R1-06 ---------------------------------------------------------------------------------------------------------


async def test_r1_06_vanished_video_fails_fast_and_releases_activity(world):
    """R1-06: a video prompt in neither /history nor /queue fails at once (not after timeout_s) and frees the guard."""
    w = await world(Comfy(history=lambda n: {}), timeout_s=3600)
    act = w.service.activity(VIDEO)
    act.last = 5.0
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "failed" and "队列与历史中均未找到该任务" in job["error"]
    assert w.comfy.counts["history"] == 2 and w.comfy.counts["queue"] == 1  # one re-check of /history, then fail
    assert not act._inflight and act.last == 5.0  # released with useful=False


async def test_r1_06_vanished_image_raises_chinese_error_and_releases_activity(world):
    """R1-06: image jobs also consult /queue; a vanished prompt raises MediaError and is not counted as useful."""
    w = await world(Comfy(history=lambda n: {}), timeout_s=3600)
    act = w.service.activity(IMAGE)
    act.last = 5.0
    with pytest.raises(MediaError, match="引擎可能已重启或任务已被取消") as caught:
        await w.service.images(IMAGE, ImageRequest(model="m", prompt="x"))
    assert caught.value.status == 502 and w.comfy.counts["queue"] == 1
    assert not act._inflight and act.last == 5.0


async def test_r1_06_finished_between_history_and_queue_is_not_vanished(world):
    """R1-06: if the prompt completes between the /history and /queue requests, the re-check returns the result."""
    w = await world(Comfy(history=lambda n: {} if n == 1 else {"p1": DONE}))
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "completed"
    assert w.comfy.counts == {"history": 2, "queue": 1, "view": 1}


async def test_r1_06_pending_prompt_keeps_waiting(world):
    """R1-06: a prompt still in queue_pending is waited for (status queued), not declared vanished."""
    w = await world(Comfy(history=lambda n: {} if n < 4 else {"p1": DONE},
                          queue=lambda n: {"queue_running": [[0, "other"]], "queue_pending": [[1, "p1"]]}))
    seen, original = [], w.service.files.write_json

    def record(ident, job):
        seen.append(job.get("status"))
        original(ident, job)

    w.service.files.write_json = record
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "completed", job
    assert "queued" in seen and w.comfy.counts["queue"] == 3


# R1-07 ---------------------------------------------------------------------------------------------------------


async def test_r1_07_history_errors_tolerated_within_window(world):
    """R1-07: >10 consecutive /history 502s (a tunnel reconnect) no longer kill a job that then completes."""
    w = await world(Comfy(history=lambda n: httpx.Response(502) if n <= 25 else {"p1": DONE}))
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "completed" and w.comfy.counts["history"] == 26


async def test_r1_07_history_errors_fail_after_window(world):
    """R1-07: errors lasting longer than failure_window_s still fail the job (capped backoff, not forever)."""
    w = await world(Comfy(history=lambda n: httpx.Response(502)), failure_window_s=0.1, backoff_max_s=0.02)
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "failed" and "ComfyUI 请求失败" in job["error"]
    assert 3 <= w.comfy.counts["history"] < 40  # backed off: far fewer than 0.1 s / poll_s polls
    assert not w.sup._activity[VIDEO.sid]._inflight


def test_r1_07_default_failure_window_is_about_90s(tmp_path):
    """R1-07: production default tolerates ~90 s of poll errors and /view gets 3 retries."""
    sup = SimpleNamespace(s=Settings(data_dir=tmp_path), _activity={}, now=lambda: 1.0, http=object())

    async def build():
        service = MediaService(sup)
        await service.close()
        return service

    service = asyncio.run(build())
    assert service.failure_window_s == 90 and service.fetch_retries == 3


async def test_r1_07_view_fetch_retries(world):
    """R1-07: /view download retries transient 502s instead of failing a finished render."""
    w = await world(Comfy(view=lambda n: httpx.Response(502) if n <= 2 else None))
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "completed" and w.comfy.counts["view"] == 3


async def test_r1_07_view_fetch_gives_up_after_retries(world):
    """R1-07: /view retries are bounded (1 try + 3 retries) and then fail with the Chinese download error."""
    w = await world(Comfy(view=lambda n: httpx.Response(503)))
    job = await wait_video(w.service, (await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x")))["id"])
    assert job["status"] == "failed" and job["error"] == "无法下载 ComfyUI 输出文件" and w.comfy.counts["view"] == 4


# R1-08 ---------------------------------------------------------------------------------------------------------


def enospc(*_args, **_kwargs):
    raise OSError(errno.ENOSPC, "No space left on device")


async def test_r1_08_create_video_status_write_failure_releases_activity(world):
    """R1-08(1): ENOSPC on the initial status write in create_video propagates but releases the activity key."""
    w = await world(Comfy())
    w.service.files.write_json = enospc
    with pytest.raises(OSError):
        await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x"))
    assert not w.sup._activity[VIDEO.sid]._inflight
    assert not [t for t in w.service.tasks if t.get_coro().__name__ == "_video"]


async def test_r1_08_create_video_spawn_failure_releases_activity(world):
    """R1-08(1): a failing spawn (after the status write) also releases the activity key."""
    w = await world(Comfy())

    def broken_spawn(coroutine):
        coroutine.close()
        raise RuntimeError("spawn failed")

    w.service.spawn = broken_spawn
    with pytest.raises(RuntimeError):
        await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x"))
    assert not w.sup._activity[VIDEO.sid]._inflight


async def test_r1_08_video_finally_releases_activity_even_if_status_write_fails(world):
    """R1-08(2): save() and the final write_json both hit ENOSPC; act.end still runs and no exception escapes."""
    w = await world(Comfy())
    real, calls = w.service.files.write_json, []

    def first_only(ident, job):
        calls.append(job.get("status"))
        if len(calls) > 1:
            enospc()
        return real(ident, job)

    w.service.files.write_json, w.service.files.save = first_only, enospc
    await w.service.create_video(VIDEO, VideoRequest(model="m", prompt="x"))
    tasks = [t for t in w.service.tasks if t.get_coro().__name__ == "_video"]
    await asyncio.wait_for(asyncio.gather(*tasks), 2)  # finished normally: the OSError in finally was suppressed
    assert calls[-1] == "failed" and not w.sup._activity[VIDEO.sid]._inflight


async def playground_image(media, images):  # noqa: F811
    media.service.images = images
    app = create_app(media.sup, "local-api-key", 7860, csrf="csrf-test")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as c:
        return await c.post("/ui/media/image", headers={"X-CSRF-Token": "csrf-test"},
                            json={"session_id": 1, "prompt": "a cat", "workflow": "anima"})


async def test_r1_08_web_stream_image_oserror_returns_json_error(media):  # noqa: F811
    """R1-08(3): stream_image turns an OSError after the 200 into a JSON error body and ends the stream."""
    async def images(target, body, origin="gateway"):
        enospc()

    r = await playground_image(media, images)
    assert r.status_code == 200 and "媒体文件处理失败" in r.json()["error"]


async def test_r1_08_web_stream_image_cancelled_job_returns_json_error(media):  # noqa: F811
    """R1-08(3): a job cancelled by service shutdown ends the playground stream with a JSON error, not a cut."""
    async def images(target, body, origin="gateway"):
        raise asyncio.CancelledError

    r = await playground_image(media, images)
    assert r.status_code == 200 and r.json()["error"] == "服务已关闭，生图任务已取消"


# R1-10 ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_r1_10_blank_negative_prompt_is_none_and_graph_untouched(workflow, blank):
    """R1-10: ""/whitespace negative_prompt normalises to None: accepted everywhere, template graph unchanged."""
    body = ImageRequest(model="m", prompt="a cat", workflow=workflow, negative_prompt=blank)
    assert body.negative_prompt is None
    image = "in.png" if workflow == "h3edit" else None
    assert image_graph(body, 1, image) == image_graph(ImageRequest(model="m", prompt="a cat", workflow=workflow), 1,
                                                      image)


def test_r1_10_moss_default_negative_kept_and_explicit_one_applied():
    """R1-10: moss keeps its default quality negative for "", and still takes a real negative prompt."""
    default = image_graph(ImageRequest(model="m", prompt="cat", workflow="moss", negative_prompt=""), 1)
    assert "worst quality" in default["5"]["inputs"]["text"]
    custom = image_graph(ImageRequest(model="m", prompt="cat", workflow="moss", negative_prompt="blurry"), 1)
    assert custom["5"]["inputs"]["text"] == "blurry"


@pytest.mark.parametrize("workflow", sorted(set(WORKFLOWS) - set(NEGATIVE_WORKFLOWS)))
def test_r1_10_cfg1_graph_never_rewrites_conditioning_zero_out(workflow):
    """R1-10: even an unvalidated body cannot turn a cfg=1 ConditioningZeroOut into CLIPTextEncode."""
    with pytest.raises(ValueError, match="负面提示词不会生效"):
        ImageRequest(model="m", prompt="x", workflow=workflow, negative_prompt="bad")
    body = ImageRequest.model_construct(**{**ImageRequest(model="m", prompt="x", workflow=workflow).model_dump(),
                                           "negative_prompt": "bad"})
    graph = image_graph(body, 1, "in.png" if workflow == "h3edit" else None)
    original = template(workflow)
    for ident, node in original.items():
        if node["class_type"] == "ConditioningZeroOut":
            assert graph[ident]["class_type"] == "ConditioningZeroOut"
    assert "bad" not in json.dumps(graph)


# R1-14 ---------------------------------------------------------------------------------------------------------


def agents(comfy):
    return {(req.url.path, req.headers["user-agent"]) for req in comfy.calls}


async def test_r1_14_gateway_media_user_agents(media):  # noqa: F811
    """R1-14(3): gateway jobs use colabinet-gateway for /prompt, /upload, /view; polling uses colabinet-poll."""
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={
        "model": "images", "prompt": "edit", "workflow": "h3edit", "image": base64.b64encode(PNG).decode()})
    assert r.status_code == 200, r.text
    r = await media.c.post("/v1/videos", headers=AUTH, json={
        "model": "videos", "prompt": "x", "loras": [{"name": "extra.safetensors", "strength": 1}]})
    for _ in range(100):
        if (await media.c.get(f"/v1/videos/{r.json()['id']}", headers=AUTH)).json()["status"] == "completed":
            break
        await asyncio.sleep(.01)
    seen = agents(media.comfy)
    for path in ("/prompt", "/upload/image", "/view"):
        assert {ua for p, ua in seen if p == path} == {"colabinet-gateway"}, path
    polled = {ua for p, ua in seen if p.startswith("/history/") or p in ("/queue", "/models/loras")}
    assert polled == {"colabinet-poll"}


async def test_r1_14_playground_media_user_agents(media):  # noqa: F811
    """R1-14(3): playground image and video jobs mark /prompt and /view as colabinet-playground."""
    app = create_app(media.sup, "local-api-key", 7860, csrf="csrf-test")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as c:
        r = await c.post("/ui/media/image", headers={"X-CSRF-Token": "csrf-test"},
                         json={"session_id": 1, "prompt": "a cat", "workflow": "anima"})
        assert r.status_code == 200 and r.json()["data"]
        r = await c.post("/ui/media/video", headers={"X-CSRF-Token": "csrf-test"},
                         json={"session_id": 2, "prompt": "a stream"})
        for _ in range(100):
            if (await c.get(f"/ui/media/videos/{r.json()['id']}")).json()["status"] == "completed":
                break
            await asyncio.sleep(.01)
    seen = agents(media.comfy)
    for path in ("/prompt", "/view"):
        assert {ua for p, ua in seen if p == path} == {"colabinet-playground"}, path


async def seed_requests(db):
    from colabinet.store import Store

    store = Store(db)
    sid = await store.insert_session(colab_name="s", gpu_requested="G4", gpu_class="G4", proxy_port=3080,
                                     hidden_port=23000, vm_port=8090, state="ready", state_since=1.0, created_at=1.0,
                                     rate_cu_h=8.9, rate_source="measured", idle_min=30)
    rid = await store.insert_run(session_id=sid, engine="comfyui", alias="m", started_at=1.0)
    for client in ("colabinet-playground", "colabinet-gateway", "colabinet-poll", "pi/0.99", None, "colabinet-media"):
        await store.insert_request({"run_id": rid, "ts": 10.0, "method": "POST", "path": "/prompt", "status": 200,
                                    "client": client})


async def test_r1_14_request_stats_classification(tmp_path):
    """R1-14(3): only colabinet-playground is is_playground; gateway/other clients count; poll rows are hidden."""
    db = Database(tmp_path / "s.db")
    db.init()
    try:
        await seed_requests(db)
        rows = {r["client"]: r["is_playground"] for r in await db.fetchall("SELECT client, is_playground "
                                                                           "FROM request_stats")}
        assert rows == {"colabinet-playground": 1, "colabinet-gateway": 0, "pi/0.99": 0, None: 0,
                        "colabinet-media": 0}
        assert (await db.fetchone("SELECT COUNT(*) AS n FROM requests"))["n"] == 6  # raw rows are kept
    finally:
        db.close()


async def test_r1_14_existing_database_view_is_rebuilt(tmp_path):
    """R1-14(3): a v5 database created with the old LIKE 'colabinet-%' view gets the new classification on init."""
    path = tmp_path / "v5.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1 + SCHEMA_V2 + SCHEMA_V3 + SCHEMA_V4)
        assert "LIKE 'colabinet-%'" in connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'request_stats'").fetchone()[0]
        connection.execute("PRAGMA user_version = 5")
    db = Database(path)
    db.init()
    db.init()  # idempotent
    try:
        await seed_requests(db)
        rows = {r["client"]: r["is_playground"] for r in await db.fetchall("SELECT client, is_playground "
                                                                           "FROM request_stats")}
        assert rows["colabinet-gateway"] == 0 and rows["colabinet-playground"] == 1 and "colabinet-poll" not in rows
    finally:
        db.close()
