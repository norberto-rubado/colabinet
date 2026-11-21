import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from colabinet.config import Settings
from colabinet.gateway import Gateway
from colabinet.media import MediaError, MediaFiles, MediaService, decode_image, output_refs

PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"
MP4 = b"\x00\x00\x00\x18ftypmp42test-video-with-audio"
KEY = "media-test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class Store:
    async def live_sessions(self):
        return [{"id": 1, "colab_name": "image-session", "proxy_port": 3001, "state": "ready"},
                {"id": 2, "colab_name": "video-session", "proxy_port": 3002, "state": "ready"}]

    async def current_run(self, sid):
        return {"alias": "images" if sid == 1 else "videos", "engine": "comfyui",
                "spec_json": json.dumps({"kind": "image" if sid == 1 else "video"})}


class Comfy:
    def __init__(self):
        self.calls = []
        self.graphs = []
        self.histories = {}
        self.delay = 0
        self.error = False
        self.pending = False
        self.release = asyncio.Event()

    async def __call__(self, req):
        self.calls.append(req)
        assert "authorization" not in req.headers and "x-api-key" not in req.headers
        path = req.url.path
        if path == "/upload/image":
            assert b"Content-Disposition: form-data" in req.content and PNG in req.content
            return httpx.Response(200, json={"name": "uploaded.png", "subfolder": "colabinet", "type": "input"})
        if path == "/models/loras":
            return httpx.Response(200, json=["extra.safetensors"])
        if path == "/prompt":
            graph = json.loads(req.content)["prompt"]
            self.graphs.append(graph)
            ident = str(len(self.graphs))
            self.histories[ident] = graph
            return httpx.Response(200, json={"prompt_id": ident, "node_errors": {}})
        if path.startswith("/history/"):
            if self.delay:
                await asyncio.sleep(self.delay)
            ident = path.rsplit("/", 1)[1]
            if self.pending and not self.release.is_set():
                return httpx.Response(200, json={})
            video = any(n["class_type"] == "SaveVideo" for n in self.histories[ident].values())
            return httpx.Response(200, json={ident: {
                "status": {"completed": True, "status_str": "error" if self.error else "success", "messages": []},
                "outputs": {"save": {"images": [{"filename": "clip.mp4" if video else "image.png",
                                                "subfolder": "test", "type": "output"}]}}}})
        if path == "/view":
            assert req.url.params["subfolder"] == "test" and req.url.params["type"] == "output"
            video = req.url.params["filename"].endswith(".mp4")
            return httpx.Response(200, content=MP4 if video else PNG,
                                  headers={"content-type": "video/mp4" if video else "image/png"})
        if path == "/queue":
            return httpx.Response(200, json={"queue_running": [[0, str(len(self.graphs))]], "queue_pending": []})
        if path == "/object_info":
            return httpx.Response(200, json={"UNETLoader": {"input": {}}})
        return httpx.Response(200, content=req.content,
                              headers={"content-type": req.headers.get("content-type", "text/plain")})


@pytest.fixture
async def media(tmp_path):
    comfy = Comfy()
    async def streaming(req):
        response = await comfy(req)
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=httpx.ByteStream(response.content))
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(streaming))
    sup = SimpleNamespace(store=Store(), s=Settings(data_dir=tmp_path, public_api_base="https://api.example/v1"),
                          _activity={}, now=lambda: 1000, http=upstream)
    sup._media = MediaService(sup, poll_s=.01, timeout_s=2)
    gateway = Gateway(sup, KEY, client=upstream, cache_s=0, grace=.02, interval=.01)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="https://api.example") as c:
        yield SimpleNamespace(c=c, comfy=comfy, sup=sup, service=sup._media, gateway=gateway)
    await gateway.aclose()
    await upstream.aclose()


@pytest.mark.parametrize("path,method", [("/v1/images/generations", "POST"), ("/v1/videos", "POST"),
    ("/v1/videos/video_" + "0" * 32, "GET"), ("/v1/files/file_" + "0" * 32 + "/content", "GET"),
    ("/v1/videos/video_" + "0" * 32 + "/content", "GET"), ("/comfy/images/prompt", "POST"),
    ("/comfy/images/object_info", "GET")])
async def test_all_media_endpoints_need_auth(media, path, method):
    r = await media.c.request(method, path, json={})
    assert r.status_code == 401 and not media.comfy.calls


async def test_models_have_media_kind(media):
    r = await media.c.get("/v1/models", headers=AUTH)
    assert {x["id"]: x["kind"] for x in r.json()["data"]} == {"videos": "video", "images": "image"}


@pytest.mark.parametrize("model", ["images", "image-session"])
async def test_images_prompt_history_view_b64_n_seed(media, model):
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={
        "model": model, "prompt": "a landscape", "n": 2, "seed": 42, "workflow": "anima"})
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 2
    assert all(base64.b64decode(i["b64_json"]) == PNG for i in r.json()["data"])
    assert len(media.comfy.graphs) == 2
    assert [g["76"]["inputs"]["seed"] for g in media.comfy.graphs] == [42, 43]
    assert media.sup._activity[1].last == 1000 and not media.sup._activity[1]._inflight


async def test_url_files_auth_and_expiry(media):
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={
        "model": "images", "prompt": "x", "response_format": "url"})
    url = r.json()["data"][0]["url"]
    assert url.startswith("https://api.example/v1/files/file_") and url.endswith("/content")
    assert (await media.c.get(url)).status_code == 401
    got = await media.c.get(url, headers=AUTH)
    assert got.content == PNG and got.headers["content-type"].startswith("image/png")
    ident = url.split("/")[-2]
    meta = media.service.files.read(ident)
    media.service.files.now = lambda: meta["expires_at"] + 1
    assert (await media.c.get(url, headers=AUTH)).status_code == 404
    assert not media.service.files.path(ident, "bin").exists()


async def test_edit_upload_and_upscale(media):
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={
        "model": "images", "prompt": "change the light", "workflow": "h3edit", "upscale": True,
        "image": "data:image/png;base64," + base64.b64encode(PNG).decode()})
    assert r.status_code == 200, r.text
    assert len(media.comfy.graphs) == 2
    assert media.comfy.graphs[0]["20"]["inputs"]["image"] == "colabinet/uploaded.png"
    assert media.comfy.graphs[1]["52"]["inputs"]["unet_name"].startswith("seedvr2")
    assert len([r for r in media.comfy.calls if r.url.path == "/upload/image"]) == 2


@pytest.mark.parametrize("payload,status", [({"model": "wrong", "prompt": "x"}, 404),
    ({"model": "videos", "prompt": "x"}, 404), ({"model": "images", "prompt": "x", "n": 0}, 400),
    ({"model": "images", "prompt": "x", "workflow": "h3edit"}, 400),
    ({"model": "images", "prompt": "x", "image": "a"}, 400)])
async def test_media_validation_errors(media, payload, status):
    r = await media.c.post("/v1/images/generations", headers=AUTH, json=payload)
    assert r.status_code == status and "error" in r.json(), r.text
    assert not media.comfy.graphs


async def test_images_cold_start_keepalive_and_error(media):
    media.comfy.delay = .08
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={"model": "images", "prompt": "x"})
    assert r.status_code == 200 and r.content.startswith(b" ")
    assert r.json()["data"]
    media.comfy.error = True
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={"model": "images", "prompt": "x"})
    assert r.status_code == 200 and r.json()["error"]["type"] == "media_error"
    assert not media.sup._activity[1]._inflight


async def finish_video(media, ident):
    for _ in range(100):
        response = await media.c.get(f"/v1/videos/{ident}", headers=AUTH)
        if response.json()["status"] in ("completed", "failed"):
            return response.json()
        await asyncio.sleep(.01)
    raise AssertionError("video worker did not finish")


@pytest.mark.parametrize("image", [None, base64.b64encode(PNG).decode()])
async def test_videos_async_status_content_with_audio_graph(media, image):
    media.comfy.pending = True
    r = await media.c.post("/v1/videos", headers=AUTH, json={
        "model": "video-session", "prompt": "a stream and birds singing", "image": image,
        "loras": [{"name": "extra.safetensors", "strength": .5}]})
    assert r.status_code == 200, r.text
    ident = r.json()["id"]
    assert r.json() == {"id": ident, "status": "queued"}
    assert media.sup._activity[2]._inflight
    assert (await media.c.get(f"/v1/videos/{ident}/content", headers=AUTH)).status_code == 409
    await asyncio.sleep(.02)
    running = (await media.c.get(f"/v1/videos/{ident}", headers=AUTH)).json()
    assert running["status"] == "in_progress" and running["progress"] == 50
    media.comfy.release.set()
    job = await finish_video(media, ident)
    assert job == {"id": ident, "status": "completed", "progress": 100, "error": None}
    got = await media.c.get(f"/v1/videos/{ident}/content", headers=AUTH)
    assert got.content == MP4 and got.headers["content-type"].startswith("video/mp4")
    graph = media.comfy.graphs[0]
    assert ("first_frame" in graph["i2v"]["inputs"]) == bool(image)
    assert graph["video"]["inputs"]["audio"] == ["adec", 0]
    assert not media.sup._activity[2]._inflight


async def test_video_unknown_lora_rejected_without_submit(media):
    r = await media.c.post("/v1/videos", headers=AUTH, json={"model": "videos", "prompt": "x",
        "loras": [{"name": "not-installed.safetensors", "strength": 1}]})
    assert r.status_code == 400 and not media.comfy.graphs
    assert not media.sup._activity[2]._inflight  # failure path releases the idle guard


async def test_video_first_frame_upload_is_idle_protected(media):
    # The upload is not an inference path and the Comfy queue is still empty: activity must already be held.
    seen = []
    original = media.service.upload_bytes

    async def upload_bytes(target, *args, **kwargs):
        seen.append(bool(media.sup._activity[target.sid]._inflight))
        return await original(target, *args, **kwargs)

    media.service.upload_bytes = upload_bytes
    r = await media.c.post("/v1/videos", headers=AUTH, json={
        "model": "videos", "prompt": "x", "image": base64.b64encode(PNG).decode()})
    assert r.status_code == 200, r.text
    assert seen == [True]
    await finish_video(media, r.json()["id"])
    assert not media.sup._activity[2]._inflight


async def test_video_failure_and_restart_durable_status(media):
    media.comfy.error = True
    r = await media.c.post("/v1/videos", headers=AUTH, json={"model": "videos", "prompt": "x"})
    ident = r.json()["id"]
    job = await finish_video(media, ident)
    assert job["status"] == "failed" and job["error"]
    new = MediaService(media.sup, root=media.service.files.root)
    assert new.video_status(ident) == job
    meta = new.files.read(ident)
    meta.update(status="in_progress", error=None)
    new.files.write_json(ident, meta)
    recovered = MediaService(media.sup, root=new.files.root)
    assert recovered.video_status(ident)["status"] == "failed"
    assert "重启" in recovered.video_status(ident)["error"]
    await new.close()
    await recovered.close()


async def test_raw_passthrough_methods_binary_query_and_credentials(media):
    r = await media.c.get("/comfy/image-session/object_info", headers=AUTH)
    assert r.status_code == 200 and "UNETLoader" in r.json()
    r = await media.c.put("/comfy/images/custom?thing=1", headers={**AUTH, "content-type": "application/octet-stream"},
                          content=b"raw-binary\xff")
    assert r.content == b"raw-binary\xff"
    req = media.comfy.calls[-1]
    assert req.url.path == "/custom" and req.url.params["thing"] == "1"
    assert req.headers["host"] == "127.0.0.1:3001"
    r = await media.c.get("/comfy/no-such-session/queue", headers=AUTH)
    assert r.status_code == 404


async def test_media_history_timeout_releases_activity(media):
    media.comfy.pending = True
    media.service.timeout_s = .02
    r = await media.c.post("/v1/images/generations", headers=AUTH, json={"model": "images", "prompt": "x"})
    assert "error" in r.json()
    assert not media.sup._activity[1]._inflight


@pytest.mark.parametrize("value", ["https://example.com/image.png", "!!!", "data:image/png,xxx", "",
                                    "data:image/jpeg;base64," + base64.b64encode(PNG).decode()])
def test_image_input_validation(value):
    with pytest.raises(MediaError):
        decode_image(value)


def test_files_expire_prune_and_traversal(tmp_path):
    clock = SimpleNamespace(t=1000)
    files = MediaFiles(tmp_path, now=lambda: clock.t)
    ident = files.save(PNG, "image/png")
    assert files.content(ident)[0].read_bytes() == PNG
    for bad in ("../../passwd", "file_../../", "a", "video_" + "0" * 33):
        with pytest.raises(MediaError):
            files.read(bad)
    clock.t += 86401
    files.prune()
    assert not list(tmp_path.iterdir())


async def test_raw_binary_wait_never_inserts_json_keepalive(media):
    original = media.gateway.client

    async def slow(req):
        await asyncio.sleep(.08)
        return httpx.Response(200, stream=httpx.ByteStream(MP4), headers={"content-type": "video/mp4"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        media.gateway.client = client
        result = await media.c.get("/comfy/images/view?filename=clip.mp4", headers=AUTH)
        assert result.status_code == 200 and result.content == MP4
        assert result.headers["content-type"] == "video/mp4"
    media.gateway.client = original


async def test_image_job_survives_request_task_cancellation(media):
    media.comfy.pending = True
    task = asyncio.create_task(media.c.post("/v1/images/generations", headers=AUTH,
        json={"model": "images", "prompt": "x"}))
    for _ in range(20):
        if media.comfy.graphs:
            break
        await asyncio.sleep(.005)
    assert media.sup._activity[1]._inflight
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert media.sup._activity[1]._inflight  # HTTP lifetime is not render lifetime
    media.comfy.release.set()
    for _ in range(30):
        if not media.sup._activity[1]._inflight:
            break
        await asyncio.sleep(.005)
    assert not media.sup._activity[1]._inflight


async def test_gateway_images_through_real_session_proxy(tmp_path):
    import time
    from unittest.mock import AsyncMock

    from colabinet.activity import Activity
    from colabinet.net import first_free, start_server, stop_server
    from colabinet.proxy import Route, SessionProxy

    comfy = Comfy()

    async def streaming(req):
        response = await comfy(req)
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=httpx.ByteStream(response.content))

    async with httpx.AsyncClient(transport=httpx.MockTransport(streaming)) as upstream:
        port = first_free(28200, 28499)
        activity = Activity()
        record = AsyncMock()
        proxy = SessionProxy(port, lambda: Route("forward", 8188, run_id=1), record,
                             activity, upstream, time.time)
        server, task = await start_server(proxy, port)
        sup = SimpleNamespace(s=Settings(data_dir=tmp_path), _activity={1: activity}, now=time.time)
        gw = Gateway(sup, KEY)
        # Exercise network -> per-session proxy Host/Origin checks and metering, no VM.
        sup.store = SimpleNamespace()
        target = __import__('colabinet.gateway', fromlist=['Target']).Target(1, "image", "images", port, 0,
                                                                          "comfyui", "image")
        gw.targets = AsyncMock(return_value=[target])
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw), base_url="https://api.example") as c:
                result = await c.post("/v1/images/generations", headers=AUTH,
                                      json={"model": "images", "prompt": "a cat"})
                assert result.status_code == 200, result.text
                assert base64.b64decode(result.json()["data"][0]["b64_json"]) == PNG
            await asyncio.sleep(.01)
            paths = {call.args[0].path for call in record.call_args_list}
            # /history polling goes through the real proxy but is not stored as a request row (R1-14).
            assert paths >= {"/prompt", "/view"} and not any(p.startswith("/history") for p in paths)
            assert not activity._inflight and activity.last > 0
        finally:
            await gw.aclose()
            await stop_server(server, task)


def test_history_output_video_is_in_images():
    hist = {"outputs": {"1": {"images": [{"filename": "x.mp4", "subfolder": "video", "type": "output"}]} }}
    assert output_refs(hist, video=True)[0]["filename"] == "x.mp4"
    assert output_refs(hist) == []
