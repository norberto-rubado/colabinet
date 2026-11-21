import dataclasses
import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import ValidationError

from colabinet.activity import Activity, is_inference
from colabinet.alerts import Alerts, NullNotifier
from colabinet.comfy_files import disk_gb, validate_files
from colabinet.comfy_graphs import ImageRequest, VideoRequest, frames, image_graph, upscale_graph, video_graph
from colabinet.config import GPU_CLASSES, REPO_ROOT, Settings
from colabinet.engines import ENGINES, RunSpec
from colabinet.engines.base import VM_DIR
from colabinet.presets import PresetError, load_presets
from colabinet.supervisor import CreateRefused, CreateRequest, Supervisor
from colabinet.web.views import default_gpu, session_fit


@pytest.fixture
def presets():
    return load_presets(REPO_ROOT / "presets.toml")


def test_media_presets_and_spec_roundtrip(presets):
    for name, kind, count, expected in [("comfy-image-uncensored", "image", 16, 132.6475),
                                         ("comfy-video-h3", "video", 11, 44.0666)]:
        preset = presets.presets[name]
        assert preset.engine == "comfyui" and preset.kind == kind and len(preset.files) == count
        assert disk_gb(preset.files) == pytest.approx(expected, abs=.001)
        assert preset.measured["G4"].seconds_per_output and preset.measured["G4"].gen_tps is None
        spec = RunSpec.from_preset(preset, "G4", presets.engines)
        assert RunSpec(**json.loads(json.dumps(dataclasses.asdict(spec)))) == spec
        assert spec.release == presets.engines["comfyui"]["release"]
        ENGINES["comfyui"].validate_spec(spec)
        assert not any("Noct" in f["repo"] for f in spec.files)


@pytest.mark.parametrize("change", [{"file": "../bad"}, {"file": "/bad"}, {"file": "x//y"},
                                    {"file": "x\\y"}, {"file": "--help"}, {"dir": "../bad"},
                                    {"repo": "../bad"}, {"size_gb": -1}, {"size_gb": float("nan")}])
def test_manifest_rejects_invalid_files(change):
    item = {"repo": "org/repo", "file": "model.safetensors", "dir": "loras", **change}
    with pytest.raises(ValueError):
        validate_files([item])


def test_manifest_duplicates_and_required():
    item = {"repo": "org/repo", "file": "model.safetensors", "dir": "loras"}
    for manifest in ([], [item, item], [{"file": "a"}], ["bad"]):
        with pytest.raises(ValueError):
            validate_files(manifest)


def test_parse_invalid_comfy_preset(tmp_path):
    path = tmp_path / "presets.toml"
    text = '[[preset]]\nid="x"\nengine="comfyui"\nlabel="x"\nrepo="org/repo"\nkind="image"\n'
    path.write_text(text)
    with pytest.raises(PresetError, match="files"):
        load_presets(path)
    path.write_text(text + 'files=[{repo="org/repo",file="x",dir="vae"}]\n')
    assert load_presets(path).presets["x"].files
    path.write_text(path.read_text().replace('kind="image"', 'kind="text"'))
    with pytest.raises(PresetError, match="kind"):
        load_presets(path)


def test_comfy_setup_env_uploads_and_validation(presets):
    spec = RunSpec.from_preset(presets.presets["comfy-video-h3"], "G4", presets.engines)
    engine = ENGINES["comfyui"]
    env = engine.setup_env(spec, 8188, 4)
    assert env["COMFY_PORT"] == "8188" and env["EVENTS"].endswith("run-4.jsonl")
    assert env["HF_TOKEN"] == "" and engine.health_path == "/system_stats"
    assert json.loads(env["COMFY_FILES"]) == list(spec.files)
    assert all(local.is_file() for local, remote in engine.uploads(spec))
    assert {local.name for local, _ in engine.uploads(spec)} == {"setup_comfyui.py", "comfy_files.py"}
    with pytest.raises(ValueError, match="LLM"):
        engine.setup_env(dataclasses.replace(spec, extra_args=("--listen", "0.0.0.0")), 8188, 4)


def assert_links(graph):
    for node in graph.values():
        assert node["class_type"]
        for value in node["inputs"].values():
            if isinstance(value, list):
                assert len(value) == 2 and value[0] in graph and isinstance(value[1], int)
    assert "__PROMPT__" not in json.dumps(graph) and "__SEED__" not in json.dumps(graph)


@pytest.mark.parametrize("workflow,steps", [("moss", 25), ("kroma", 10), ("krea2turbo", 8),
                                           ("h3", 6), ("h3edit", 6), ("anima", 10), ("wai", 28)])
def test_image_graphs(workflow, steps):
    body = ImageRequest(model="image", prompt="a landscape", workflow=workflow, size="1344x768")
    graph = image_graph(body, 123, "uploaded.png")
    assert_links(graph)
    assert any(n["class_type"] == "SaveImage" for n in graph.values())
    samplers = [n["inputs"] for n in graph.values() if n["class_type"] == "KSampler"]
    assert samplers[0]["seed"] == 123 and samplers[0]["steps"] == steps
    assert "a landscape" in json.dumps(graph)
    if workflow in ("moss", "wai"):
        other = image_graph(body.model_copy(update={"negative_prompt": "bad", "steps": 2}), 456, "uploaded.png")
        assert other != graph
        assert any(n["inputs"].get("text") == "bad" for n in other.values())
    else:
        # cfg=1.0 graphs: ComfyUI skips uncond, so a negative prompt would be silently ignored -> reject it.
        with pytest.raises(ValueError, match="负面提示词不会生效"):
            ImageRequest(model="image", prompt="a landscape", workflow=workflow, negative_prompt="bad")
    assert image_graph(body, 123, "uploaded.png") == graph  # no mutation across requests
    if workflow == "h3edit":
        assert graph["5"]["inputs"]["images.image_1"] == ["20", 0]
        assert graph["5"]["inputs"]["width"] == 0


def test_edit_requires_image():
    with pytest.raises(ValueError, match="image"):
        image_graph(ImageRequest(model="x", prompt="edit", workflow="h3edit"), 1)


def test_upscale_graph():
    graph = upscale_graph(44, "upload.png")
    assert_links(graph)
    assert graph["1"]["inputs"]["image"] == "upload.png"
    assert graph["57"]["inputs"]["resize_type.multiplier"] == 2
    assert graph["54"]["inputs"]["seed"] == 44


@pytest.mark.parametrize("seconds,want", [(4, 107), (5, 124), (6, 158), (15, 362), (4.01, 107)])
def test_frame_rounding(seconds, want):
    assert frames(seconds) == want and want >= seconds * 24 and want % 17 == 5


@pytest.mark.parametrize("image", [None, "first-frame.png"])
def test_video_graph_loras_audio_and_text_mode(image):
    body = VideoRequest(model="h3", prompt="a stream with sound", seconds=5,
                        loras=[{"name": "one.safetensors", "strength": .7},
                               {"name": "two.safetensors", "strength": -.3}])
    graph = video_graph(body, 77, image)
    assert_links(graph)
    assert graph["i2v"]["inputs"]["length"] == 124
    assert graph["sched"]["inputs"]["steps"] == 8
    assert graph["extra_lora_0"]["inputs"]["model"] == ["lora", 0]
    assert graph["extra_lora_1"]["inputs"]["model"] == ["extra_lora_0", 0]
    assert graph["sched"]["inputs"]["model"] == graph["guider"]["inputs"]["model"] == ["extra_lora_1", 0]
    assert graph["video"]["inputs"]["audio"] == ["adec", 0]
    assert graph["save"]["inputs"]["format"] == "mp4"
    assert graph["save"]["inputs"]["format.codec"] == "h264"
    assert ("first_frame" in graph["i2v"]["inputs"]) == bool(image)
    assert ("img" in graph) == bool(image)
    if image:  # first frame is center-cropped to the clip aspect, never stretched
        fit = graph["fit"]["inputs"]
        assert graph["i2v"]["inputs"]["first_frame"] == ["fit", 0] and fit["image"] == ["img", 0]
        assert (fit["width"], fit["height"], fit["crop"]) == (768, 1344, "center")


@pytest.mark.parametrize("schema,extra", [(ImageRequest, {"n": 0}), (ImageRequest, {"n": 9}),
    (ImageRequest, {"size": "1023x1024"}), (ImageRequest, {"workflow": "unknown"}),
    (ImageRequest, {"seed": -1}), (ImageRequest, {"steps": True}), (ImageRequest, {"upscale": "yes"}),
    (VideoRequest, {"seconds": 3}), (VideoRequest, {"seconds": 16}), (VideoRequest, {"seconds": float("nan")}),
    (VideoRequest, {"size": "4096x4096"}), (VideoRequest, {"loras": [{"name": "../bad"}]}),
    (VideoRequest, {"loras": [{"name": "x", "strength": float("inf")}]}), (VideoRequest, {"steps": 0})])
def test_request_validation(schema, extra):
    with pytest.raises(ValidationError):
        schema.model_validate({"model": "x", "prompt": "test", **extra})


def setup_module_vm():
    import sys
    sys.path.insert(0, str(VM_DIR.parent))
    spec = importlib.util.spec_from_file_location("setup_comfyui", VM_DIR / "setup_comfyui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def test_setup_loopback_torch_pins_and_retries(tmp_path, monkeypatch):
    vm = setup_module_vm()
    assert vm.build_command(8188)[-4:] == ["--listen", "127.0.0.1", "--port", "8188"]
    versions = '{"torch":"2.14.1+cu130","torchvision":"0.30","torchaudio":"2.14.1"}'
    monkeypatch.setattr(vm, "run", lambda args, **kw: versions)
    assert "torch==2.14.1+cu130" in vm.torch_constraints()
    monkeypatch.setattr(vm, "CACHE", tmp_path / "hf")
    monkeypatch.setattr(vm, "COMFY", tmp_path / "comfy")
    monkeypatch.setattr(vm, "event", Mock())
    sleeps = []
    monkeypatch.setattr(vm.time, "sleep", sleeps.append)
    calls = []

    def run(args, **kw):
        calls.append(args)
        assert kw["env"]["HF_TOKEN"] == "test"
        if len(calls) < 3:
            raise RuntimeError("HTTP 429")
        path = tmp_path / "hf/org/repo/weights/x.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
        return ""

    monkeypatch.setattr(vm, "run", run)
    item = {"repo": "org/repo", "file": "weights/x.safetensors", "dir": "vae"}
    assert vm.download_file(item, {"HF_TOKEN": "test"}) == 4
    assert len(calls) == 3 and sleeps[1] > sleeps[0]
    target = tmp_path / "comfy/models/vae/x.safetensors"
    assert target.is_symlink() and target.read_bytes() == b"test"


def test_setup_install_constrains_existing_torch(tmp_path, monkeypatch):
    vm = setup_module_vm()
    monkeypatch.setattr(vm, "ROOT", tmp_path)
    monkeypatch.setattr(vm, "COMFY", tmp_path / "ComfyUI")
    monkeypatch.setattr(vm, "event", Mock())
    monkeypatch.setattr(vm, "torch_constraints", lambda: "torch==2.14.1+cu130\n")
    calls = []
    monkeypatch.setattr(vm, "run", lambda argv, **kw: calls.append(argv) or "sha")
    vm.install("sha")
    pip = next(c for c in calls if "pip" in c)
    assert "-c" in pip and "-r" in pip and not {"--force-reinstall", "--upgrade"} & set(pip)
    assert (tmp_path / "torch-constraints.txt").read_text() == "torch==2.14.1+cu130\n"


def test_g4_fit_uses_disk_not_weight_vram(presets):
    preset = presets.presets["comfy-image-uncensored"]
    fits = session_fit("comfyui", preset, list(GPU_CLASSES), {})
    assert default_gpu(fits) == "G4"
    assert fits["G4"]["disk_gb"] > 130 and fits["G4"]["status"] == "ok"
    assert fits["G4"]["measured"]["unit"] == "秒/张"
    assert all(row["status"] == "blocked" for gpu, row in fits.items() if gpu != "G4")


def test_supervisor_validation_before_allocation(presets, tmp_path):
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=None, cli=None, tunnels=None, proxies=None,
                     alerts=None, clock=None, presets=presets, http=object())
    req = CreateRequest(name="image", gpu_class="G4", preset_id="comfy-image-uncensored")
    spec = sup._run_spec(req)
    sup._validate_spec(spec, "G4")
    with pytest.raises(CreateRefused, match="G4"):
        sup._validate_spec(spec, "L4")
    with pytest.raises(CreateRefused, match="files"):
        sup._run_spec(CreateRequest(name="x", gpu_class="G4", engine="comfyui"))


async def test_supervisor_health_is_system_stats(presets, tmp_path):
    paths = []
    def upstream(req):
        paths.append(req.url.path)
        return httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        sup = Supervisor(settings=Settings(data_dir=tmp_path), store=SimpleNamespace(current_run=AsyncMock(
                         return_value={"engine": "comfyui"})), cli=None, tunnels=None, proxies=None,
                         alerts=None, clock=None, presets=presets, http=client)
        assert await sup._healthy({"id": 1, "hidden_port": 1234})
    assert paths == ["/system_stats"]


@pytest.mark.parametrize("queue,stops", [({"queue_running": [[0, "p"]]}, False),
    ({"queue_pending": [[0, "p"]]}, False), ({"queue_running": [], "queue_pending": []}, True), (None, False)])
async def test_raw_queue_prevents_idle_stop_but_not_forever(presets, tmp_path, queue, stops):
    row = {"id": 1, "state": "ready", "hidden_port": 1234, "last_activity": 1, "idle_min": 1,
           "created_at": 1, "ready_at": 1, "cu_cap": None, "deadline": None, "colab_name": "comfy",
           "rate_cu_h": 1, "gpu_class": "G4", "stopped_at": None}
    store = SimpleNamespace(live_sessions=AsyncMock(return_value=[row]), current_run=AsyncMock(
        return_value={"engine": "comfyui"}), update_session=AsyncMock(), event=AsyncMock())
    def upstream(req):
        return httpx.Response(200, json=queue) if queue is not None else httpx.Response(502)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=SimpleNamespace(enabled=True),
            tunnels=None, proxies=None, alerts=Alerts(NullNotifier(), lambda: 500),
            clock=SimpleNamespace(now=lambda: 500), presets=presets, http=client)
        sup._activity[1] = Activity(1)
        sup.spend_today, sup.cap_today, sup.stop = AsyncMock(return_value=0), AsyncMock(return_value=100), AsyncMock()
        await sup.guards_once()
        assert sup.stop.called == stops


def test_vm_status_comfy_health_path(tmp_path, monkeypatch):
    import runpy
    import urllib.request

    seen = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_url(url, **kw):
        seen.append(url)
        return Response()

    monkeypatch.setenv("COLABINET_ROOT", str(tmp_path))
    monkeypatch.setenv("ENGINE_KIND", "comfyui")
    monkeypatch.setenv("VM_PORT", "8188")
    monkeypatch.setenv("WITH_TORCH", "0")
    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    runpy.run_path(str(VM_DIR / "status.py"))
    assert seen == ["http://127.0.0.1:8188/system_stats"]


def test_media_activity_paths():
    assert is_inference("POST", "/prompt") and is_inference("POST", "/v1/videos")
    assert is_inference("POST", "/v1/images/generations")
    assert not is_inference("GET", "/queue") and not is_inference("GET", "/history/p")
