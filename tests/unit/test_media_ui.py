import asyncio
import base64

import httpx

from colabinet.api import create_app
from colabinet.config import REPO_ROOT, SEED_RATES
from colabinet.presets import load_presets
from colabinet.web import views
from colabinet.web.app import make_templates
from tests.fakes.browser import run_browser
from tests.unit.test_media_gateway import AUTH, PNG, media  # noqa: F401

BASE = {"balance": None, "burn_rate": 0, "live_count": 1, "alert_count": 0,
        "command_sessions": [], "daily": {"spend": 0, "cap": 10}}


def media_rows():
    return [{"id": 1, "colab_name": "image-session", "proxy_port": 3001, "gpu_class": "G4", "media_kind": "image",
             "run": {"alias": "images", "engine": "comfyui"}},
            {"id": 2, "colab_name": "video-session", "proxy_port": 3002, "gpu_class": "G4", "media_kind": "video",
             "run": {"alias": "videos", "engine": "comfyui"}}]


def render_media(selected=1):
    return make_templates().env.get_template("pages/playground.html").render(
        csrf="test-csrf", active="playground", base=BASE, selected=selected,
        sessions=[{"id": 3, "colab_name": "chat", "proxy_port": 3003, "gpu_class": "G4", "run": {"alias": "llm"}}],
        media_sessions=media_rows())


def test_media_playground_browser_image_video_tabs_and_csrf(tmp_path):
    result = run_browser(tmp_path, render_media(), r"""
return await evaluate(`(async () => {
  const requests = [];
  window.fetch = async (url, options={}) => {
    requests.push({url, method:options.method, csrf:options.headers?.['X-CSRF-Token'],
      body: options.body ? JSON.parse(options.body) : null});
    const data = url.endsWith('/image') ? {created:1,data:[{b64_json:'aW1hZ2U='}]} :
      url.endsWith('/video') ? {id:'video_test', status:'queued'} :
      {id:'video_test',status:'completed',progress:100,error:null};
    return {ok:true, json:async()=>data};
  };
  // Speed only the test's local polling timer, not unrelated UI timers.
  const timer = window.setTimeout;
  window.setTimeout = (fn, ms) => timer(fn, ms === 3000 ? 1 : ms);
  const initial = {image:!document.getElementById('pg-image-panel').hidden,
    chat:!document.getElementById('pg-chat-panel').hidden};
  document.getElementById('media-image-prompt').value = '一只猫';
  document.getElementById('media-workflow').value = 'anima';
  document.getElementById('media-upscale').checked = true;
  await cbMediaGenerate('image');
  const image = document.querySelector('#media-image-result img');
  cbMediaTab('video');
  const input = document.querySelector('[data-media-lora]');
  input.checked = true;
  input.closest('label').querySelector('input[type="number"]').value = '.6';
  document.getElementById('media-video-prompt').value = 'a stream with sound';
  await cbMediaGenerate('video');
  const video = document.querySelector('#media-video-result video');
  return {initial, requests, image:image?.getAttribute('src'),
    video:video?.getAttribute('src'), controls:video?.controls,
    imageHidden:document.getElementById('pg-image-panel').hidden,
    videoStatus:document.getElementById('media-video-status').textContent};
})()`);
""")
    assert result["initial"] == {"image": True, "chat": False}
    image, video = result["requests"][:2]
    assert image["csrf"] == video["csrf"] == "test-csrf"
    assert image["body"]["workflow"] == "anima" and image["body"]["upscale"] is True
    assert video["body"]["loras"][0]["strength"] == .6
    assert result["image"].startswith("data:image/png;base64,")
    assert result["video"] == "/ui/media/videos/video_test/content" and result["controls"]
    assert result["imageHidden"] and "已完成" in result["videoStatus"]


def test_new_media_preset_browser_hides_llm_settings(tmp_path):
    presets = load_presets(REPO_ROOT / "presets.toml")
    preset = presets.presets["comfy-image-uncensored"]
    fits = views.session_fit("comfyui", preset, ["CPU", "L4", "G4"], {})
    html = make_templates().env.get_template("pages/new.html").render(
        csrf="test-csrf", active="new", base=BASE, presets=list(presets.presets.values()), preset=preset.id,
        engine="comfyui", gpu="G4", gpus=["CPU", "L4", "G4"], gpu_fit=fits,
        rates={gpu: rate for gpu, (rate, _) in SEED_RATES.items()}, idle_default=30)
    result = run_browser(tmp_path, html, r"""
return await evaluate(`(() => ({
  submitted:[...new FormData(document.getElementById('new-form')).keys()],
  blocked:[...document.querySelectorAll('#new-form input[name="gpu"]:disabled')].map(n=>n.value),
  hidden:['ctx','kv','parallel','extra_args'].every(id=>document.getElementById(id).closest('.field').hidden),
  g4:document.querySelector('.cb-gpu input[value="G4"]').closest('label').textContent,
}))()`);
""")
    assert result["hidden"] and result["blocked"] == ["CPU", "L4"]
    assert not set(result["submitted"]) & {"ctx", "kv", "parallel", "extra_args"}
    assert "秒/张" in result["g4"] and "tok/s" not in result["g4"]


async def test_panel_media_routes_require_csrf_and_use_same_service(media):  # noqa: F811
    app = create_app(media.sup, "local-api-key", 7860, csrf="csrf-test")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as c:
        payload = {"session_id": 1, "prompt": "a cat", "workflow": "anima"}
        assert (await c.post("/ui/media/image", json=payload)).status_code == 403
        r = await c.post("/ui/media/image", headers={"X-CSRF-Token": "csrf-test"}, json=payload)
        assert r.status_code == 200 and base64.b64decode(r.json()["data"][0]["b64_json"]) == PNG
        r = await c.post("/ui/media/video", headers={"X-CSRF-Token": "csrf-test"},
                         json={"session_id": 2, "prompt": "a stream"})
        ident = r.json()["id"]
        await asyncio.sleep(.03)
        assert (await c.get(f"/ui/media/videos/{ident}")).json()["status"] == "completed"
        r = await c.get(f"/ui/media/videos/{ident}/content")
        assert r.status_code == 200 and r.headers["content-type"].startswith("video/mp4")
