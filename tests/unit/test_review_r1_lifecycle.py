"""Regression tests for review round 1: R1-01, R1-03, R1-11, R1-12, R1-13 and R1-14 items 1-2."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.alerts import Alerts, NullNotifier
from colabinet.cli_parse import Listing, Runtime, Usage
from colabinet.colab_cli import CmdResult
from colabinet.comfy_files import validate_files
from colabinet.config import PACKAGE_DIR, REPO_ROOT, SEED_RATES, Settings
from colabinet.engines import ENGINES, RunSpec
from colabinet.engines.base import VM_DIR
from colabinet.presets import PresetError, _model_path, load_presets
from colabinet.proxy import ProxyManager
from colabinet.supervisor import QUEUE_BUSY_MAX_S, QUEUE_GRACE_S, CreateRefused, CreateRequest, Supervisor


def vm_module(name):
    sys.path.insert(0, str(VM_DIR))
    sys.path.insert(0, str(VM_DIR.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"r1_{name}", VM_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


setup_llamacpp = vm_module("setup_llamacpp")
setup_comfyui = vm_module("setup_comfyui")
setup_exllamav3 = vm_module("setup_exllamav3")
setup_strata = vm_module("setup_strata")

node = pytest.mark.skipif(shutil.which("node") is None, reason="Node is needed for frontend regressions")


# ------------------------------------------------------------------------------------------------------- R1-01
def comfy_guard(tmp_path, responder, *, start=1000.0, idle_min=30, ready_at=1000.0):
    """A ready ComfyUI session row, a fake clock and a guards-only Supervisor whose stop() is a mock."""
    clock = SimpleNamespace(t=start)
    clock.now = lambda: clock.t
    row = {"id": 1, "state": "ready", "hidden_port": 23001, "last_activity": start, "idle_min": idle_min,
           "created_at": start, "ready_at": ready_at, "cu_cap": None, "deadline": None, "colab_name": "comfy",
           "rate_cu_h": 8.9, "gpu_class": "G4", "stopped_at": None, "state_since": start}

    async def update_session(sid, **fields):
        row.update(fields)

    store = SimpleNamespace(live_sessions=AsyncMock(side_effect=lambda: [dict(row)]),
                            current_run=AsyncMock(return_value={"engine": "comfyui"}),
                            update_session=AsyncMock(side_effect=update_session), event=AsyncMock())
    client = httpx.AsyncClient(transport=httpx.MockTransport(responder))
    sup = Supervisor(settings=Settings(data_dir=tmp_path, daily_cap_cu=0), store=store,
                     cli=SimpleNamespace(enabled=True), tunnels=None, proxies=None,
                     alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=SimpleNamespace(presets={}, engines={}), http=client)
    sup._activity[1] = Activity(start)
    sup.spend_today, sup.cap_today, sup.stop = AsyncMock(return_value=0.0), AsyncMock(return_value=0.0), AsyncMock()
    return sup, row, clock, client


async def run_until_stop(sup, clock, limit_s, step=30.0):
    while not sup.stop.called and clock.t - 1000.0 < limit_s:
        clock.t += step
        await sup.guards_once()
    return clock.t - 1000.0


def tunnel_down(request):
    raise httpx.ConnectError("SSH tunnel down", request=request)


async def test_r1_01_broken_queue_probe_does_not_keep_session_alive_forever(tmp_path):
    """R1-01: a /queue probe that keeps failing (tunnel down) no longer resets the idle clock; the session idles out."""
    sup, row, clock, client = comfy_guard(tmp_path, tunnel_down)
    async with client:
        elapsed = await run_until_stop(sup, clock, 10 * 3600)
    assert sup.stop.called and sup.stop.call_args.args == (1, "idle")
    assert elapsed == pytest.approx(30 * 60, abs=30)  # idle_min, from the real last activity
    assert row["last_activity"] == 1000.0  # a failed probe is never persisted as activity


async def test_r1_01_failed_probe_defers_idle_stop_only_within_grace(tmp_path):
    """R1-01: an unknown queue defers the idle stop for QUEUE_GRACE_S after the last good /queue answer, no longer."""
    state = {"up": True}

    def upstream(request):
        if not state["up"]:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, json={"queue_running": [], "queue_pending": []})

    sup, row, clock, client = comfy_guard(tmp_path, upstream, idle_min=30)
    async with client:
        clock.t += 29 * 60  # idle for 29 min with a healthy, empty queue
        await sup.guards_once()
        ok_at = clock.t
        state["up"] = False  # the tunnel drops just before the idle limit
        elapsed = await run_until_stop(sup, clock, 10 * 3600)
    assert sup.stop.called
    assert clock.t - ok_at == pytest.approx(QUEUE_GRACE_S, abs=31)  # deferred by the grace window only
    assert row["last_activity"] == 1000.0
    assert elapsed < 30 * 60 + QUEUE_GRACE_S + 60


async def test_r1_01_grace_counts_from_ready_at_without_any_good_answer(tmp_path):
    """R1-01: with no successful /queue answer yet, the grace window starts at ready_at."""
    sup, row, clock, client = comfy_guard(tmp_path, tunnel_down, start=1000.0, idle_min=1, ready_at=1000.0)
    async with client:
        elapsed = await run_until_stop(sup, clock, 3600)
    assert sup.stop.called
    assert elapsed == pytest.approx(QUEUE_GRACE_S, abs=31)


async def test_r1_01_nonempty_queue_keeps_session_alive_at_most_the_busy_cap(tmp_path):
    """R1-01: a queue that never drains (stuck node) keeps the VM alive at most QUEUE_BUSY_MAX_S, then idles out."""
    sup, row, clock, client = comfy_guard(
        tmp_path, lambda request: httpx.Response(200, json={"queue_running": [[0, "p"]], "queue_pending": []}))
    async with client:
        elapsed = await run_until_stop(sup, clock, 24 * 3600, step=60.0)
    assert sup.stop.called and sup.stop.call_args.args == (1, "idle")
    assert QUEUE_BUSY_MAX_S <= elapsed <= QUEUE_BUSY_MAX_S + 30 * 60 + 60


async def test_r1_01_busy_cap_restarts_after_the_queue_drains(tmp_path):
    """R1-01: the busy cap is per streak: a queue that empties and fills again starts a new streak."""
    state = {"busy": True}

    def upstream(request):
        return httpx.Response(200, json={"queue_running": [[0, "p"]] if state["busy"] else []})

    sup, row, clock, client = comfy_guard(tmp_path, upstream)
    async with client:
        clock.t += QUEUE_BUSY_MAX_S - 60
        await sup.guards_once()
        state["busy"] = False
        clock.t += 60
        await sup.guards_once()
        state["busy"] = True
        clock.t += 60
        await sup.guards_once()
        assert sup._queue_busy_since[1] == clock.t
        assert not sup.stop.called


# ------------------------------------------------------------------------------------------------------- R1-03
class StopCLI:
    enabled = True

    def __init__(self, clock):
        self.clock = clock
        self.runtimes = [Runtime("ext-g4", "gpu-g4-ext-1", "G4", "Standard", "GPU")]
        self.stops = 0
        self.result = CmdResult(-9, "", timed_out=True)

    def command(self):
        return ["colab"]

    async def listing(self):
        return Listing(list(self.runtimes), Usage(500.0, SEED_RATES["G4"][0], len(self.runtimes)), True,
                       self.clock.now())

    async def exec_code(self, name, code, timeout=None):
        return CmdResult(0, "1\n")

    async def stop(self, name):
        self.stops += 1
        if self.result.ok:
            self.runtimes = []
        return self.result


@pytest.fixture
def ext(store, clock, tmp_path):
    cli = StopCLI(clock)
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=cli, tunnels=None,
                     proxies=ProxyManager(), alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=PresetError("unused"), http=object())
    return sup, cli


async def test_r1_03_manual_stop_failure_raises_and_keeps_keepalive(ext, store):
    """R1-03: a failed `colab stop` raises CreateRefused (shown by the panel) and keeps keepalive and the snapshot."""
    sup, cli = ext
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-ext-1", True, stop_after_min=10)
    with pytest.raises(CreateRefused, match="colab stop 失败：超时"):
        await sup.stop_external("ext-g4")
    cli.result = CmdResult(1, "HTTP 503 Service Unavailable")
    with pytest.raises(CreateRefused, match="colab stop 失败：HTTP 503"):
        await sup.stop_external("ext-g4")
    row = await store.external("gpu-g4-ext-1")
    assert row["keepalive"] == 1 and row["stop_after_min"] == 10
    assert "gpu-g4-ext-1" in sup._externals
    assert await store.db.fetchall("SELECT * FROM events WHERE kind = 'external_stopped'") == []


async def test_r1_03_keepalive_timer_alerts_and_retries_until_stopped(ext, store, clock):
    """R1-03: an expired keepalive timer whose stop fails raises external_keepalive and retries on the next tick."""
    sup, cli = ext
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-ext-1", True, stop_after_min=10)
    clock.advance(600)
    await sup.keepalive_externals_once()
    assert cli.stops == 1
    alerts = [a for a in sup.alerts.list() if a.key == "external_keepalive:gpu-g4-ext-1"]
    assert alerts and "自动停止失败" in alerts[0].message and "colab stop 失败" in alerts[0].message
    row = await store.external("gpu-g4-ext-1")
    assert row["keepalive"] == 1 and row["stop_after_min"] == 10
    await sup.reconcile_once()  # the VM is still listed: the timer survives
    clock.advance(15)
    await sup.keepalive_externals_once()
    assert cli.stops == 2
    cli.result = CmdResult(0, "[colab] Session terminated.")
    clock.advance(15)
    await sup.keepalive_externals_once()
    assert cli.stops == 3
    row = await store.external("gpu-g4-ext-1")
    assert row["keepalive"] == 0 and "gpu-g4-ext-1" not in sup._externals
    assert not [a for a in sup.alerts.list() if a.key == "external_keepalive:gpu-g4-ext-1"]


# ------------------------------------------------------------------------------------------------------- R1-11
def setup_sup(tmp_path, recorded, status):
    row = {"id": 1, "colab_name": "llm", "state": "setting_up", "state_since": 100.0, "gpu_class": "G4",
           "gpu_requested": "G4", "vm_port": 8090, "event_offset": 0, "log_offset": 0, "endpoint": "gpu-g4-1"}
    run = {"id": 7, "engine": "llamacpp", "status": "started", "alias": "ds", "download_s": 0, "download_gb": 0,
           "spec_json": json.dumps({"engine": "llamacpp", "repo": "o/r", "file": "m.gguf", "alias": "ds",
                                    "ctx": 8192, "kv": "q8_0", "extra_args": [], "prewarm": ["m.gguf"]})}
    store = SimpleNamespace(current_run=AsyncMock(return_value=run), session=AsyncMock(return_value=row),
                            update_session=AsyncMock(), update_run=AsyncMock(),
                            event=AsyncMock(side_effect=lambda ts, kind, **kw: recorded.append((kind, kw))))
    sup = Supervisor(settings=Settings(data_dir=tmp_path, setup_poll_s=0.001), store=store,
                     cli=SimpleNamespace(enabled=True), tunnels=None, proxies=None,
                     alerts=Alerts(NullNotifier(), lambda: 101.0),
                     clock=SimpleNamespace(now=lambda: 101.0), presets=SimpleNamespace(presets={}, engines={}),
                     http=object())
    sup._status = AsyncMock(return_value=status)
    return sup, row, store


async def test_r1_11_setting_up_records_the_whole_batch_after_server_ready(tmp_path):
    """R1-11: events after server_ready in the same batch (prewarm progress/done) are recorded, then tunnelling."""
    recorded = []
    sup, row, _ = setup_sup(tmp_path, recorded, {"events_offset": 300, "events": [
        {"event": "server_ready", "seconds": 5.0}, {"event": "prewarm", "progress": 0.0},
        {"event": "prewarm", "progress": 40.0},
        {"event": "prewarm_done", "gb": 28.8, "seconds": 90.0, "mb_s": 320.0, "files": 1}]})
    assert await sup._setting_up(row) == "tunnelling"
    kinds = [kind for kind, _ in recorded]
    assert kinds == ["vm:server_ready", "vm:prewarm", "vm:prewarm_done"]  # only the newest progress is kept
    assert recorded[1][1]["progress"] == 40.0


async def test_r1_11_status_reads_the_current_runs_events_file(tmp_path):
    """R1-11: the periodic status call reads run-N.jsonl of the current run, not the legacy setup.jsonl."""
    seen = []

    async def exec_file(name, path, env):
        seen.append(env)
        return SimpleNamespace(status={"events": [], "events_offset": 0, "log_offset": 0})

    row = {"id": 1, "colab_name": "llm", "vm_port": 8090, "event_offset": 12, "log_offset": 0,
           "endpoint": "gpu-g4-1"}
    store = SimpleNamespace(current_run=AsyncMock(return_value={"id": 7, "engine": "llamacpp", "alias": "ds"}),
                            update_session=AsyncMock())
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=SimpleNamespace(exec_file=exec_file),
                     tunnels=None, proxies=None, alerts=None, clock=SimpleNamespace(now=lambda: 5.0),
                     presets=SimpleNamespace(presets={}, engines={}), http=object())
    await sup._status(row)
    assert seen[0]["EVENTS"] == "/content/colabinet/run-7.jsonl" and seen[0]["EVENTS_OFFSET"] == "12"
    spec = RunSpec(engine="llamacpp", repo="o/r", file="m.gguf", alias="ds", ctx=8192, kv="f16")
    assert ENGINES["llamacpp"].setup_env(spec, 8090, 7)[
        "EVENTS"] == seen[0]["EVENTS"]  # the same file setup_llamacpp writes


async def test_r1_11_ready_consumes_prewarm_and_error_events(tmp_path, monkeypatch):
    """R1-11: while ready, vm:prewarm / vm:prewarm_done / error events are recorded and the offset advances."""
    recorded = []
    sup, row, store = setup_sup(tmp_path, recorded, {
        "events_offset": 900, "engine_healthy": True, "marker": {"session_id": 1},
        "events": [{"event": "prewarm", "progress": 10.0}, {"event": "prewarm", "progress": 55.5},
                   {"event": "prewarm_done", "gb": 0.0, "files": 0, "error": "disk gone"}]})
    row.update(state="ready", tunnel_pid=None)
    monkeypatch.setattr(sup, "_sample_due", lambda sid: True)
    monkeypatch.setattr(sup, "_sleep_or_wake", AsyncMock(return_value=False))
    monkeypatch.setattr(sup, "_healthy", AsyncMock(return_value=True))
    sup.tunnels = Mock()
    sup.tunnels.alive.return_value = True
    assert await sup._ready(row) == "ready"
    assert [kind for kind, _ in recorded] == ["vm:prewarm", "vm:prewarm_done"]
    assert recorded[0][1]["progress"] == 55.5 and recorded[1][1]["error"] == "disk gone"
    store.update_session.assert_any_await(1, event_offset=900)


def test_r1_11_activity_feed_labels_prewarm_in_chinese():
    """R1-11: vm:prewarm and vm:prewarm_done have Chinese labels in the overview activity feed."""
    from colabinet.web.app import make_templates

    events = [{"ts": 1.0, "kind": "vm:prewarm", "session_id": 1, "colab_name": "llm", "detail": {"progress": 40.0}},
              {"ts": 2.0, "kind": "vm:prewarm_done", "session_id": 1, "colab_name": "llm",
               "detail": {"gb": 28.8, "seconds": 90.0}}]
    html = make_templates().env.get_template("partials/overview_activity.html").render(
        overview_data={"activity": events})
    assert "页缓存预热中" in html and "40.0%" in html and "页缓存预热完成" in html
    assert "prewarm done" not in html and ">prewarm<" not in html


@pytest.mark.parametrize("module", [setup_comfyui, setup_exllamav3, setup_strata], ids=lambda m: m.__name__)
def test_r1_11_engine_switch_also_stops_llamacpp_prewarm(module, monkeypatch):
    """R1-11: stop_engines of the other engines also kills a leftover `setup_llamacpp.py --prewarm` reader."""
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    monkeypatch.setattr(module, "port_busy", lambda port: False)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    assert module.stop_engines("8090")
    process = "/usr/bin/python3 -u /content/colabinet/setup_llamacpp.py --prewarm /content/models/o%2Fr/m.gguf"
    matching = [cmd for cmd in commands if re.search(cmd[-1] if cmd[1] == "-KILL" else cmd[2], process)]
    assert len(matching) == 2  # TERM then KILL
    for cmd in commands:
        pattern = cmd[-1] if cmd[1] == "-KILL" else cmd[2]
        assert not re.search(pattern, " ".join(cmd))  # never matches its own pkill


# ------------------------------------------------------------------------------------------------------- R1-12
COMFY_PRESET = ('[[preset]]\nid = "c"\nengine = "comfyui"\nkind = "image"\nlabel = "c"\nrepo = "o/r"\n'
                'files = [{repo = "o/r", file = "a.safetensors", dir = "vae"}]\n')


@pytest.mark.parametrize("release", ['"--bad-flag"', '"has space"', '"a/b"', "42", '""'])
def test_r1_12_load_presets_rejects_bad_comfyui_release(tmp_path, release):
    """R1-12: an invalid [engines.comfyui] release is rejected when presets.toml is loaded, before any VM."""
    path = tmp_path / "presets.toml"
    path.write_text(f"[engines.comfyui]\nrelease = {release}\n\n" + COMFY_PRESET)
    with pytest.raises(PresetError, match="engines.comfyui"):
        load_presets(path)


def test_r1_12_valid_release_loads_and_bad_spec_release_is_refused(tmp_path):
    """R1-12: a commit release loads; a bad spec.release is refused by ComfyUIEngine.validate_spec / _validate_spec."""
    path = tmp_path / "presets.toml"
    path.write_text('[engines.comfyui]\nrelease = "afcb984e76fe9f870bf7746cf394766e1f690563"\n\n' + COMFY_PRESET)
    presets = load_presets(path)
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=None, cli=None, tunnels=None, proxies=None,
                     alerts=None, clock=None, presets=presets, http=object())
    spec = sup._run_spec(CreateRequest(name="img", gpu_class="G4", preset_id="c"))
    sup._validate_spec(spec, "G4")
    assert load_presets(REPO_ROOT / "presets.toml").engines["comfyui"]["release"]
    from dataclasses import replace
    bad = replace(spec, release="--upload-pack=x")
    with pytest.raises(ValueError, match="release"):
        ENGINES["comfyui"].validate_spec(bad)
    with pytest.raises(CreateRefused, match="release"):
        sup._validate_spec(bad, "G4")


def test_r1_12_comfy_stop_engines_waits_for_the_port(monkeypatch):
    """R1-12: setup_comfyui.stop_engines waits up to 30 s for the port and reports a port that stays busy."""
    sleeps = []
    monkeypatch.setattr(setup_comfyui.subprocess, "run", lambda cmd, **kw: None)
    monkeypatch.setattr(setup_comfyui.time, "sleep", sleeps.append)
    busy = iter([True, True, True, False])
    monkeypatch.setattr(setup_comfyui, "port_busy", lambda port: next(busy))
    assert setup_comfyui.stop_engines("8090") is True
    assert sleeps == [2, 1, 1, 1]
    sleeps.clear()
    monkeypatch.setattr(setup_comfyui, "port_busy", lambda port: True)
    assert setup_comfyui.stop_engines("8090") is False
    assert sleeps.count(1) == 30


def test_r1_12_comfy_main_fails_instead_of_probing_a_stale_listener(tmp_path, monkeypatch):
    """R1-12: when the port stays busy, main() errors before starting ComfyUI or probing /system_stats."""
    monkeypatch.setattr(setup_comfyui, "ROOT", tmp_path)
    monkeypatch.setattr(setup_comfyui, "stop_engines", lambda port: False)
    for name in ("install", "download"):
        monkeypatch.setattr(setup_comfyui, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.setattr(setup_comfyui.subprocess, "Popen", Mock(side_effect=AssertionError("Popen")))
    monkeypatch.setattr(setup_comfyui.urllib.request, "urlopen", Mock(side_effect=AssertionError("urlopen")))
    monkeypatch.setenv("COMFY_FILES", '[{"repo":"o/r","file":"a.safetensors","dir":"vae"}]')
    monkeypatch.setenv("COMFY_PORT", "8090")
    with pytest.raises(RuntimeError, match="port 8090 is still busy"):
        setup_comfyui.main()


# ------------------------------------------------------------------------------------------------------- R1-13
BAD_PATHS = ["--help", "-x.gguf", "sub/", "./a.gguf", "a//b.gguf", "a\x00b.gguf", "a\\b.gguf", "/abs.gguf", "..",
             "a/../b.gguf", "a/.", ""]


@pytest.mark.parametrize("value", BAD_PATHS)
def test_r1_13_model_paths_and_comfy_files_share_the_segment_rule(value):
    """R1-13: presets._model_path and comfy_files reject leading '-', '/', trailing '/', '//', '.', '..', NUL, '\\'."""
    assert not _model_path(value)
    with pytest.raises(ValueError, match="files.file"):
        validate_files([{"repo": "o/r", "file": value, "dir": "vae"}])
    assert not setup_llamacpp.safe_path(value)


def test_r1_13_good_paths_still_pass():
    """R1-13: ordinary nested repo paths are still accepted."""
    for value in ("a.gguf", "Q4/m-00001-of-00002.gguf", "split_files/vae/x.v1.safetensors"):
        assert _model_path(value) and setup_llamacpp.safe_path(value)
    validate_files([{"repo": "Comfy-Org/MiniMax-H3", "file": "vae/x.safetensors", "dir": "vae"}])


@pytest.mark.parametrize("repo", ["-flag/repo", "org/-repo", ".hidden/repo", "ünï/repo", "org/repo/x", "org"])
def test_r1_13_comfy_repo_must_start_alphanumeric(repo):
    """R1-13: files.repo starts with an ASCII letter or digit in both segments (never an `hf` option)."""
    with pytest.raises(ValueError, match="files.repo"):
        validate_files([{"repo": repo, "file": "x.safetensors", "dir": "vae"}])


def test_r1_13_preset_links_reject_option_like_names(tmp_path):
    """R1-13: a links entry such as "--help" is rejected by load_presets."""
    path = tmp_path / "presets.toml"
    path.write_text('[[preset]]\nid = "l"\nengine = "llamacpp"\nlabel = "l"\nrepo = "o/r"\nfile = "m.gguf"\n'
                    'links = { "--help" = "m.gguf" }\n')
    with pytest.raises(PresetError, match="相对路径"):
        load_presets(path)


@pytest.fixture
def escape(tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"SECRET")
    models = tmp_path / "models"
    repo_dir = Path(setup_llamacpp.model_directory("org/repo", str(models)))
    repo_dir.mkdir(parents=True)
    (repo_dir / "escape.gguf").symlink_to(outside)
    (repo_dir / "ok.gguf").write_bytes(b"ok")
    return models, repo_dir


def test_r1_13_prewarm_refuses_symlinks_out_of_the_repo(escape, monkeypatch):
    """R1-13: prewarm_paths and prewarm() refuse a repo file that resolves outside the repo directory."""
    models, repo_dir = escape
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(models))
    with pytest.raises(RuntimeError, match="outside"):
        setup_llamacpp.prewarm_paths("org/repo", ["escape.gguf"], models=str(models))
    with pytest.raises(RuntimeError, match="outside"):
        setup_llamacpp.prewarm([str(repo_dir / "escape.gguf")], workers=1)
    assert setup_llamacpp.prewarm(setup_llamacpp.prewarm_paths("org/repo", ["ok.gguf"], models=str(models)),
                                  workers=1) == 2
    with pytest.raises(ValueError, match="LLM_PREWARM"):
        setup_llamacpp.prewarm_paths("org/repo", ["../x.gguf"], models=str(models))


def test_r1_13_make_links_refuses_targets_out_of_the_repo(escape):
    """R1-13: a link target that resolves outside the repo directory is refused; bad link names too."""
    _, repo_dir = escape
    with pytest.raises(RuntimeError, match="outside"):
        setup_llamacpp.make_links(str(repo_dir), {"m-00002-of-00002.gguf": "escape.gguf"})
    with pytest.raises(ValueError, match="LLM_LINKS"):
        setup_llamacpp.make_links(str(repo_dir), {"../m.gguf": "ok.gguf"})
    assert setup_llamacpp.make_links(str(repo_dir), {"m-00002-of-00002.gguf": "ok.gguf"}) == 1


def test_r1_13_download_refuses_repo_symlinks_out_of_the_repo(tmp_path, monkeypatch):
    """R1-13: a downloaded file that is a symlink pointing outside the repo directory fails the download."""
    outside = tmp_path / "etc-passwd"
    outside.write_bytes(b"x")
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)
    monkeypatch.setattr(setup_llamacpp, "ensure_hf", lambda mode: False)

    def sh(cmd, env=None, **kw):
        dest = Path(setup_llamacpp.model_directory("org/repo", str(tmp_path / "models")))
        (dest / "m.gguf").symlink_to(outside)
        return 0, ""

    monkeypatch.setattr(setup_llamacpp, "sh", sh)
    with pytest.raises(RuntimeError, match="outside"):
        setup_llamacpp.download("org/repo", "m.gguf")
    with pytest.raises(ValueError, match="repo"):
        setup_llamacpp.download("-x/repo", "m.gguf")
    with pytest.raises(ValueError, match="model file"):
        setup_llamacpp.download("org/repo", "--help")


@pytest.mark.parametrize("key,value", [("LLM_LINKS", '{"--help": "m.gguf"}'), ("LLM_LINKS", '{"a.gguf": "../x"}'),
                                       ("LLM_LINKS", '["a"]'), ("LLM_PREWARM", '["../../etc/passwd"]'),
                                       ("LLM_PREWARM", '"m.gguf"')])
def test_r1_13_main_revalidates_links_and_prewarm_before_downloading(monkeypatch, key, value):
    """R1-13: setup_llamacpp.main re-validates LLM_LINKS / LLM_PREWARM on the VM before any download."""
    monkeypatch.setattr(setup_llamacpp, "event", lambda *a, **kw: None)
    monkeypatch.setattr(setup_llamacpp, "sh", lambda cmd, **kw: (0, ""))
    monkeypatch.setattr(setup_llamacpp, "stop_engines", lambda port: True)
    monkeypatch.setattr(setup_llamacpp, "stop_prewarm", lambda: None)
    monkeypatch.setattr(setup_llamacpp, "install", lambda release: ("/x/llama-server", []))
    monkeypatch.setattr(setup_llamacpp, "download", Mock(side_effect=AssertionError("downloaded")))
    for k in list(os.environ):
        if k.startswith("LLM_"):
            monkeypatch.delenv(k)
    for k, v in {"LLM_PORT": "8090", "LLM_RELEASE": "b1", "LLM_REPO": "org/repo", "LLM_FILE": "m.gguf",
                 key: value}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError, match=key):
        setup_llamacpp.main()


def test_r1_13_comfy_download_refuses_symlinks_out_of_the_repo(tmp_path, monkeypatch):
    """R1-13: setup_comfyui.download_file refuses a downloaded file that resolves outside its repo directory."""
    outside = tmp_path / "secret"
    outside.write_bytes(b"x")
    monkeypatch.setattr(setup_comfyui, "CACHE", tmp_path / "hf")
    monkeypatch.setattr(setup_comfyui, "COMFY", tmp_path / "comfy")
    monkeypatch.setattr(setup_comfyui, "event", Mock())

    def run(argv, **kw):
        source = tmp_path / "hf/org/repo/x.safetensors"
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_symlink():
            source.symlink_to(outside)
        return ""

    monkeypatch.setattr(setup_comfyui, "run", run)
    with pytest.raises(RuntimeError, match="outside"):
        setup_comfyui.download_file({"repo": "org/repo", "file": "x.safetensors", "dir": "vae"}, {}, attempts=1)
    assert not (tmp_path / "comfy/models/vae/x.safetensors").exists()


# ------------------------------------------------------------------------------------------------------- R1-14
async def test_r1_14_reconfigure_dialog_hides_llm_fields_for_comfyui(store, tmp_path):
    """R1-14(1): on a ComfyUI session the reconfigure dialog hides and disables ctx, KV cache and extra args."""
    from colabinet.config import Clock
    from colabinet.web import views
    from colabinet.web.app import make_templates

    presets = load_presets(REPO_ROOT / "presets.toml")
    sid = await store.insert_session(colab_name="img", gpu_requested="G4", gpu_class="G4", proxy_port=3080,
                                     hidden_port=23000, vm_port=8090, state="ready", state_since=1.0,
                                     created_at=1.0, rate_cu_h=8.9, rate_source="measured", idle_min=30)
    await store.insert_run(session_id=sid, engine="comfyui", alias="comfy-image-uncensored", ctx=32768, kv="f16",
                           parallel=1, started_at=1.0, spec_json='{"kind":"image"}')
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store,
                     cli=SimpleNamespace(enabled=False, version=None), tunnels=None, proxies=ProxyManager(),
                     alerts=Alerts(NullNotifier(), lambda: 1000.0), clock=Clock(), presets=presets, http=object())
    html = make_templates().env.get_template("pages/session.html").render(
        csrf="c", active="sessions", tab="overview", detail=await views.session_detail(sup, sid),
        presets=list(presets.presets.values()), base=await views.base_context(sup))
    for control in ("rc-ctx", "rc-kv", "rc-extra"):
        field = re.search(rf'<div class="field[^"]*"([^>]*)><label class="label" for="{control}">.*?</div>',
                          html, re.S)
        assert field and "hidden" in field.group(1), control
        tag = re.search(rf'<input [^>]*id="{control}"[^>]*>', html).group(0)
        assert "disabled" in tag, control
    assert "rc-kv-types-comfyui" not in html


def run_node(script, path, *args):
    result = subprocess.run([shutil.which("node"), "-e", script, str(path), *args], text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


RECONFIGURE_HARNESS = r"""
const fs = require('node:fs');
const elements = {};
const field = (id) => ({ hidden: false, id: `${id}-field` });
const control = (id, extra = {}) => {
  const f = field(id);
  return { id, value: 'stale', disabled: false, placeholder: '', dataset: {}, attrs: {}, field: f,
           closest: () => f, setAttribute(name, value) { this.attrs[name] = value; }, ...extra };
};
for (const id of ['rc-ctx', 'rc-kv', 'rc-extra', 'rc-projection']) elements[id] = control(id);
elements['rc-projection-field'] = { hidden: false };
global.window = global;
global.document = { body: { dataset: { csrf: 't' } }, addEventListener: () => {},
                    getElementById: (id) => elements[id] || null };
eval(fs.readFileSync(process.argv[1], 'utf8'));
const select = (engine, value) => ({ value, selectedOptions: [{ dataset: { engine, qwen36: 'false' } }] });
const snap = () => Object.fromEntries(['rc-ctx', 'rc-kv', 'rc-extra'].map((id) => [id, {
  hidden: elements[id].field.hidden, disabled: elements[id].disabled, value: elements[id].value,
  list: elements[id].attrs.list }]));
cbReconfigureChanged(select('comfyui', 'comfy-image-uncensored'));
const comfy = snap();
cbReconfigureChanged(select('llamacpp', 'qwen'));
process.stdout.write(JSON.stringify({ comfy, llm: snap() }));
"""


@node
def test_r1_14_cb_reconfigure_changed_toggles_llm_fields_for_comfyui():
    """R1-14(1): cbReconfigureChanged hides/disables/clears ctx, kv, extra_args for comfyui, restores them otherwise."""
    out = run_node(RECONFIGURE_HARNESS, PACKAGE_DIR / "static/app.js")
    for control in ("rc-ctx", "rc-kv", "rc-extra"):
        assert out["comfy"][control]["hidden"] and out["comfy"][control]["disabled"], control
        assert out["comfy"][control]["value"] == "", control
        assert not out["llm"][control]["hidden"] and not out["llm"][control]["disabled"], control
    assert out["comfy"]["rc-kv"]["list"] != "rc-kv-types-comfyui"
    assert out["llm"]["rc-kv"]["list"] == "rc-kv-types-llamacpp"


MEDIA_HARNESS = r"""
const fs = require('node:fs');
const scenario = process.argv[2];
const elements = {};
const el = (id) => elements[id] ||= { id, value: '', textContent: '', disabled: false, checked: false, files: [],
  children: [], replaceChildren() { this.children = []; }, append(node) { this.children.push(node); } };
for (const kind of ['image', 'video']) {
  el(`media-${kind}-session`).value = '1';
  el(`media-${kind}-prompt`).value = 'p';
}
el('media-seconds').value = '5';
const realTimeout = setTimeout;
global.setTimeout = (fn) => realTimeout(fn, 1);
global.window = global;
global.Image = class { constructor() { this.style = {}; } };
global.DOMException = DOMException;
global.document = { body: { dataset: { csrf: 't' } }, addEventListener: () => {}, getElementById: el,
  querySelectorAll: () => [], createElement: (tag) => ({ tag, style: {} }) };
const pending = [];
let polls = 0;
let imageDone = scenario !== 'cross';
const respond = (data) => ({ ok: true, json: async () => data });
global.fetch = (url, options = {}) => new Promise((resolve, reject) => {
  options.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
  if (url === '/ui/media/video') return resolve(respond({ id: 'video_1' }));
  if (url === '/ui/media/videos/video_1') {
    polls++;
    const job = imageDone ? { status: 'completed', progress: 100 } : { status: 'in_progress', progress: 50 };
    return resolve(respond(job));
  }
  if (url === '/ui/media/image') {
    if (scenario === 'cross') return resolve(respond({ data: [{ b64_json: 'eA==' }] }));
    pending.push({ resolve: () => resolve(respond({ data: [{ b64_json: 'eA==' }] })) });
    return;
  }
  reject(new Error(`unexpected ${url}`));
});
eval(fs.readFileSync(process.argv[1], 'utf8'));
const tick = () => new Promise((r) => realTimeout(r, 5));
(async () => {
  if (scenario === 'cross') {
    const video = cbMediaGenerate('video');
    while (polls < 1) await tick();  // the video job is being polled
    await cbMediaGenerate('image');
    const pollsAfterImage = polls;
    imageDone = true;
    await video;
    if (polls <= pollsAfterImage) throw new Error('video polling stopped');
    process.stdout.write(JSON.stringify({ video: el('media-video-result').children.map((c) => c.tag),
      videoStatus: el('media-video-status').textContent, image: el('media-image-result').children.length,
      buttons: [el('media-image-send').disabled, el('media-video-send').disabled] }));
    return;
  }
  const first = cbMediaGenerate('image');
  await tick();
  const second = cbMediaGenerate('image');
  await first;
  await tick();
  const whileSecond = el('media-image-send').disabled;
  pending[pending.length - 1].resolve();
  await second;
  process.stdout.write(JSON.stringify({ whileSecond, after: el('media-image-send').disabled,
    status: el('media-image-status').textContent, images: el('media-image-result').children.length }));
})();
"""


@node
def test_r1_14_image_request_does_not_abort_video_polling():
    """R1-14(2): starting an image request no longer aborts a running video job's status polling."""
    out = run_node(MEDIA_HARNESS, PACKAGE_DIR / "static/media.js", "cross")
    assert out["video"] == ["video"] and out["image"] == 1
    assert out["buttons"] == [False, False]


@node
def test_r1_14_superseded_request_does_not_reenable_the_button():
    """R1-14(2): a superseded request's finally leaves the button disabled while the newer request runs."""
    out = run_node(MEDIA_HARNESS, PACKAGE_DIR / "static/media.js", "same")
    assert out["whileSecond"] is True
    assert out["after"] is False and out["status"] == "已完成" and out["images"] == 1
