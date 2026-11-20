"""Per-preset llama.cpp release, aliased shards (links), page-cache prewarm and the Hugging Face download mode."""

import dataclasses
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from colabinet.engines import ENGINES, RunSpec
from colabinet.engines.base import VM_DIR
from colabinet.presets import PresetError, load_presets
from colabinet.vm.gguf_files import gguf_files, repo_files

sys.path.insert(0, str(VM_DIR))
import setup_llamacpp  # noqa: E402

sys.path.pop(0)

REPO = "Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF"
STEM = "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS"
FIRST = f"{STEM}-00001-of-00006.gguf"
SIXTH = f"{STEM}-00006-of-00006.gguf"
ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
LINKS = {SIXTH: ENGRAM}

ENGINES_TOML = '[engines.llamacpp]\nrelease = "b12359"\n\n'
PRESET = f"""[[preset]]
id = "dsv41-flash"
engine = "llamacpp"
label = "DeepSeek V4.1 Flash"
repo = "{REPO}"
file = "{FIRST}"
"""


def write(tmp_path, text):
    f = tmp_path / "presets.toml"
    f.write_text(text)
    return f


def test_new_preset_fields_parse(tmp_path):
    text = ENGINES_TOML + PRESET + f"""release = "b12999"
links = {{ "{SIXTH}" = "{ENGRAM}" }}
prewarm = ["{SIXTH}", "{FIRST}"]
"""
    p = load_presets(write(tmp_path, text))
    preset = p.presets["dsv41-flash"]
    assert preset.release == "b12999"
    assert preset.links == LINKS and isinstance(preset.links, dict)
    assert preset.prewarm == (SIXTH, FIRST)


def test_new_preset_fields_default_empty(tmp_path):
    preset = load_presets(write(tmp_path, ENGINES_TOML + PRESET)).presets["dsv41-flash"]
    assert (preset.release, preset.links, preset.prewarm) == (None, {}, ())


@pytest.mark.parametrize("extra,msg", [
    ('release = 12359\n', "release 必须是 llama.cpp 发布标签"),
    ('release = "latest"\n', "release 必须是 llama.cpp 发布标签"),
    ('release = "b12359x"\n', "release 必须是 llama.cpp 发布标签"),
    ('links = ["a.gguf"]\n', "links 必须是"),
    ('links = { "a.gguf" = 3 }\n', "links 必须是"),
    ('links = { "a.gguf" = { b = "c" } }\n', "links 必须是"),
    ('links = { "../a.gguf" = "b.gguf" }\n', "必须是模型目录内的相对路径"),
    ('links = { "a.gguf" = "/etc/passwd" }\n', "必须是模型目录内的相对路径"),
    ('links = { "a.gguf" = "a.gguf" }\n', "不能指向自身"),
    ('links = { "a.gguf" = "b.gguf", "b.gguf" = "c.gguf" }\n', "不支持链式链接"),
    ('prewarm = "a.gguf"\n', "prewarm 必须是字符串列表"),
    ('prewarm = ["a.gguf", 2]\n', "prewarm 必须是字符串列表"),
    ('prewarm = ["../../x"]\n', "必须是模型目录内的相对路径"),
])
def test_new_preset_field_errors(tmp_path, extra, msg):
    with pytest.raises(PresetError, match=msg) as info:
        load_presets(write(tmp_path, ENGINES_TOML + PRESET + extra))
    assert str(info.value).startswith("第 1 个预设：")


@pytest.mark.parametrize("key,value", [("release", '"b1"'), ("links", '{ a = "b" }'), ("prewarm", '["a"]')])
def test_new_preset_fields_are_llamacpp_only(tmp_path, key, value):
    text = ('[[preset]]\nid = "x"\nengine = "exllamav3"\nlabel = "x"\nrepo = "o/r"\nrevision = "4.00bpw"\n'
            f"{key} = {value}\n")
    with pytest.raises(PresetError, match=f"只有 llamacpp 预设支持 {key}"):
        load_presets(write(tmp_path, text))


def test_from_preset_prefers_preset_release(tmp_path):
    text = ENGINES_TOML + PRESET + f'release = "b12999"\nlinks = {{ "{SIXTH}" = "{ENGRAM}" }}\nprewarm = ["{SIXTH}"]\n'
    p = load_presets(write(tmp_path, text))
    spec = RunSpec.from_preset(p.presets["dsv41-flash"], "G4", p.engines)
    assert spec.release == "b12999"
    assert spec.links == LINKS and spec.prewarm == (SIXTH,)
    plain = load_presets(write(tmp_path, ENGINES_TOML + PRESET))
    spec2 = RunSpec.from_preset(plain.presets["dsv41-flash"], "G4", plain.engines)
    assert spec2.release == "b12359" and spec2.links == {} and spec2.prewarm == ()


def test_runspec_round_trips_through_spec_json_and_loads_old_rows():
    spec = RunSpec(engine="llamacpp", repo=REPO, file=FIRST, alias="ds", ctx=8192, kv="q8_0",
                   extra_args=("-fa", "on"), release="b12999", links=LINKS, prewarm=(SIXTH, FIRST))
    stored = json.loads(json.dumps(dataclasses.asdict(spec)))
    assert stored["links"] == LINKS and stored["prewarm"] == [SIXTH, FIRST]
    # Same rebuild as supervisor.py: only extra_args is converted there; prewarm comes back as a tuple anyway.
    rebuilt = RunSpec(**{**stored, "extra_args": tuple(stored["extra_args"])})
    assert rebuilt == spec and isinstance(rebuilt.prewarm, tuple)
    old = {k: v for k, v in stored.items() if k not in ("links", "prewarm")}
    legacy = RunSpec(**{**old, "extra_args": tuple(old["extra_args"])})
    assert legacy.links == {} and legacy.prewarm == ()
    assert replace(legacy, ctx=4096).links == {}


def test_setup_env_exports_links_prewarm_and_hf_mode(monkeypatch):
    monkeypatch.delenv("COLABINET_LLM_HF_MODE", raising=False)
    spec = RunSpec(engine="llamacpp", repo=REPO, file=FIRST, alias="ds", ctx=8192, kv="q8_0", release="b12999",
                   links=LINKS, prewarm=(SIXTH,))
    env = ENGINES["llamacpp"].setup_env(spec, 8090, run_id=7)
    assert env["LLM_RELEASE"] == "b12999"
    assert json.loads(env["LLM_LINKS"]) == LINKS
    assert json.loads(env["LLM_PREWARM"]) == [SIXTH]
    assert env["LLM_HF_MODE"] == "xet-fast"
    bare = ENGINES["llamacpp"].setup_env(replace(spec, links={}, prewarm=()), 8090, run_id=7)
    assert (bare["LLM_LINKS"], bare["LLM_PREWARM"]) == ("{}", "[]")
    monkeypatch.setenv("COLABINET_LLM_HF_MODE", "no-xet")
    assert ENGINES["llamacpp"].setup_env(spec, 8090, run_id=7)["LLM_HF_MODE"] == "no-xet"
    monkeypatch.setenv("COLABINET_LLM_HF_MODE", "turbo")
    assert ENGINES["llamacpp"].setup_env(spec, 8090, run_id=7)["LLM_HF_MODE"] == "xet-fast"


def test_repo_files_replaces_link_names_with_targets():
    assert gguf_files(FIRST)[-1] == SIXTH
    files = repo_files(FIRST, LINKS)
    assert files == [f"{STEM}-{i:05d}-of-00006.gguf" for i in range(1, 6)] + [ENGRAM]
    assert repo_files("model.gguf") == ["model.gguf"]
    assert repo_files("model.gguf", {"alias.gguf": "model.gguf"}) == ["model.gguf"]  # no duplicates


@pytest.fixture
def vm(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(setup_llamacpp, "MODELS", str(tmp_path))
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **kw: events.append((kind, kw)))
    monkeypatch.setattr(setup_llamacpp, "ensure_hf", lambda mode: False)
    return tmp_path / REPO.replace("/", "%2F"), events


def fake_hf(calls, content=b"gguf"):
    def sh(cmd, env=None, **kw):
        args = shlex.split(cmd)
        dest = Path(args[args.index("--local-dir") + 1])
        paths = args[3:args.index("--local-dir")]
        calls.append({"args": args, "paths": paths, "env": env})
        for path in paths:
            (dest / path).parent.mkdir(parents=True, exist_ok=True)
            (dest / path).write_bytes(content)
        return 0, ""
    return sh


def test_download_fetches_link_targets_not_link_names_and_creates_links(vm, monkeypatch):
    models, events = vm
    calls = []
    monkeypatch.setattr(setup_llamacpp, "sh", fake_hf(calls))
    setup_llamacpp.download(REPO, FIRST, links=LINKS)
    (call,) = calls
    assert call["args"][:3] == ["hf", "download", REPO]
    assert call["paths"] == [f"{STEM}-{i:05d}-of-00006.gguf" for i in range(1, 6)] + [ENGRAM]
    assert SIXTH not in call["paths"]
    assert call["args"][-2:] == ["--max-workers", "8"]
    link = models / SIXTH
    assert link.is_symlink() and os.readlink(link) == ENGRAM  # relative
    assert link.read_bytes() == b"gguf"
    done = [kw for kind, kw in events if kind == "download_done"][-1]
    assert done["links"] == 1 and done["links_made"] == 1 and done["hf_mode"] == "xet-fast"
    # Second run: everything present, nothing fetched, the link is kept as is.
    setup_llamacpp.download(REPO, FIRST, links=LINKS)
    assert len(calls) == 1
    done = [kw for kind, kw in events if kind == "download_done"][-1]
    assert done["skipped"] and done["links_made"] == 0


def test_download_only_fetches_missing_targets(vm, monkeypatch):
    models, _ = vm
    models.mkdir(parents=True)
    for i in range(1, 6):
        (models / f"{STEM}-{i:05d}-of-00006.gguf").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(setup_llamacpp, "sh", fake_hf(calls))
    setup_llamacpp.download(REPO, FIRST, links=LINKS, mode="no-xet", workers=3)
    assert calls[0]["paths"] == [ENGRAM]
    assert calls[0]["args"][-2:] == ["--max-workers", "3"]
    assert calls[0]["env"]["HF_HUB_DISABLE_XET"] == "1"


def test_download_without_links_requires_every_shard(vm, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_llamacpp, "sh", fake_hf(calls))
    setup_llamacpp.download(REPO, FIRST)
    assert calls[0]["paths"][-1] == SIXTH and ENGRAM not in calls[0]["paths"]


def test_download_fails_when_link_target_missing(vm, monkeypatch):
    def sh(cmd, env=None, **kw):
        return 0, ""  # hf "succeeds" but writes nothing
    monkeypatch.setattr(setup_llamacpp, "sh", sh)
    with pytest.raises(RuntimeError, match=ENGRAM):
        setup_llamacpp.download(REPO, FIRST, links=LINKS)


def test_make_links_is_idempotent_and_replaces_wrong_entries(tmp_path):
    (tmp_path / "real.gguf").write_bytes(b"real")
    (tmp_path / "other.gguf").write_bytes(b"other")
    links = {"model-00002-of-00002.gguf": "real.gguf"}
    link = tmp_path / "model-00002-of-00002.gguf"
    assert setup_llamacpp.make_links(str(tmp_path), links) == 1
    assert os.readlink(link) == "real.gguf"
    inode = os.lstat(link).st_ino
    assert setup_llamacpp.make_links(str(tmp_path), links) == 0
    assert os.lstat(link).st_ino == inode  # untouched
    link.unlink()
    link.symlink_to("other.gguf")  # wrong link
    assert setup_llamacpp.make_links(str(tmp_path), links) == 1 and os.readlink(link) == "real.gguf"
    link.unlink()
    link.symlink_to(tmp_path / "real.gguf")  # absolute link: replaced by the relative form
    assert setup_llamacpp.make_links(str(tmp_path), links) == 1 and os.readlink(link) == "real.gguf"
    link.unlink()
    link.write_bytes(b"stale partial")  # a regular file in the way
    assert setup_llamacpp.make_links(str(tmp_path), links) == 1
    assert link.is_symlink() and link.read_bytes() == b"real"
    assert not list(tmp_path.glob("*.colabinet-link"))


def test_make_links_in_subdirectory_is_relative(tmp_path):
    (tmp_path / "Q4").mkdir()
    (tmp_path / "engram.gguf").write_bytes(b"e")
    setup_llamacpp.make_links(str(tmp_path), {"Q4/m-00002-of-00002.gguf": "engram.gguf"})
    link = tmp_path / "Q4" / "m-00002-of-00002.gguf"
    assert os.readlink(link) == "../engram.gguf" and link.read_bytes() == b"e"


def test_make_links_needs_its_target(tmp_path):
    with pytest.raises(RuntimeError, match="link target missing.gguf"):
        setup_llamacpp.make_links(str(tmp_path), {"a.gguf": "missing.gguf"})


@pytest.mark.parametrize("mode,xet_disabled,high_perf", [
    ("xet-fast", None, "1"), ("xet", None, None), ("no-xet", "1", None)])
def test_hf_env_modes(mode, xet_disabled, high_perf):
    base = {"PATH": "/usr/bin", "HF_HUB_DISABLE_XET": "1", "HF_XET_HIGH_PERFORMANCE": "1"}
    env = setup_llamacpp.hf_env(mode, base)
    assert env.get("HF_HUB_DISABLE_XET") == xet_disabled
    assert env.get("HF_XET_HIGH_PERFORMANCE") == high_perf
    assert env["HF_HUB_DISABLE_PROGRESS_BARS"] == "1" and env["PATH"] == "/usr/bin"
    assert base["HF_HUB_DISABLE_XET"] == "1"  # not mutated


def test_hf_env_rejects_unknown_mode():
    with pytest.raises(ValueError, match="LLM_HF_MODE"):
        setup_llamacpp.hf_env("fast", {})


def test_download_uses_xet_fast_by_default(vm, monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "1")
    calls = []
    monkeypatch.setattr(setup_llamacpp, "sh", fake_hf(calls))
    setup_llamacpp.download("org/repo", "model.gguf")
    assert "HF_HUB_DISABLE_XET" not in calls[0]["env"] and calls[0]["env"]["HF_XET_HIGH_PERFORMANCE"] == "1"


def ensure_hf_with(monkeypatch, *, has_hf, has_xet, pip_rc=0, hf_after=True):
    events, commands = [], []
    state = {"hf": has_hf}
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **kw: events.append((kind, kw)))
    monkeypatch.setattr(setup_llamacpp.shutil, "which", lambda name: "/usr/local/bin/hf" if state["hf"] else None)

    def sh(cmd, **kw):
        commands.append(cmd)
        if "import hf_xet" in cmd:
            return (0 if has_xet else 1), ""
        assert "pip install -q -U" in cmd and "'huggingface_hub[hf_xet]'" in cmd
        state["hf"] = hf_after
        return pip_rc, "pip output"

    monkeypatch.setattr(setup_llamacpp, "sh", sh)
    return events, commands


def test_ensure_hf_installs_hf_xet_when_missing(monkeypatch):
    events, commands = ensure_hf_with(monkeypatch, has_hf=True, has_xet=False)
    assert setup_llamacpp.ensure_hf("xet-fast") is True
    assert [kind for kind, _ in events] == ["hf_install", "hf_install_done"]
    assert events[0][1]["reason"] == "hf_xet missing" and events[1][1]["ok"] is True
    assert any("pip install" in c for c in commands)


def test_ensure_hf_skips_when_present(monkeypatch):
    events, commands = ensure_hf_with(monkeypatch, has_hf=True, has_xet=True)
    assert setup_llamacpp.ensure_hf("xet-fast") is False
    assert events == [] and not any("pip install" in c for c in commands)


def test_ensure_hf_no_xet_does_not_need_hf_xet(monkeypatch):
    events, commands = ensure_hf_with(monkeypatch, has_hf=True, has_xet=False)
    assert setup_llamacpp.ensure_hf("no-xet") is False
    assert commands == []


def test_ensure_hf_failure_is_fatal_only_without_hf(monkeypatch):
    events, _ = ensure_hf_with(monkeypatch, has_hf=True, has_xet=False, pip_rc=1)
    assert setup_llamacpp.ensure_hf("xet") is True  # hf still works over plain HTTP
    assert events[-1][1]["ok"] is False
    ensure_hf_with(monkeypatch, has_hf=False, has_xet=False, pip_rc=1, hf_after=False)
    with pytest.raises(RuntimeError, match="could not install"):
        setup_llamacpp.ensure_hf("xet-fast")


def test_start_prewarm_detaches_with_resolved_paths(tmp_path, monkeypatch):
    models = tmp_path / REPO.replace("/", "%2F")
    models.mkdir()
    (models / ENGRAM).write_bytes(b"e")
    (models / SIXTH).symlink_to(ENGRAM)
    launched = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            launched.append((argv, kwargs))
            self.pid = 4321

    monkeypatch.setenv("OUT", str(tmp_path / "setup.out"))
    monkeypatch.setattr(setup_llamacpp.subprocess, "Popen", FakePopen)
    proc = setup_llamacpp.start_prewarm(REPO, [SIXTH, ENGRAM, FIRST], models=str(tmp_path))
    assert proc.pid == 4321
    ((argv, kwargs),) = launched
    assert argv[:4] == [sys.executable, "-u", str(Path(setup_llamacpp.__file__).resolve()), "--prewarm"]
    # The link and its target are one file: read once.
    assert argv[4:] == [str(models / SIXTH), str(models / FIRST)]
    assert kwargs["start_new_session"] is True and kwargs["stdin"] is subprocess.DEVNULL
    assert setup_llamacpp.start_prewarm(REPO, [], models=str(tmp_path)) is None


def test_prewarm_pattern_never_matches_its_own_pkill():
    import re
    pattern = setup_llamacpp.PREWARM_PATTERN
    assert re.search(pattern, "/usr/bin/python3 -u /content/colabinet/setup_llamacpp.py --prewarm /x")
    assert not re.search(pattern, f"pkill -f {pattern}")


def test_prewarm_reads_every_byte_in_parallel_ranges(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **kw: events.append((kind, kw)))
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    a.write_bytes(os.urandom(1000))
    b.write_bytes(os.urandom(4097))
    (tmp_path / "alias.gguf").symlink_to("b.gguf")
    reads = []
    real_pread = os.pread

    def pread(fd, n, offset):
        reads.append((fd, n, offset))
        return real_pread(fd, n, offset)

    monkeypatch.setattr(setup_llamacpp.os, "pread", pread)
    monkeypatch.setattr(setup_llamacpp, "PREWARM_READ", 256)
    got = setup_llamacpp.prewarm([str(a), str(tmp_path / "alias.gguf"), str(b), str(tmp_path / "gone.gguf")],
                                 workers=4, chunk=512, interval=0.01)
    assert got == 1000 + 4097
    assert sum(n for _, n, _ in reads) >= got and max(n for _, n, _ in reads) <= 256
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "prewarm" and kinds[-1] == "prewarm_done" and kinds.count("prewarm_done") == 1
    assert events[0][1]["progress"] == 0.0 and events[0][1]["files"] == 2
    done = events[-1][1]
    assert done["files"] == 2 and done["missing"] == [str(tmp_path / "gone.gguf")]
    assert {"gb", "mb_s", "seconds"} <= done.keys() and "error" not in done


def test_prewarm_entry_point_logs_events(tmp_path):
    """Run the real `--prewarm` entry point the way start_prewarm launches it."""
    data = tmp_path / "m.gguf"
    data.write_bytes(b"\0" * (3 << 20))
    events = tmp_path / "ev.jsonl"
    env = {**os.environ, "EVENTS": str(events), "PYTHONPATH": ""}
    out = subprocess.run([sys.executable, "-u", str(VM_DIR / "setup_llamacpp.py"), "--prewarm", str(data)],
                         env=env, cwd=str(VM_DIR), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    kinds = [json.loads(line)["event"] for line in events.read_text().splitlines()]
    assert kinds[0] == "prewarm" and kinds[-1] == "prewarm_done"
    assert json.loads(events.read_text().splitlines()[-1])["gb"] == round((3 << 20) / 1e9, 2)


def run_main(tmp_path, monkeypatch, start_prewarm):
    order = []
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **kw: order.append((kind, kw)))
    monkeypatch.setattr(setup_llamacpp, "sh", lambda cmd, **kw: (0, "1000"))
    monkeypatch.setattr(setup_llamacpp, "stop_engines", lambda port: True)
    monkeypatch.setattr(setup_llamacpp, "stop_prewarm", lambda: order.append(("stop_prewarm", {})))
    monkeypatch.setattr(setup_llamacpp, "install", lambda release: ("/x/llama-server", []))
    downloads = []
    monkeypatch.setattr(setup_llamacpp, "download", lambda repo, name, **kw: downloads.append((name, kw)))
    monkeypatch.setattr(setup_llamacpp, "start_prewarm", lambda repo, names: start_prewarm(order, names))

    class Proc:
        returncode = None

        def poll(self):
            return None

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(setup_llamacpp.subprocess, "Popen", lambda *a, **kw: Proc())
    monkeypatch.setattr(setup_llamacpp.urllib.request, "urlopen", lambda *a, **kw: Resp())
    monkeypatch.setattr(setup_llamacpp, "ENGINE_LOG", str(tmp_path / "engine.log"))
    for key in list(os.environ):
        if key.startswith("LLM_"):
            monkeypatch.delenv(key)
    env = {"LLM_PORT": "8090", "LLM_RELEASE": "b12999", "LLM_REPO": REPO, "LLM_FILE": FIRST,
           "LLM_LINKS": json.dumps(LINKS), "LLM_PREWARM": json.dumps([SIXTH]), "LLM_HF_MODE": "xet"}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    setup_llamacpp.main()
    return [kind for kind, _ in order], order, downloads


def test_main_starts_prewarm_after_server_ready_without_waiting(tmp_path, monkeypatch):
    kinds, _, downloads = run_main(tmp_path, monkeypatch,
                                   lambda order, names: order.append(("start_prewarm", {"names": names})))
    assert downloads == [(FIRST, {"links": LINKS, "mode": "xet", "workers": 8})]
    assert kinds.index("stop_prewarm") < kinds.index("server_ready") < kinds.index("start_prewarm")
    assert kinds[-1] == "start_prewarm"


def test_prewarm_launch_failure_does_not_fail_setup(tmp_path, monkeypatch):
    def boom(order, names):
        raise OSError("no fork")

    kinds, order, _ = run_main(tmp_path, monkeypatch, boom)
    assert kinds[-2:] == ["server_ready", "prewarm_done"] and order[-1][1]["error"] == "no fork"


def test_main_rejects_unknown_hf_mode_before_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_llamacpp, "event", lambda kind, **kw: None)
    monkeypatch.setattr(setup_llamacpp, "sh", lambda cmd, **kw: (0, ""))
    monkeypatch.setattr(setup_llamacpp, "stop_engines", lambda port: True)
    monkeypatch.setattr(setup_llamacpp, "stop_prewarm", lambda: None)
    monkeypatch.setattr(setup_llamacpp, "install", lambda release: ("/x/llama-server", []))
    monkeypatch.setattr(setup_llamacpp, "download", lambda *a, **kw: pytest.fail("downloaded"))
    for key, value in {"LLM_PORT": "8090", "LLM_RELEASE": "b1", "LLM_REPO": REPO, "LLM_FILE": FIRST,
                       "LLM_HF_MODE": "turbo"}.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="LLM_HF_MODE"):
        setup_llamacpp.main()
