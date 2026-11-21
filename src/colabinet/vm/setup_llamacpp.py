# Runs ON the Colab VM, started detached by bg.py.
# Idempotent: skips the install and downloads that are already present, stops all three engines first (so a
# reconfigure can switch engines), then starts llama-server and waits for /health. Progress goes to $EVENTS.
# `setup_llamacpp.py --prewarm FILE...` is the detached page-cache reader started once the server is healthy.
import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

from gguf_files import gguf_files, repo_files

E = os.environ.get
ROOT = E("COLABINET_ROOT", "/content/colabinet")
EVENTS = E("EVENTS", os.path.join(ROOT, "setup.jsonl"))
ENGINE_LOG = E("ENGINE_LOG", os.path.join(ROOT, "engine.log"))
MODELS = E("MODELS_DIR", "/content/models")
LLAMA_DIR = E("LLAMA_DIR", "/content/llama")
T0 = time.time()
HF_MODES = ("xet-fast", "xet", "no-xet")
HF_PACKAGE = "huggingface_hub[hf_xet]"
PREWARM_WORKERS = 16
PREWARM_CHUNK = 64 << 20  # range handed to one reader
PREWARM_READ = 16 << 20  # bytes per pread call: bounds the transient buffers to workers x 16 MiB
PREWARM_PATTERN = "[s]etup_llamacpp.py --prewarm"  # bracket: the stop command must not match its own argv


def event(kind, **kw):
    rec = {"t": round(time.time() - T0, 1), "event": kind, **kw}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def sh(cmd, env=None, timeout=None):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def model_directory(repo, models):
    """Repository-specific cache: different publishers reuse projector filenames."""
    return os.path.join(models, quote(repo, safe=""))


def build_command(env, server, models=MODELS):
    """llama-server argv from the LLM_* environment."""
    models = model_directory(env["LLM_REPO"], models)
    cmd = [server, "-m", os.path.join(models, env["LLM_FILE"]), "--host", "127.0.0.1", "--port", env["LLM_PORT"],
           "-c", env.get("LLM_CTX", "32768"), "-np", env.get("LLM_PARALLEL", "1"),
           "-ctk", env.get("LLM_KV", "f16"), "-ctv", env.get("LLM_KV", "f16"), "--jinja"]
    if env.get("LLM_ALIAS"):
        cmd += ["--alias", env["LLM_ALIAS"]]
    if env.get("LLM_MMPROJ_FILE"):
        cmd += ["--mmproj", os.path.join(models, env["LLM_MMPROJ_FILE"])]
        if env.get("LLM_MMPROJ_GPU") != "1":
            cmd += ["--no-mmproj-offload"]
    if env.get("LLM_TEMPLATE"):
        cmd += ["--chat-template-file", env["LLM_TEMPLATE"]]
    cmd += json.loads(env.get("LLM_EXTRA_ARGS", "[]"))
    return cmd


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def stop_engines(port):
    # A shell command containing these patterns can kill its own shell before reaching Tabby.
    patterns = ("[l]lama-server", "[t]abbyAPI/main.py", "[s]erve/server.py.*--engine strata",
                "[/]engine/strata([ -]|$)", "[/]engine/strata-vision([ ]|$)", "[/]ComfyUI/main.py")
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True, text=True)
    time.sleep(2)
    # The listening socket can disappear while Strata's model/vision children still own RAM and VRAM.
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern, "--signal", "KILL"], capture_output=True, text=True)
    for _ in range(30):
        if not port_busy(port):
            return True
        time.sleep(1)
    return False


def installed_paths(directory):
    servers = [path for path in directory.rglob("llama-server") if path.is_file()]
    libraries = [path for path in directory.rglob("lib*.so*") if path.is_file()]
    if len(servers) != 1 or not any(path.name.startswith("libcudart.so") for path in libraries):
        return None
    return servers[0].relative_to(directory), sorted({path.parent.relative_to(directory) for path in libraries})


def install(release):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release):
        raise ValueError(f"invalid llama.cpp release: {release!r}")
    t = time.time()
    root = Path(LLAMA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    directory = root / release
    staging = root / f".{release}.partial"
    with (root / f".{release}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = installed_paths(directory)
        skipped = paths is not None
        if paths is None:
            # A killed setup may leave partial files. Publish only after both archives are extracted.
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            try:
                base = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/"
                suffix = f"{release}-bin-ubuntu-cuda-12.8-x64.tar.gz"
                for name in (f"llama-{suffix}", f"cudart-llama-{suffix}"):
                    path = staging / name
                    urllib.request.urlretrieve(base + name, path)
                    with tarfile.open(path) as archive:
                        archive.extractall(staging, filter="data")
                    path.unlink()
                paths = installed_paths(staging)
                if paths is None:
                    raise RuntimeError(f"llama.cpp {release} archives did not provide llama-server and CUDA libraries")
                if directory.exists():
                    shutil.rmtree(directory)
                staging.rename(directory)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
    server, libs = paths
    event("install_done", engine="llamacpp", release=release, seconds=round(time.time() - t, 1), skipped=skipped)
    return str(directory / server), [str(directory / lib) for lib in libs]


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            if os.path.islink(full):  # a link would count its target twice
                continue
            try:
                total += os.path.getsize(full)
            except OSError:
                pass
    return total


def hf_env(mode, base=None):
    """Env for `hf download`. xet-fast: Xet with HF_XET_HIGH_PERFORMANCE (measured on a G4: 79 GB in ~4 min,
    ~330 MB/s); xet: Xet with its default concurrency; no-xet: plain HTTP (the old behaviour, ~100 MB/s)."""
    if mode not in HF_MODES:
        raise ValueError(f"invalid LLM_HF_MODE {mode!r} (expected one of {', '.join(HF_MODES)})")
    env = {**(os.environ if base is None else base), "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    env.pop("HF_HUB_DISABLE_XET", None)
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    if mode == "no-xet":
        env["HF_HUB_DISABLE_XET"] = "1"
    elif mode == "xet-fast":
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
    return env


def ensure_hf(mode):
    """Make sure the `hf` CLI (and hf_xet unless no-xet) exist; pip-installs huggingface_hub[hf_xet] if not.
    Returns True when it installed. A failed install only fails the run when `hf` itself is missing: without
    hf_xet the Hub client falls back to plain HTTP."""
    py = shlex.quote(sys.executable)
    has_hf = shutil.which("hf") is not None
    has_xet = mode == "no-xet" or sh(f"{py} -c 'import hf_xet'", timeout=120)[0] == 0
    if has_hf and has_xet:
        return False
    event("hf_install", package=HF_PACKAGE, reason="hf CLI missing" if not has_hf else "hf_xet missing")
    t = time.time()
    rc, out = sh(f"{py} -m pip install -q -U {shlex.quote(HF_PACKAGE)}", timeout=900)
    ok = rc == 0 and shutil.which("hf") is not None
    event("hf_install_done", package=HF_PACKAGE, ok=ok, seconds=round(time.time() - t, 1),
          **({} if ok else {"output": out[-300:]}))
    if not ok and shutil.which("hf") is None:
        raise RuntimeError(f"could not install {HF_PACKAGE}: {out[-500:]}")
    return True


def make_links(models, links):
    """Create `name -> target` symlinks inside the model dir (relative, so the dir can move). Idempotent: a
    correct link is kept; a wrong link or a regular file at `name` is replaced atomically. Returns links made."""
    made = 0
    for name, target in (links or {}).items():
        link = os.path.join(models, name)
        real = os.path.join(models, target)
        if not os.path.isfile(real):
            raise RuntimeError(f"link target {target} for {name} is missing")
        rel = os.path.relpath(real, os.path.dirname(link))
        if os.path.islink(link) and os.readlink(link) == rel:
            continue
        if os.path.isdir(link) and not os.path.islink(link):
            raise RuntimeError(f"cannot link {name}: a directory is in the way")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        tmp = f"{link}.colabinet-link"
        if os.path.lexists(tmp):
            os.unlink(tmp)
        os.symlink(rel, tmp)
        os.replace(tmp, link)
        made += 1
    return made


def download(repo, name, links=None, mode="xet-fast", workers=8):
    """Fetch `name` (every shard of a split GGUF) from `repo`. `links` maps file names llama.cpp expects to real
    files in the same repo: link names are not fetched, their targets are, and the links are created after."""
    links = dict(links or {})
    models = model_directory(repo, MODELS)
    expected = gguf_files(name)
    fetch = repo_files(name, links)
    missing = [path for path in fetch if not os.path.isfile(os.path.join(models, path))]

    def total_bytes():
        return sum(os.path.getsize(os.path.join(models, path)) for path in fetch)

    def finish_links():
        made = make_links(models, links)
        absent = [path for path in expected if not os.path.exists(os.path.join(models, path))]
        if absent:
            raise RuntimeError(f"download of {name} is incomplete (missing {absent})")
        return {"links": len(links), "links_made": made} if links else {}

    if not missing:
        event("download_done", file=name, gb=round(total_bytes() / 1e9, 2), seconds=0, skipped=True,
              **finish_links())
        return
    os.makedirs(models, exist_ok=True)
    env = hf_env(mode)
    ensure_hf(mode)
    start_bytes, t = dir_bytes(models), time.time()
    done = threading.Event()

    def progress():
        while not done.wait(10):
            got = dir_bytes(models) - start_bytes
            event("download", file=name, done_gb=round(got / 1e9, 2), mb_s=round(got / (time.time() - t) / 1e6, 1))

    threading.Thread(target=progress, daemon=True).start()
    # One Hub request lets hf fetch independent shards concurrently and resume partial downloads.
    files = " ".join(shlex.quote(path) for path in missing)
    rc, out = sh(f"hf download {shlex.quote(repo)} {files} --local-dir {shlex.quote(models)} "
                 f"--max-workers {int(workers)}", env=env)
    done.set()
    remaining = [path for path in fetch if not os.path.isfile(os.path.join(models, path))]
    if rc != 0 or remaining:
        raise RuntimeError(f"download of {name} failed (missing {remaining}): {out[-500:]}")
    linked = finish_links()
    size, dt = total_bytes(), max(time.time() - t, 1e-6)
    transferred = sum(os.path.getsize(os.path.join(models, path)) for path in missing)
    event("download_done", file=name, gb=round(size / 1e9, 2), seconds=round(dt, 1),
          mb_s=round(transferred / dt / 1e6, 1), hf_mode=mode, **linked)


def prewarm_paths(repo, names, models=MODELS):
    """Absolute paths for the prewarm list (link names allowed), deduplicated by the file they resolve to."""
    directory = model_directory(repo, models)
    paths = {}
    for name in names:
        path = os.path.join(directory, name)
        paths.setdefault(os.path.realpath(path), path)
    return list(paths.values())


def start_prewarm(repo, names, models=MODELS):
    """Launch the page-cache reader in its own session so it outlives this script and never delays the run."""
    paths = prewarm_paths(repo, names, models)
    if not paths:
        return None
    script = os.path.abspath(__file__)
    with open(os.environ.get("OUT") or os.devnull, "a") as out:
        return subprocess.Popen([sys.executable, "-u", script, "--prewarm", *paths], stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, cwd=os.path.dirname(script),
                                start_new_session=True)


def stop_prewarm():
    """A previous run's reader would compete with this run's download for disk bandwidth."""
    subprocess.run(["pkill", "-f", PREWARM_PATTERN], capture_output=True, text=True)


def prewarm(paths, workers=PREWARM_WORKERS, chunk=PREWARM_CHUNK, interval=10.0):
    """Read whole files into the page cache with `workers` parallel os.pread range readers (no vmtouch).
    Emits `prewarm` progress about every `interval` s and one `prewarm_done`; missing files are skipped."""
    files, missing, seen = [], [], set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        try:
            files.append((real, os.path.getsize(real)))
        except OSError:
            missing.append(path)
    total = sum(size for _, size in files)
    extra = {"missing": missing} if missing else {}
    event("prewarm", progress=0.0, gb=0.0, mb_s=0.0, total_gb=round(total / 1e9, 2), files=len(files), **extra)
    t = time.time()
    fds = [os.open(path, os.O_RDONLY) for path, _ in files]
    ranges = [(fd, offset, min(chunk, size - offset))
              for fd, (_, size) in zip(fds, files) for offset in range(0, size, chunk)]
    queue = iter(ranges)
    lock = threading.Lock()
    state = {"bytes": 0, "error": None}

    def reader():
        while True:
            with lock:
                item = next(queue, None) if state["error"] is None else None
            if item is None:
                return
            fd, offset, length = item
            got = 0
            try:
                while got < length:
                    data = os.pread(fd, min(length - got, PREWARM_READ), offset + got)
                    if not data:
                        break
                    got += len(data)
                    del data
            except OSError as e:
                with lock:
                    state["error"] = str(e)
            with lock:
                state["bytes"] += got

    threads = [threading.Thread(target=reader, daemon=True) for _ in range(max(1, min(workers, len(ranges))))]
    try:
        for thread in threads:
            thread.start()
        while True:
            deadline = time.time() + interval
            for thread in threads:
                thread.join(max(0.0, deadline - time.time()))
            if not any(thread.is_alive() for thread in threads):
                break
            got, dt = state["bytes"], max(time.time() - t, 1e-6)
            event("prewarm", progress=round(100 * got / total, 1) if total else 100.0, gb=round(got / 1e9, 2),
                  mb_s=round(got / dt / 1e6, 1))
    finally:
        for thread in threads:
            thread.join()
        for fd in fds:
            os.close(fd)
    got, dt = state["bytes"], max(time.time() - t, 1e-6)
    event("prewarm_done", gb=round(got / 1e9, 2), seconds=round(dt, 1), mb_s=round(got / dt / 1e6, 1),
          files=len(files), **extra, **({"error": state["error"]} if state["error"] else {}))
    return got


def log_tail(n=2000):
    try:
        with open(ENGINE_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def log_size():
    try:
        return os.path.getsize(ENGINE_LOG)
    except OSError:
        return 0


def exit_reason(returncode, log_start):
    """Why llama-server stopped while loading: SIGKILL (measured on a CPU runtime loading a model larger than its
    13.6 GB RAM), or the last error line this run logged."""
    if returncode == -9:
        return "llama-server was killed (SIGKILL), usually by the out-of-memory killer: not enough host RAM"
    try:
        with open(ENGINE_LOG, "rb") as f:
            f.seek(log_start)
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        lines = []
    errors = [line.strip() for line in lines if re.search(r"\berror\b", line, re.IGNORECASE)]
    return f"llama-server exited with {returncode}" + (f": {errors[-1][:300]}" if errors else "")


def main():
    env = dict(os.environ)
    port = env["LLM_PORT"]
    rc, out = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    rc2, torch = sh("python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'", timeout=120)
    event("gpu_info", nvidia_smi=out.strip(), torch=torch.strip())
    if not stop_engines(port):
        raise RuntimeError(f"port {port} is still busy after stopping the engines")
    stop_prewarm()
    server, libs = install(env["LLM_RELEASE"])
    mode = env.get("LLM_HF_MODE") or "xet-fast"
    hf_env(mode)  # reject an unknown mode before any download starts
    workers = int(env.get("LLM_HF_MAX_WORKERS") or 8)
    links = json.loads(env.get("LLM_LINKS") or "{}")
    download(env["LLM_REPO"], env["LLM_FILE"], links=links, mode=mode, workers=workers)
    if env.get("LLM_MMPROJ_FILE"):
        download(env["LLM_REPO"], env["LLM_MMPROJ_FILE"], mode=mode, workers=workers)
    run_env = {**env, "LD_LIBRARY_PATH": ":".join([os.path.dirname(server), *libs, "/usr/lib64-nvidia",
                                                   env.get("LD_LIBRARY_PATH", "")])}
    cmd = build_command(env, server)
    t = time.time()
    log_start = log_size()
    proc = subprocess.Popen(cmd, env=run_env, stdout=open(ENGINE_LOG, "a"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    while time.time() - t < float(env.get("LOAD_TIMEOUT", "600")):
        if proc.poll() is not None:
            raise RuntimeError(exit_reason(proc.returncode, log_start))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        time.sleep(3)
    else:
        raise RuntimeError("llama-server did not become healthy in time")
    props = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            props = json.loads(r.read())
    except Exception:
        pass
    used = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")[1].strip()
    event("server_ready", seconds=round(time.time() - t, 1), vram_used_mib=int(used) if used.isdigit() else None,
          modalities=props.get("modalities"), cmd=" ".join(cmd[1:]))
    names = json.loads(env.get("LLM_PREWARM") or "[]")
    if names:
        try:  # the server is already serving: a prewarm problem is reported, never a setup error
            start_prewarm(env["LLM_REPO"], names)
        except Exception as e:
            event("prewarm_done", gb=0.0, seconds=0, mb_s=0.0, files=0, error=str(e)[:500])


def prewarm_main(paths):
    try:
        prewarm(paths, workers=int(E("LLM_PREWARM_WORKERS") or PREWARM_WORKERS))
    except Exception as e:
        event("prewarm_done", gb=0.0, seconds=0, mb_s=0.0, files=0, error=str(e)[:500])


if __name__ == "__main__" and sys.argv[1:2] == ["--prewarm"]:
    prewarm_main(sys.argv[2:])
elif __name__ == "__main__":
    try:
        main()
    except Exception as e:  # report every failure as an event the supervisor can show
        event("error", message=str(e)[:1000], log_tail=log_tail())
        sys.exit(1)
