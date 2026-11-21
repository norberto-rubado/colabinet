# Runs ON the Colab VM, detached via bg.py. Never installs/replaces the system torch stack.
import concurrent.futures
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from comfy_files import validate_files

ROOT = Path(os.environ.get("COLABINET_ROOT", "/content/colabinet"))
COMFY = Path(os.environ.get("COMFY_DIR", "/content/ComfyUI"))
CACHE = Path(os.environ.get("COMFY_CACHE", "/content/hf"))
EVENTS = Path(os.environ.get("EVENTS", str(ROOT / "setup.jsonl")))
LOG = Path(os.environ.get("ENGINE_LOG", str(ROOT / "engine.log")))
T0 = time.time()
LOCK = threading.Lock()


def event(kind, **kw):
    row = json.dumps({"t": round(time.time() - T0, 1), "event": kind, **kw})
    with LOCK:
        with EVENTS.open("a") as f:
            f.write(row + "\n")
        print(row, flush=True)


def run(argv, **kw):
    result = subprocess.run(argv, capture_output=True, text=True, **kw)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed: {(result.stdout + result.stderr)[-1500:]}")
    return result.stdout.strip()


def torch_constraints(python=sys.executable):
    # Exact pins make pip fail rather than silently downgrade the working CUDA 13/Blackwell stack.
    raw = run([python, "-c", "import importlib.metadata as m, json; "
               "print(json.dumps({p:m.version(p) for p in ('torch','torchvision','torchaudio')}))"])
    return "".join(f"{p}=={v}\n" for p, v in json.loads(raw).items())


def install(release="master"):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release):
        raise ValueError("invalid ComfyUI release")
    t = time.time()
    if not (COMFY / ".git").exists():
        run(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI", str(COMFY)], timeout=600)
    if release != "master":
        run(["git", "-C", str(COMFY), "fetch", "--depth", "1", "origin", release], timeout=600)
        run(["git", "-C", str(COMFY), "checkout", "--detach", "FETCH_HEAD"], timeout=120)
    constraints = ROOT / "torch-constraints.txt"
    constraints.write_text(torch_constraints())
    run([sys.executable, "-m", "pip", "install", "-c", str(constraints), "-r", str(COMFY / "requirements.txt"),
         "huggingface_hub[hf_xet]"], timeout=1200)
    event("install_done", engine="comfyui", release=run(["git", "-C", str(COMFY), "rev-parse", "HEAD"]),
          seconds=round(time.time() - t, 1))


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                p = Path(root) / name
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


def download_file(item, env, attempts=4) -> int:
    repo, name, directory = (item[k] for k in ("repo", "file", "dir"))
    local = CACHE / repo
    source = local / name
    for attempt in range(attempts):
        try:
            run(["hf", "download", repo, name, "--local-dir", str(local)], env=env, timeout=3600)
            if not source.is_file():
                raise RuntimeError(f"download missing {repo}/{name}")
            if not source.resolve().is_relative_to(local.resolve()):  # e.g. a repo symlink to /etc or ../other
                raise RuntimeError(f"{repo}/{name} resolves outside its repository directory")
            target = COMFY / "models" / directory / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".colabinet-link")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(source)
            temporary.replace(target)
            return source.stat().st_size
        except (RuntimeError, subprocess.TimeoutExpired):
            if attempt + 1 == attempts:
                raise
            event("download_retry", file=name, attempt=attempt + 1)
            time.sleep(20 * 2 ** attempt + random.random() * 5)
    raise RuntimeError("download attempts exhausted")


def download(files, workers=8):
    validate_files(files)
    env = {**os.environ, "HF_XET_HIGH_PERFORMANCE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    env.pop("HF_HUB_DISABLE_XET", None)
    t, start = time.time(), dir_bytes(CACHE)
    # Aggregate events, not sum of overlapping per-file durations (supervisor sums download_done).
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download_file, item, env) for item in files}
        total = 0
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=10,
                                                    return_when=concurrent.futures.FIRST_COMPLETED)
            for task in done:
                total += int(task.result())  # fail setup, never declare ready after a failed download
            got, elapsed = max(0, dir_bytes(CACHE) - start), max(time.time() - t, 0.001)
            event("download", file="ComfyUI", done_gb=round(got / 1e9, 2), mb_s=round(got / elapsed / 1e6, 1))
    event("download_done", file="ComfyUI", gb=round(total / 1e9, 3), seconds=round(time.time() - t, 1))


def build_command(port):
    return [sys.executable, str(COMFY / "main.py"), "--listen", "127.0.0.1", "--port", str(port)]


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def stop_engines(port):
    # Also the llama.cpp page-cache prewarm reader: it would compete with the model download for disk bandwidth.
    patterns = ("[l]lama-server", "[t]abbyAPI/main.py", "[s]erve/server.py.*--engine strata",
                "[/]engine/strata([ -]|$)", "[/]engine/strata-vision([ ]|$)", "[/]ComfyUI/main.py",
                "[s]etup_llamacpp.py --prewarm")
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)
    time.sleep(2)
    for pattern in patterns:
        subprocess.run(["pkill", "-KILL", "-f", pattern], capture_output=True)
    # A stale listener would answer the first /system_stats probe and fake server_ready.
    for _ in range(30):
        if not port_busy(port):
            return True
        time.sleep(1)
    return False


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    files = json.loads(os.environ["COMFY_FILES"])
    validate_files(files)
    port = os.environ["COMFY_PORT"]
    if not stop_engines(port):
        raise RuntimeError(f"port {port} is still busy after stopping the engines")
    install(os.environ.get("COMFY_RELEASE", "master"))
    download(files)
    t = time.time()
    with LOG.open("a") as out:
        proc = subprocess.Popen(build_command(os.environ["COMFY_PORT"]), cwd=COMFY, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - t < 600:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with {proc.returncode}; see engine.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{os.environ['COMFY_PORT']}/system_stats", timeout=5) as r:
                if r.status == 200:
                    event("server_ready", seconds=round(time.time() - t, 1))
                    return
        except OSError:
            pass
        time.sleep(3)
    proc.terminate()
    raise RuntimeError("ComfyUI did not become healthy in time")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        event("error", message=str(exc)[-1500:])
        sys.exit(1)
