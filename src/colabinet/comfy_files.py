"""Safe, serializable ComfyUI model manifests (also uploaded to the VM)."""
import math
import re
from pathlib import PurePosixPath

REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
RELEASE_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"  # same as vm/setup_comfyui.install
MODEL_DIRS = {"diffusion_models", "text_encoders", "vae", "loras", "checkpoints", "upscale_models",
              "clip_vision", "controlnet", "embeddings"}


def safe_relative_path(value):
    """A relative path that stays inside its directory, checked per segment: no leading '/' or '-', no backslash or
    NUL, no empty, '.' or '..' segment (so no '//', './' or trailing '/'). Shared with presets._model_path."""
    return (isinstance(value, str) and bool(value) and not value.startswith(("/", "-")) and "\\" not in value
            and "\x00" not in value and all(part not in ("", ".", "..") for part in value.split("/")))


def valid_release(value):
    return isinstance(value, str) and re.fullmatch(RELEASE_RE, value) is not None


def validate_files(files):
    if not isinstance(files, (list, tuple)) or not files:
        raise ValueError("ComfyUI 需要非空的 files 模型文件列表")
    targets = set()
    for item in files:
        if not isinstance(item, dict) or set(item) - {"repo", "file", "dir", "size_gb"}:
            raise ValueError("files 条目必须包含 repo、file、dir，可选 size_gb")
        repo, name, directory = (item.get(k) for k in ("repo", "file", "dir"))
        if not isinstance(repo, str) or not re.fullmatch(REPO_RE, repo):
            raise ValueError("files.repo 必须是 Hugging Face 的 org/repo")
        if not isinstance(name, str) or not safe_relative_path(name):
            raise ValueError("files.file 必须是仓库内的安全相对路径")
        if not isinstance(directory, str) or directory not in MODEL_DIRS:
            raise ValueError("files.dir 不是支持的 ComfyUI 模型目录")
        target = (directory, PurePosixPath(name).name)
        if target in targets:
            raise ValueError("files 中的模型目标文件名重复")
        targets.add(target)
        size = item.get("size_gb")
        if size is not None and (type(size) not in (int, float) or not math.isfinite(size) or size <= 0):
            raise ValueError("files.size_gb 必须是有限正数")


def disk_gb(files):
    return sum(f["size_gb"] for f in files) if files and all(f.get("size_gb") for f in files) else None
