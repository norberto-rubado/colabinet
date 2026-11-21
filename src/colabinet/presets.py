"""Model presets from presets.toml (stdlib tomllib). Errors are reported, never fatal to the app."""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

KNOWN_ENGINES = {"llamacpp", "exllamav3", "strata", "comfyui"}
FALLBACK_CLASS = {"A100-80": "A100", "T4-HIGHMEM": "T4"}


class PresetError(ValueError):
    pass


def _model_path(value: object) -> bool:
    """A non-empty relative path that stays inside the model directory."""
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and all(part not in ("", ".", "..") for part in parts)


def _llamacpp_fields(where: str, raw: dict) -> None:
    """Validate the llama.cpp-only fields release, links and prewarm."""
    for key in ("release", "links", "prewarm"):
        if key in raw and raw["engine"] != "llamacpp":
            raise PresetError(f"{where}：只有 llamacpp 预设支持 {key}")
    if "release" in raw and not (isinstance(raw["release"], str) and re.fullmatch(r"b\d+", raw["release"])):
        raise PresetError(f"{where}：release 必须是 llama.cpp 发布标签（例如 b12359）")
    links = raw.get("links", {})
    if not isinstance(links, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in links.items()):
        raise PresetError(f"{where}：links 必须是“期望的文件名 = 仓库中的真实文件名”的字符串表")
    for name, target in links.items():
        for path in (name, target):
            if not _model_path(path):
                raise PresetError(f"{where}：links 中的 {path!r} 必须是模型目录内的相对路径")
        if name == target:
            raise PresetError(f"{where}：links 中的 {name!r} 不能指向自身")
        if target in links:
            raise PresetError(f"{where}：links 的目标 {target!r} 本身也是链接，不支持链式链接")
    prewarm = raw.get("prewarm", [])
    if not isinstance(prewarm, list) or not all(isinstance(item, str) for item in prewarm):
        raise PresetError(f"{where}：prewarm 必须是字符串列表")
    for path in prewarm:
        if not _model_path(path):
            raise PresetError(f"{where}：prewarm 中的 {path!r} 必须是模型目录内的相对路径")


@dataclass(frozen=True)
class GpuDefaults:
    ctx: int = 32768
    kv: str = "f16"
    mmproj_gpu: bool = False
    extra_args: tuple[str, ...] = ()  # engine args for this GPU class, e.g. Strata's ("--low-ram", "on")


@dataclass(frozen=True)
class GpuEvidence:
    status: str
    message: str
    performance: str = ""  # Saved benchmark summary, separate from local request aggregates.


@dataclass(frozen=True)
class Measured:
    """Speeds measured on a real runtime of one GPU class at the preset's defaults, engine-reported."""
    gen_tps: float | None = None  # LLM generation only; media uses seconds_per_output
    prefill_tps: float | None = None  # prompt processing on a ~4K-token prompt; None when too slow to try
    date: str = ""
    note: str = ""
    seconds_per_output: float | None = None
    unit: str = ""


@dataclass(frozen=True)
class Preset:
    id: str
    engine: str
    label: str
    repo: str
    file: str | None = None
    revision: str | None = None
    mmproj: str | None = None
    template: str | None = None
    alias: str | None = None
    weights_gb: float | None = None
    kv_bytes_per_token_f16: int | None = None
    licence: str | None = None
    parallel: int = 1
    gpu: dict[str, GpuDefaults] = field(default_factory=dict)
    vision: bool = False  # Bundled ExLlamaV3/Strata vision; llama.cpp uses mmproj instead.
    vision_offload: bool = False  # Keep ExLlamaV3 vision weights in pinned system RAM.
    refusal_projection: bool = False  # Strata's experimental refusal-direction projection, opt-in.
    evidence: dict[str, GpuEvidence] = field(default_factory=dict)
    measured: dict[str, Measured] = field(default_factory=dict)
    release: str | None = None  # llama.cpp release tag for this preset only, overrides engines.llamacpp.release
    links: dict[str, str] = field(default_factory=dict)  # name llama.cpp expects -> real file in the same repo
    prewarm: tuple[str, ...] = ()  # files read into the VM page cache after the server is healthy

    files: tuple[dict, ...] = ()  # ComfyUI {repo, file, dir, size_gb?}
    kind: str = "text"  # text | image | video

    def defaults_for(self, gpu_class: str) -> GpuDefaults:
        generic = GpuDefaults(kv={"exllamav3": "FP16", "strata": "int8"}.get(self.engine, "f16"))
        return (self.gpu.get(gpu_class) or self.gpu.get(FALLBACK_CLASS.get(gpu_class, ""))
                or self.gpu.get("default", generic))


@dataclass(frozen=True)
class Presets:
    engines: dict[str, dict]
    presets: dict[str, Preset]


def load_presets(path: Path) -> Presets:
    try:
        data = tomllib.loads(Path(path).read_text())
    except FileNotFoundError as e:
        raise PresetError(f"未找到 {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise PresetError(f"{path}: {e}") from e
    presets: dict[str, Preset] = {}
    for i, raw in enumerate(data.get("preset", [])):
        where = f"第 {i + 1} 个预设"
        for key in ("id", "engine", "label", "repo"):
            if not raw.get(key):
                raise PresetError(f"{where}：缺少 '{key}'")
        if raw["engine"] not in KNOWN_ENGINES:
            raise PresetError(f"{where}：未知推理引擎 {raw['engine']!r}")
        if raw["engine"] in ("llamacpp", "strata") and not raw.get("file"):
            raise PresetError(f"{where}：{raw['engine']} 预设需要 'file'")
        if raw["engine"] == "exllamav3" and not raw.get("revision"):
            raise PresetError(f"{where}：exllamav3 预设需要 'revision'（码率分支，例如 4.00bpw）")
        for key in ("vision", "vision_offload", "refusal_projection"):
            if key in raw and not isinstance(raw[key], bool):
                raise PresetError(f"{where}：{key} 必须是布尔值（true 或 false）")
        if raw.get("refusal_projection") and raw["engine"] != "strata":
            raise PresetError(f"{where}：只有 Strata 支持 refusal_projection")
        _llamacpp_fields(where, raw)
        if raw["engine"] == "comfyui":
            from .comfy_files import validate_files
            try:
                validate_files(raw.get("files"))
                if raw.get("kind") not in ("image", "video"):
                    raise ValueError("ComfyUI kind 必须为 image 或 video")
            except ValueError as e:
                raise PresetError(f"{where}：{e}") from e
        elif "files" in raw or raw.get("kind", "text") != "text":
            raise PresetError(f"{where}：files 和媒体 kind 只用于 ComfyUI")
        if raw["id"] in presets:
            raise PresetError(f"{where}：id {raw['id']!r} 重复")
        fields = {k: v for k, v in raw.items() if k not in ("gpu", "evidence", "measured")}
        if "files" in fields:
            fields["files"] = tuple(dict(f) for f in fields["files"])
        if "links" in fields:
            fields["links"] = dict(fields["links"])
        if "prewarm" in fields:
            fields["prewarm"] = tuple(fields["prewarm"])
        try:
            gpu = {}
            for name, values in raw.get("gpu", {}).items():
                extra = values.get("extra_args", [])
                if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
                    raise ValueError(f"gpu.{name}.extra_args 必须是字符串列表")
                gpu[name] = GpuDefaults(**{**GpuDefaults().__dict__, **values, "extra_args": tuple(extra)})
            evidence = {k: GpuEvidence(**v) for k, v in raw.get("evidence", {}).items()}
            if any(e.status not in ("tested", "reported", "build_only") for e in evidence.values()):
                raise ValueError("evidence status 必须是 tested、reported 或 build_only")
            measured = {k: Measured(**v) for k, v in raw.get("measured", {}).items()}
            for name, speed in measured.items():
                if not all(type(value) in (int, float) and math.isfinite(value) and value > 0
                           for value in (speed.gen_tps, speed.prefill_tps, speed.seconds_per_output)
                           if value is not None):
                    raise ValueError(f"measured.{name}：gen_tps 和 prefill_tps 必须是正数")
                if raw["engine"] == "comfyui" and (speed.seconds_per_output is None or speed.unit != raw["kind"]):
                    raise ValueError(f"measured.{name}：媒体测量需要 seconds_per_output 和匹配的 unit")
                if raw["engine"] != "comfyui" and speed.gen_tps is None:
                    raise ValueError(f"measured.{name}：LLM 测量需要 gen_tps")
            presets[raw["id"]] = Preset(**fields, gpu=gpu, evidence=evidence, measured=measured)
        except (TypeError, ValueError) as e:
            raise PresetError(f"{where}：{e}") from e
    return Presets(engines=data.get("engines", {}), presets=presets)
