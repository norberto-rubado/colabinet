"""Model presets from presets.toml (stdlib tomllib). Errors are reported, never fatal to the app."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KNOWN_ENGINES = {"llamacpp", "exllamav3", "strata"}
FALLBACK_CLASS = {"A100-80": "A100", "T4-HIGHMEM": "T4"}


class PresetError(ValueError):
    pass


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
    gen_tps: float  # generation, on a short prompt
    prefill_tps: float | None = None  # prompt processing on a ~4K-token prompt; None when too slow to try
    date: str = ""
    note: str = ""


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
        if raw["id"] in presets:
            raise PresetError(f"{where}：id {raw['id']!r} 重复")
        fields = {k: v for k, v in raw.items() if k not in ("gpu", "evidence", "measured")}
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
                if not all(isinstance(value, (int, float)) and value > 0
                           for value in (speed.gen_tps, speed.prefill_tps) if value is not None):
                    raise ValueError(f"measured.{name}：gen_tps 和 prefill_tps 必须是正数")
            presets[raw["id"]] = Preset(**fields, gpu=gpu, evidence=evidence, measured=measured)
        except (TypeError, ValueError) as e:
            raise PresetError(f"{where}：{e}") from e
    return Presets(engines=data.get("engines", {}), presets=presets)
