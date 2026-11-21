"""What every engine adapter provides: which files go to the VM and the env for its setup script."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..config import PACKAGE_DIR
from ..presets import Preset

VM_ROOT = "/content/colabinet"
VM_DIR = PACKAGE_DIR / "vm"


@dataclass(frozen=True)
class RunSpec:
    engine: str
    repo: str
    file: str | None
    alias: str
    ctx: int
    kv: str
    parallel: int = 1
    mmproj: str | None = None
    mmproj_gpu: bool = False
    template: str | None = None  # file name in src/colabinet/chat_templates/
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    preset_id: str | None = None
    release: str | None = None
    revision: str | None = None
    vision: bool = False
    vision_offload: bool = False
    refusal_projection: bool = False
    links: dict[str, str] = field(default_factory=dict)  # llama.cpp: expected file name -> real file in the repo
    prewarm: tuple[str, ...] = field(default_factory=tuple)  # llama.cpp: files read into page cache when ready

    files: tuple[dict, ...] = ()
    kind: str = "text"

    def __post_init__(self) -> None:
        # spec_json stores tuples as lists, and rows written before a field existed lack it: normalise both.
        object.__setattr__(self, "extra_args", tuple(self.extra_args or ()))
        object.__setattr__(self, "links", dict(self.links or {}))
        object.__setattr__(self, "files", tuple(dict(f) for f in self.files or ()))
        object.__setattr__(self, "prewarm", tuple(self.prewarm or ()))

    @classmethod
    def from_preset(cls, preset: Preset, gpu_class: str, engines: dict[str, dict], **overrides) -> RunSpec:
        d = preset.defaults_for(gpu_class)
        pins = engines.get(preset.engine, {})
        release = preset.release or pins.get("release") or (f"{pins['wheel']}@{pins['tabbyapi_commit']}"
                                           if pins.get("wheel") and pins.get("tabbyapi_commit") else None)
        spec = cls(engine=preset.engine, repo=preset.repo, file=preset.file, alias=preset.alias or preset.id,
                   ctx=d.ctx, kv=d.kv, parallel=preset.parallel, mmproj=preset.mmproj, mmproj_gpu=d.mmproj_gpu,
                   template=preset.template, preset_id=preset.id, release=release, revision=preset.revision,
                   vision=preset.vision, vision_offload=preset.vision_offload,
                   refusal_projection=preset.refusal_projection, extra_args=d.extra_args,
                   links=dict(preset.links), prewarm=tuple(preset.prewarm), files=preset.files, kind=preset.kind)
        clean = {k: v for k, v in overrides.items() if v is not None}
        return cls(**{**spec.__dict__, **clean})


class Engine(Protocol):
    name: str
    health_path: str

    def uploads(self, spec: RunSpec) -> list[tuple[Path, str]]:
        """(local file, remote path) pairs to upload after vm/prepare.py has created the directories."""

    def setup_env(self, spec: RunSpec, vm_port: int, run_id: int) -> dict[str, str]:
        """Env for bg.py: must include SCRIPT and OUT, plus whatever the setup script reads."""
