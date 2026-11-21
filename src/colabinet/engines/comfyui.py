"""ComfyUI: native image/video nodes, loopback-only server and an explicit model manifest."""
import json
from pathlib import Path

from ..comfy_files import validate_files
from ..config import PACKAGE_DIR
from .base import VM_DIR, VM_ROOT, RunSpec


class ComfyUIEngine:
    name = "comfyui"
    health_path = "/system_stats"
    script = VM_DIR / "setup_comfyui.py"

    @staticmethod
    def validate_spec(spec: RunSpec) -> None:
        validate_files(spec.files)
        if spec.kind not in ("image", "video"):
            raise ValueError("ComfyUI kind 必须为 image 或 video")
        if (spec.extra_args or spec.template or spec.mmproj or spec.links or spec.prewarm
                or spec.vision or spec.vision_offload or spec.mmproj_gpu or spec.refusal_projection):
            raise ValueError("ComfyUI 不支持 LLM 参数、模板或链接；请通过 files 添加模型与 LoRA")

    def uploads(self, spec: RunSpec) -> list[tuple[Path, str]]:
        self.validate_spec(spec)
        return [(self.script, f"{VM_ROOT}/{self.script.name}"),
                (PACKAGE_DIR / "comfy_files.py", f"{VM_ROOT}/comfy_files.py")]

    def setup_env(self, spec: RunSpec, vm_port: int, run_id: int) -> dict[str, str]:
        self.validate_spec(spec)
        return {"SCRIPT": f"{VM_ROOT}/{self.script.name}", "OUT": f"{VM_ROOT}/setup.out",
                "EVENTS": f"{VM_ROOT}/run-{run_id}.jsonl", "ENGINE_LOG": f"{VM_ROOT}/engine.log",
                "COMFY_FILES": json.dumps(spec.files), "COMFY_PORT": str(vm_port),
                "COMFY_RELEASE": spec.release or "master", "HF_TOKEN": ""}
