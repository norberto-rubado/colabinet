"""API-format graphs ported from the verified G4 benchmark and h3-shortdrama.

Templates are packaged data; every invocation loads a fresh graph. No user prompt becomes code or a path.
MiniMaxH3ImageToVideo.first_frame is optional (ComfyUI afcb984): omitting it is text-to-video.
"""
import json
import math
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

WORKFLOWS = ("moss", "kroma", "krea2turbo", "h3", "h3edit", "anima", "wai")
# Distilled/turbo graphs sample at cfg=1.0, where ComfyUI skips the uncond pass entirely (comfy/samplers.py), so a
# negative prompt would be accepted and silently ignored. Only graphs with real CFG take one.
NEGATIVE_WORKFLOWS = ("moss", "wai")
TEMPLATES = Path(__file__).with_name("workflows")


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1, max_length=32768)
    size: str = "1024x1024"
    n: StrictInt = Field(default=1, ge=1, le=8)
    response_format: Literal["b64_json", "url"] = "b64_json"
    workflow: Literal["moss", "kroma", "krea2turbo", "h3", "h3edit", "anima", "wai"] = "moss"
    negative_prompt: str | None = Field(default=None, max_length=32768)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**64 - 1)
    steps: StrictInt | None = Field(default=None, ge=1, le=100)
    image: str | None = Field(default=None, max_length=45_000_000)
    upscale: StrictBool = False

    @field_validator("size")
    @classmethod
    def check_size(cls, value):
        dimensions(value)
        return value

    @model_validator(mode="after")
    def check_negative(self):
        if self.negative_prompt and self.workflow not in NEGATIVE_WORKFLOWS:
            raise ValueError(f"{self.workflow} 是蒸馏/Turbo 工作流（CFG=1），负面提示词不会生效；"
                             f"只有 {'、'.join(NEGATIVE_WORKFLOWS)} 支持 negative_prompt")
        return self


class Lora(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)
    strength: float = Field(default=1.0, ge=-4, le=4, allow_inf_nan=False)

    @field_validator("name")
    @classmethod
    def check_name(cls, value):
        if value.startswith(("/", "-")) or "\\" in value or "\x00" in value or ".." in value.split("/"):
            raise ValueError("LoRA 名称必须是模型目录内的相对路径")
        return value


class VideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str = Field(min_length=1, max_length=32768)
    image: str | None = Field(default=None, max_length=45_000_000)
    seconds: float = Field(default=5, ge=4, le=15, allow_inf_nan=False)
    size: str = "768x1344"
    loras: list[Lora] = Field(default_factory=list, max_length=16)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**64 - 1)
    steps: StrictInt = Field(default=8, ge=1, le=100)

    @field_validator("size")
    @classmethod
    def check_size(cls, value):
        dimensions(value, video=True)
        return value


def dimensions(size: str, video=False) -> tuple[int, int]:
    if not isinstance(size, str) or not re.fullmatch(r"\d{3,4}x\d{3,4}", size):
        raise ValueError("尺寸格式必须是 宽x高，例如 1024x1024")
    w, h = map(int, size.split("x"))
    maximum = 2048 if video else 4096
    if any(x % 32 or not 256 <= x <= maximum for x in (w, h)) or (video and w * h > 2_097_152):
        raise ValueError(f"宽高必须是 32 的倍数，范围 256–{maximum}" + ("；视频面积最多 2097152 像素" if video else ""))
    return w, h


def frames(seconds: float) -> int:
    # Round up, never produce less than the requested duration. 5 s -> 124, 6 s -> 158.
    wanted = math.ceil(seconds * 24)
    return wanted + (5 - wanted % 17) % 17


def template(name):
    return json.loads((TEMPLATES / f"{name}.json").read_text())


def image_graph(body: ImageRequest, seed: int, image_name: str | None = None) -> dict:
    if body.workflow == "h3edit" and not image_name:
        raise ValueError("H3 编辑需要 image 输入图像")
    g = template(body.workflow)
    w, h = dimensions(body.size)
    positive, negative = [], []
    for node in g.values():
        inp, kind = node["inputs"], node["class_type"]
        if kind == "CLIPTextEncode":
            (positive if inp["text"] == "__PROMPT__" else negative).append(inp)
        if kind == "MiniMaxH3ImageConditioning":
            inp.update(prompt=body.prompt, width=w, height=h)
            if body.workflow == "h3edit":
                inp.update(width=0, height=0)  # preserve input canvas as in the verified edit graph
        if kind.startswith("Empty") and "width" in inp:
            inp.update(width=w, height=h)
        if kind == "LoadImage":
            inp["image"] = image_name
        if kind == "KSampler":
            inp["seed"] = seed
            if body.steps is not None:
                inp["steps"] = body.steps
    for inp in positive:
        inp["text"] = body.prompt
    if body.negative_prompt is not None:
        for inp in negative:
            inp["text"] = body.negative_prompt
        # Distilled workflows normally zero-out conditioning; honour explicit negative_prompt too.
        for node in g.values():
            if node["class_type"] == "ConditioningZeroOut":
                source = node["inputs"]["conditioning"][0]
                clip = g[source]["inputs"]["clip"]
                node.update(class_type="CLIPTextEncode", inputs={"clip": clip, "text": body.negative_prompt})
    return g


def upscale_graph(seed: int, image_name: str) -> dict:
    g = template("seedvr")
    g["1"]["inputs"]["image"] = image_name
    g["54"]["inputs"]["seed"] = seed
    return g


def video_graph(body: VideoRequest, seed: int, image_name: str | None = None) -> dict:
    g = template("video")
    w, h = dimensions(body.size, video=True)
    g["i2v"]["inputs"].update(prompt=body.prompt, width=w, height=h, length=frames(body.seconds))
    if image_name:
        g["img"]["inputs"]["image"] = image_name
        # MiniMaxH3ImageToVideo resizes first_frame to width x height without keeping aspect: a 1:1 frame came out
        # visibly squashed in a 1344x768 clip (G4 test). Center-crop to the target aspect first.
        g["fit"] = {"class_type": "ImageScale", "inputs": {"image": ["img", 0], "upscale_method": "lanczos",
                                                           "width": w, "height": h, "crop": "center"}}
        g["i2v"]["inputs"]["first_frame"] = ["fit", 0]
    else:
        del g["img"]
        del g["i2v"]["inputs"]["first_frame"]
    g["noise"]["inputs"]["noise_seed"] = seed
    g["sched"]["inputs"]["steps"] = body.steps
    previous = "lora"  # the Turbo-8 LoRA is always first
    for i, lora in enumerate(body.loras):
        name = f"extra_lora_{i}"
        g[name] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [previous, 0], "lora_name": lora.name, "strength_model": lora.strength}}
        previous = name
    for key in ("sched", "guider"):
        g[key]["inputs"]["model"] = [previous, 0]
    return g
