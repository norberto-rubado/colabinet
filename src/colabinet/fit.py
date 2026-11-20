"""Will a model fit a GPU? Estimated on the Mac from Hugging Face file metadata (no download, no VM, no CU).
It warns, never blocks: llama.cpp can still run by offloading layers to the CPU, just slowly (spec §6)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import VRAM_GB
from .vm.gguf_files import gguf_files

KV_RATIO = {"f16": 1.0, "bf16": 1.0, "f32": 2.0, "f8_e4m3": 0.5, "q8_0": 34 / 64,
            "q5_1": 24 / 64, "q5_0": 22 / 64, "q4_1": 20 / 64, "q4_0": 18 / 64, "iq4_nl": 18 / 64,

            "FP16": 1.0, "Q8": 0.5, "Q6": 0.375, "Q4": 0.25}


def kv_ratio(kv: str) -> float:
    """llama.cpp types, TabbyAPI names, or TabbyAPI's "k,v" bit pairs (e.g. "8,8" or "4,6")."""
    if "," in kv:
        key, value = (int(item) for item in kv.split(",", 1))
        return (key + value) / 32
    return KV_RATIO.get(kv, 1.0)


OVERHEAD_GB = 0.8
# ExLlamaV3 reserves working memory for its autosplit on top of weights and cache. Measured 2026-10-13: the 27B
# 4.0 bpw build (13.2 GB on GPU) failed "Insufficient VRAM in split for model and cache" on a T4 at 32K even with
# a Q8 cache (~1.1 GB), and used ~1.2 GB beyond weights and cache on an L4. 2 GB reproduces both outcomes.
EXL3_OVERHEAD_GB = 2.0
TIGHT = 0.90


@dataclass(frozen=True)
class Fit:
    verdict: str
    total_gb: float | None
    kv_gb: float | None
    vram_gb: float
    message: str


def kv_gb(ctx: int, kv: str, bytes_per_token_f16: int | None) -> float | None:
    if not bytes_per_token_f16:
        return None
    return ctx * bytes_per_token_f16 * kv_ratio(kv) / 1e9


def estimate(gpu_class: str, weights_gb: float | None, ctx: int, kv: str, bytes_per_token_f16: int | None = None,
             mmproj_gb: float = 0.0, mmproj_gpu: bool = False, overhead_gb: float = OVERHEAD_GB) -> Fit:
    vram = VRAM_GB.get(gpu_class, 0.0)
    if vram == 0:
        return Fit("cpu", None, None, 0.0, "CPU 运行时：在 CPU 上运行（适合测试，运行真实模型会很慢）")
    if weights_gb is None:
        return Fit("unknown", None, None, vram, "模型大小未知")
    kv_size = kv_gb(ctx, kv, bytes_per_token_f16)
    total = weights_gb + (kv_size or 0.0) + (mmproj_gb if mmproj_gpu else 0.0) + overhead_gb
    parts = f"权重 {weights_gb:.1f} GB"
    if kv_size is not None:
        parts += f" + KV {kv_size:.1f} GB（{ctx // 1024}K，{kv}）"
    if mmproj_gpu and mmproj_gb:
        parts += f" + 视觉 {mmproj_gb:.1f} GB"
    parts += f" + 开销约 {overhead_gb} GB ≈ {total:.1f} / {vram:g} GB"
    if kv_size is None:
        if total > vram:
            return Fit("spills", total, None, vram, f"{parts}：需要 CPU 卸载；此模型的 KV 大小未知")
        verdict = "tight" if weights_gb + overhead_gb > TIGHT * vram else "fits"
        return Fit(verdict, total, None, vram, f"{parts}；此模型的 KV 大小未知，请检查上下文长度")
    if total > vram:
        return Fit("spills", total, kv_size, vram, f"{parts}：llama.cpp 会把部分层卸载到 CPU（会慢很多；"
                   "可尝试更小的 KV 类型或上下文）")
    if total > TIGHT * vram:
        return Fit("tight", total, kv_size, vram, f"{parts}：偏紧，但能放下")
    return Fit("fits", total, kv_size, vram, f"{parts}：能放下")


async def hf_file_gb(client: httpx.AsyncClient, repo: str, path: str, revision: str = "main") -> float | None:
    """Complete model size from resolve headers, including every shard of a split GGUF."""
    try:
        paths = gguf_files(path)
    except ValueError:
        return None
    total = 0.0
    for shard in paths:
        size = await _hf_file_gb(client, repo, shard, revision)
        if size is None:
            return None
        total += size
    return total


async def _hf_file_gb(client: httpx.AsyncClient, repo: str, path: str, revision: str) -> float | None:
    try:
        row = await client.head(f"https://huggingface.co/{repo}/resolve/{revision}/{path}", follow_redirects=False,
                              timeout=10)
    except httpx.HTTPError:
        return None
    size = row.headers.get("x-linked-size") or row.headers.get("content-length")
    if row.status_code >= 400 or not size:
        return None
    try:
        size_bytes = int(size)
    except ValueError:
        return None
    return size_bytes / 1e9 if size_bytes > 0 else None
