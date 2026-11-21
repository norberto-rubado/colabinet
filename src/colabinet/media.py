"""Shared ComfyUI jobs for the authenticated gateway and local CSRF-protected playground.

Only loopback session proxies are contacted. Outputs and video status live under data_dir/media, expire in 24h,
and survive a gateway restart (unfinished jobs become failed, never silently disappear or resubmit inference).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import mimetypes
import os
import re
import secrets
import time
import uuid
from pathlib import Path

import httpx

from .activity import Activity
from .comfy_graphs import ImageRequest, VideoRequest, image_graph, upscale_graph, video_graph

TTL_S = 86400
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
# The proxy records every request; the User-Agent tells request_stats who asked. Public gateway jobs are real usage,
# playground jobs are tests (is_playground), and internal status polling is excluded from the stats entirely.
USER_AGENTS = {"gateway": "colabinet-gateway", "playground": "colabinet-playground"}
POLL_UA = "colabinet-poll"


class MediaError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class PromptVanished(MediaError):
    """The prompt is in neither /history nor /queue: ComfyUI restarted or the job was deleted/interrupted."""


class MediaFiles:
    def __init__(self, root: Path, now=time.time):
        self.root, self.now = Path(root), now
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def path(self, ident, suffix):
        if not isinstance(ident, str) or not re.fullmatch(r"(?:file|video)_[0-9a-f]{32}", ident):
            raise MediaError("未找到文件或任务", 404)
        return self.root / f"{ident}.{suffix}"

    def write_json(self, ident, data):
        path = self.path(ident, "json")
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(data, ensure_ascii=False))
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def read(self, ident):
        try:
            data = json.loads(self.path(ident, "json").read_text())
        except (OSError, ValueError):
            raise MediaError("未找到文件或任务", 404) from None
        if self.now() >= data["expires_at"]:
            self.delete(ident)
            raise MediaError("文件或任务已过期（保存 24 小时）", 404)
        return data

    def delete(self, ident):
        for suffix in ("json", "bin", "partial"):
            self.path(ident, suffix).unlink(missing_ok=True)

    def prune(self):
        for path in self.root.glob("*.json"):
            with contextlib.suppress(MediaError):
                self.read(path.stem)

    def save(self, data: bytes, content_type: str, ident=None):
        ident = ident or "file_" + uuid.uuid4().hex
        path = self.path(ident, "bin")
        path.write_bytes(data)
        os.chmod(path, 0o600)
        self.write_json(ident, {"id": ident, "content_type": content_type,
                               "expires_at": self.now() + TTL_S})
        return ident

    def content(self, ident):
        meta = self.read(ident)
        if ident.startswith("video_") and meta.get("status") != "completed":
            raise MediaError("视频尚未完成", 409)
        path = self.path(ident, "bin")
        if not path.is_file():
            raise MediaError("未找到生成文件", 404)
        return path, meta.get("content_type", "application/octet-stream")


def decode_image(value):
    if not isinstance(value, str):
        raise MediaError("image 必须是 base64 或 data URL")
    ctype = ""
    if value.startswith("data:"):
        head, sep, value = value.partition(",")
        if not sep or not head.endswith(";base64"):
            raise MediaError("image data URL 必须使用 base64")
        ctype = head[5:-7]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("image 不是有效的 base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise MediaError("输入图像不能为空且不得超过 32 MiB")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        ext, real_type = ".png", "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        ext, real_type = ".jpg", "image/jpeg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        ext, real_type = ".webp", "image/webp"
    else:
        raise MediaError("输入图像仅支持 PNG、JPEG 或 WebP")
    if ctype and ctype != real_type:
        raise MediaError("data URL 的图像类型与内容不符")
    return data, ext, real_type


def output_refs(history, video=False):
    refs = []
    for output in history.get("outputs", {}).values():
        for key in ("images", "videos", "gifs"):
            for ref in output.get(key, []):
                if not isinstance(ref, dict) or not isinstance(ref.get("filename"), str):
                    continue
                is_video = ref["filename"].lower().endswith((".mp4", ".webm", ".mkv", ".mov"))
                if is_video == video:
                    refs.append({k: ref[k] for k in ("filename", "subfolder", "type") if k in ref})
    return refs


class MediaService:
    def __init__(self, sup, client=None, *, root=None, poll_s=1.0, timeout_s=3600, now=time.time,
                 failure_window_s=90.0, backoff_max_s=5.0, fetch_retries=3, fetch_backoff_s=0.5):
        self.sup = sup
        self.client = client or sup.http
        self.files = MediaFiles(root or sup.s.data_dir / "media", now)
        self.poll_s, self.timeout_s, self.now = poll_s, timeout_s, now
        # A tunnel reconnect (status.py + colab sessions/usage + ssh handshake) or an engine_down blip takes 10-22 s
        # and more: consecutive poll errors are tolerated for this long, with capped exponential backoff.
        self.failure_window_s, self.backoff_max_s = failure_window_s, backoff_max_s
        self.fetch_retries, self.fetch_backoff_s = fetch_retries, fetch_backoff_s
        self.tasks: set[asyncio.Task] = set()
        self.files.prune()
        for path in self.files.root.glob("video_*.json"):
            with contextlib.suppress(MediaError):
                job = self.files.read(path.stem)
                if job.get("status") in ("queued", "in_progress"):
                    job.update(status="failed",
                               error="网关已重启；任务状态跟踪中断，请通过原始 ComfyUI history 检查结果")
                    self.files.write_json(job["id"], job)

        self.spawn(self._cleanup())

    async def _cleanup(self):
        while True:
            await asyncio.sleep(300)
            with contextlib.suppress(OSError):
                self.files.prune()

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        # A disconnected HTTP client must not leave an unobserved task exception.
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return task

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)

    def activity(self, target):
        return self.sup._activity.setdefault(target.sid, Activity())

    async def request(self, target, method, path, ua=USER_AGENTS["gateway"], **kwargs):
        try:
            response = await self.client.request(method, f"http://127.0.0.1:{target.port}{path}",
                                                 timeout=60, headers={"User-Agent": ua}, **kwargs)
            response.raise_for_status()
            if path != "/view":
                try:
                    response.json()
                except ValueError as exc:
                    raise MediaError("ComfyUI 返回了非 JSON 响应", 502) from exc
            return response
        except httpx.HTTPError as exc:
            raise MediaError(f"ComfyUI 请求失败：{type(exc).__name__}", 502) from exc

    async def upload(self, target, value, ua=USER_AGENTS["gateway"]):
        return await self.upload_bytes(target, *decode_image(value), ua=ua)

    async def upload_bytes(self, target, data, ext=".png", ctype="image/png", ua=USER_AGENTS["gateway"]):
        result = await self.request(target, "POST", "/upload/image", ua=ua, files={
            "image": (f"colabinet-{uuid.uuid4().hex}{ext}", data, ctype)}, data={"type": "input"})
        uploaded = result.json()
        return (uploaded.get("subfolder", "").rstrip("/") + "/" + uploaded["name"]).lstrip("/")

    async def submit(self, target, graph, ua=USER_AGENTS["gateway"]):
        result = (await self.request(target, "POST", "/prompt", ua=ua, json={
            "prompt": graph, "client_id": "colabinet"})).json()
        if result.get("error") or result.get("node_errors") or not result.get("prompt_id"):
            raise MediaError(f"ComfyUI 工作流被拒绝：{json.dumps(result, ensure_ascii=False)[:1500]}", 502)
        return result["prompt_id"]

    async def history(self, target, prompt_id, update=None):
        deadline = time.monotonic() + self.timeout_s
        failures, failing_since = 0, None
        while time.monotonic() < deadline:
            try:
                hist = await self._history_once(target, prompt_id)
                if hist is None:
                    # Empty history: the prompt must still be queued or running, otherwise it vanished (ComfyUI
                    # restarted, or the job was deleted/interrupted through the raw /comfy/ channel).
                    queue = (await self.request(target, "GET", "/queue", ua=POLL_UA)).json()
                    running = any(len(q) > 1 and q[1] == prompt_id for q in queue.get("queue_running", []))
                    pending = any(len(q) > 1 and q[1] == prompt_id for q in queue.get("queue_pending", []))
                    if not running and not pending:
                        # It may have finished between the two requests: look once more before giving up.
                        hist = await self._history_once(target, prompt_id)
                        if hist is None:
                            raise PromptVanished(
                                "ComfyUI 队列与历史中均未找到该任务（引擎可能已重启或任务已被取消）", 502)
                    elif update:
                        update("in_progress" if running else "queued", 50 if running else 0)
                failures, failing_since = 0, None
            except PromptVanished:
                raise
            except MediaError:
                failures += 1
                failing_since = failing_since or time.monotonic()
                if time.monotonic() - failing_since >= self.failure_window_s:
                    raise
                await asyncio.sleep(min(self.poll_s * 2 ** min(failures, 3), self.backoff_max_s))
                continue
            if hist:
                status = hist.get("status", {})
                if status.get("status_str") == "error":
                    message = json.dumps(status.get("messages", []), ensure_ascii=False)[-1500:]
                    raise MediaError(f"ComfyUI 渲染失败：{message}", 502)
                if status.get("completed") or status.get("status_str") == "success":
                    return hist
            await asyncio.sleep(self.poll_s)
        raise MediaError("ComfyUI 任务等待超时（任务可能仍在 VM 队列中；请检查原始 history）", 504)

    async def _history_once(self, target, prompt_id):
        return (await self.request(target, "GET", f"/history/{prompt_id}", ua=POLL_UA)).json().get(prompt_id) or None

    async def fetch(self, target, ref, ua=USER_AGENTS["gateway"]):
        # A finished render must survive a short tunnel reconnect: retry transport errors and 5xx a few times.
        for attempt in range(self.fetch_retries + 1):
            try:
                return await self._fetch_once(target, ref, ua)
            except httpx.HTTPError as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code >= 500
                if not retryable or attempt >= self.fetch_retries:
                    raise MediaError("无法下载 ComfyUI 输出文件", 502) from exc
                await asyncio.sleep(self.fetch_backoff_s * 2 ** attempt)

    async def _fetch_once(self, target, ref, ua):
        # Stream into a bounded buffer, never allow an arbitrary /view response to exhaust host RAM.
        url = f"http://127.0.0.1:{target.port}/view"
        async with self.client.stream("GET", url, params=ref, timeout=120, headers={"User-Agent": ua}) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_OUTPUT_BYTES:
                    raise MediaError("生成文件超过 512 MiB 限制", 502)
            ctype = response.headers.get("content-type", "").split(";")[0]
            return bytes(data), ctype or mimetypes.guess_type(ref["filename"])[0] or "application/octet-stream"

    async def render(self, target, graph, video=False, ua=USER_AGENTS["gateway"]):
        prompt_id = await self.submit(target, graph, ua)
        history = await self.history(target, prompt_id)
        refs = output_refs(history, video)
        if not refs:
            raise MediaError("ComfyUI 工作流完成但没有生成文件", 502)
        return await self.fetch(target, refs[0], ua)

    async def images(self, target, body: ImageRequest, origin="gateway"):
        ua = USER_AGENTS[origin]
        self.files.prune()
        if body.workflow == "h3edit" and not body.image:
            raise MediaError("H3 编辑需要 image 输入图像")
        if body.image and body.workflow != "h3edit":
            raise MediaError("image 输入图像只用于 h3edit 工作流")
        act, key = self.activity(target), -secrets.randbits(63)
        act.begin(key, self.sup.now())
        useful = True
        try:
            image_name = await self.upload(target, body.image, ua) if body.image else None
            seed = body.seed if body.seed is not None else secrets.randbits(63)
            result = []
            for i in range(body.n):
                current_seed = (seed + i) % 2**64
                data, ctype = await self.render(target, image_graph(body, current_seed, image_name), ua=ua)
                if body.upscale:
                    name = await self.upload_bytes(target, data, ".png", ctype, ua=ua)
                    data, ctype = await self.render(target, upscale_graph(current_seed, name), ua=ua)
                if body.response_format == "b64_json":
                    result.append({"b64_json": base64.b64encode(data).decode()})
                else:
                    ident = self.files.save(data, ctype)
                    result.append({"url": f"/v1/files/{ident}/content"})
            return {"created": int(self.now()), "data": result}
        except PromptVanished:
            useful = False  # nothing ran to completion: do not restart the idle clock
            raise
        finally:
            act.end(key, self.sup.now(), useful=useful)

    async def create_video(self, target, body: VideoRequest, origin="gateway"):
        ua = USER_AGENTS[origin]
        self.files.prune()
        ident = "video_" + uuid.uuid4().hex
        # Begin before the first VM request: a 32 MiB first-frame upload is not an inference path and the Comfy
        # queue is still empty, so without this the idle guard could stop the runtime mid-upload.
        act, key = self.activity(target), -secrets.randbits(63)
        act.begin(key, self.sup.now())
        try:
            available = (await self.request(target, "GET", "/models/loras", ua=POLL_UA)).json() if body.loras else []
            for lora in body.loras:
                if lora.name not in available:
                    raise MediaError(f"VM 上没有此 LoRA：{lora.name}")
            image_name = await self.upload(target, body.image, ua) if body.image else None
            seed = body.seed if body.seed is not None else secrets.randbits(63)
            prompt_id = await self.submit(target, video_graph(body, seed, image_name), ua)
            job = {"id": ident, "status": "queued", "progress": 0, "error": None,
                   "prompt_id": prompt_id, "expires_at": self.now() + TTL_S, "content_type": "video/mp4"}
            # Only a spawned _video owns the activity key; a failed status write (ENOSPC) must still release it.
            self.files.write_json(ident, job)
            self.spawn(self._video(target, job, act, key, ua))
        except BaseException:
            act.end(key, self.sup.now(), useful=False)
            raise
        return {"id": ident, "status": "queued"}

    async def _video(self, target, job, act, key, ua=USER_AGENTS["gateway"]):
        ident = job["id"]
        useful = True

        def update(status, progress):
            job.update(status=status, progress=progress)
            self.files.write_json(ident, job)
            act.touch(key, self.sup.now())

        try:
            history = await self.history(target, job["prompt_id"], update)
            refs = output_refs(history, video=True)
            if not refs:
                raise MediaError("ComfyUI 未返回视频输出", 502)
            data, ctype = await self.fetch(target, refs[0], ua)
            if not refs[0]["filename"].lower().endswith(".mp4"):
                raise MediaError("ComfyUI 没有返回 MP4 视频", 502)
            self.files.save(data, ctype, ident)
            job.update(status="completed", progress=100, content_type="video/mp4")
        except asyncio.CancelledError:
            job.update(status="failed", error="服务已关闭；原始任务可能仍在 VM 上执行")
            raise
        except PromptVanished as exc:
            useful = False
            job.update(status="failed", error=str(exc))
        except Exception as exc:
            job.update(status="failed", error=str(exc)[:1500])
        finally:
            # In-memory release first: a full disk (ENOSPC) on the status write must not leak the idle guard.
            act.end(key, self.sup.now(), useful=useful)
            with contextlib.suppress(OSError):
                self.files.write_json(ident, job)

    def video_status(self, ident):
        if not ident.startswith("video_"):
            raise MediaError("未找到视频任务", 404)
        job = self.files.read(ident)
        return {k: job[k] for k in ("id", "status", "progress", "error")}


def media_service(sup, client=None) -> MediaService:
    if not getattr(sup, "_media", None):
        sup._media = MediaService(sup, client)
    assert isinstance(sup._media, MediaService)
    return sup._media
