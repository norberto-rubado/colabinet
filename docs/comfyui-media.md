# ComfyUI 媒体引擎与 API

## 运行与预设

- `comfy-image-uncensored`：MOSS、Kroma、Krea2 Turbo、H3-Image、Anima、WAI、SeedVR2，共 16 文件，132.648 GB。
- `comfy-video-h3`：MiniMax H3 FL2VA int8、Qwen3VL NVFP4 AWQ、视频/音频 VAE、Turbo-8 LoRA、6 个可选 NSFW LoRA，共 11 文件，44.067 GB。
- 目前仅允许 G4。其他 GPU 在表单中禁用，API 创建/重配也在分配 VM 前拒绝。
- 安装脚本沿用系统 Python，不创建另一个 torch 环境；为当前 torch/torchvision/torchaudio 写精确约束，pip 不能降级或替换 CUDA 栈。
- 8 并发 `hf download`、Xet high-performance、4 次指数退避；继承 supervisor 的 HF_TOKEN。模型按仓库下载一次，软链接进 ComfyUI/models。
- ComfyUI 固定到 `afcb984e76fe9f870bf7746cf394766e1f690563`。只监听 `127.0.0.1:$VM_PORT`，健康检查为 `/system_stats`。
- 数据目录：`<data_dir>/media`，目录权限 0700，文件 0600。图片 URL 和视频任务/文件保留 24 小时，访问时立即检查过期，每 5 分钟清理磁盘。
- 服务重启不重新提交推理。未完成的视频任务标记 failed，保留 prompt_id，可通过原始 history 找回 VM 结果。

## 统一网关认证

全部路由使用现有 gateway.key：`Authorization: Bearer KEY` 或 `x-api-key: KEY`。
`model` 精确匹配媒体预设别名或会话名；不采用 LLM 的单会话兜底，以免拼错模型导致误路由。
`GET /v1/models` 包含媒体会话，增加 `kind: "text" | "image" | "video"`。

## 生图

`POST /v1/images/generations`

```json
{
  "model": "comfy-image-uncensored",
  "prompt": "夕阳下的山间小屋",
  "workflow": "moss",
  "size": "1024x1024",
  "n": 1,
  "response_format": "b64_json",
  "negative_prompt": "blur",
  "seed": 42,
  "steps": 25,
  "upscale": false
}
```

必填 model、prompt；其他字段可省略。默认 workflow=moss、size=1024x1024、n=1、response_format=b64_json、随机 seed、各工作流默认步数。
workflow：moss、kroma、krea2turbo、h3、h3edit、anima、wai。
`h3edit` 必须加 `image`（纯 base64 或 `data:image/png;base64,...`），其余工作流不接受 image。
输入图像支持 PNG/JPEG/WebP，最多 32 MiB。H3 编辑沿用已验证图的 width=height=0，保留输入画布；size 不覆盖编辑画布。
`upscale=true` 在生成后通过 SeedVR2 放大 ×2。
`n` 为 1–8，顺序生成；第 i 张使用 `(seed+i) % 2**64`。
尺寸为 256–4096 的 32 倍数。steps 为 1–100。

返回（JSON 前可能有用于保持 Cloudflare 连接的空格）：

```json
{"created": 1795220000, "data": [{"b64_json": "..."}]}
```

或者 response_format=url：

```json
{"created": 1795220000, "data": [{"url": "https://PUBLIC_BASE/v1/files/file_HEX/content"}]}
```

`GET /v1/files/{id}/content` 需要同一认证，返回图片；支持 Range。URL 从 public_api_base 构建，不信任 Host。
冷启动超过 20 秒时先发 200 和 JSON 空白，每 15 秒保活，之后的错误只能在 JSON error 对象中返回。
请求断开不取消已提交的 ComfyUI 任务，后台继续跟踪到 history 完成并维持活动计时。

## 生视频（异步）

`POST /v1/videos`

```json
{
  "model": "comfy-video-h3",
  "prompt": "A stream flowing over rocks. Audio: water and birds. No cuts.",
  "seconds": 5,
  "size": "768x1344",
  "loras": [{"name": "HMNSFW-AIO-V2.5.safetensors", "strength": 1.0}],
  "seed": 42,
  "steps": 8
}
```

必填 model、prompt。image 可选（base64/data URL 首帧）；无 image 使用同一个 MiniMaxH3ImageToVideo 节点，但不连接其可选 first_frame，即文生视频。
默认 seconds=5、size=768x1344、steps=8、loras=[]、随机种子。
seconds 为 4–15；帧数 `ceil(seconds*24)` 向上取到 17k+5（4 秒=107、5 秒=124、6 秒=158、15 秒=362），最终视频可能略长于请求。
视频尺寸是 256–2048 的 32 倍数，面积最多 2097152 像素。支持 768x1344、1344x768、1024x1024 等。
LoRA 名称必须存在于 VM 的 `/models/loras`；最多 16 个，强度 -4–4，按请求顺序串联在 Turbo LoRA 后。Turbo LoRA 始终启用。

提交返回：

```json
{"id": "video_HEX", "status": "queued"}
```

`GET /v1/videos/{id}`：

```json
{"id": "video_HEX", "status": "in_progress", "progress": 50, "error": null}
```

状态 queued|in_progress|completed|failed。progress 是队列阶段估计：排队 0、运行 50、完成 100，不是采样进度。
失败 error 为字符串。`GET /v1/videos/{id}/content` 完成后返回有声 MP4（H.264），未完成返回 409；所有路由需要认证。
默认等待任务上限 1 小时；隧道历史查询短暂失败会重试 10 次，不重复 POST /prompt。

## 原始 API

任意 HTTP 方法 `/comfy/{model}/{path}` 透传到目标会话的 ComfyUI，例如：

- POST `/comfy/comfy-image-uncensored/prompt`
- GET `/comfy/comfy-image-uncensored/history/PROMPT_ID`
- GET `/comfy/comfy-image-uncensored/view?filename=...&subfolder=...&type=output`
- POST `/comfy/comfy-image-uncensored/upload/image`（multipart）
- GET `/comfy/comfy-video-h3/object_info`

原始字节保持不变，不插入 JSON 保活空格；网关认证信息不会转发给 VM。原始 HTTP 通道不代理 WebSocket。
代理将成功的 POST /prompt 计为活动；生图/视频任务只有成功交付结果才刷新活动时间，400/502、任务消失、保存失败或取消都不算。supervisor 每次闲置判定前查询 /queue：有运行或排队任务只推迟闲置停止、不写入活动时间，且最多推迟到最后一次真实活动后 6 小时（引擎抖动或服务重启不会重新计时）。查询失败不视为已闲置，但只在最后一次成功查询（或就绪时刻）后 10 分钟内推迟；健康恢复、每日/CU/截止时间保护仍生效。

## 图与实测来源

图片图来自本机已验证的 `comfy_bench.py`：g_moss、g_krea、g_h3、g_h3edit、g_anima、g_wai、g_seedvr。
视频图来自 crhywun/h3-shortdrama 的 workflow_i2v.json（16 节点）；SaveVideo 更新为固定 ComfyUI 版本的动态 format/codec 结构，指定 MP4/H.264。
ComfyUI 源码确认 first_frame 可省略、SaveVideo 在 history 的 images 字段返回视频。
27 个 HF 文件在 2026-11-21 逐一 HEAD 验证成功并取得 x-linked-size；没有下载模型权重或启动 GPU。
图片速度是此前 G4 实测；视频 80 秒/5 秒片段、约 39 GB VRAM 为第三方测量，待负责人 G4 复测。

## 验证

```sh
env -u PYTHONPATH uv run --frozen pytest -q tests/unit
```

新增测试覆盖清单验证、安装环境/约束/重试、图连接/帧数/音频/LoRA 链、认证、prompt→history→view、持久任务、过期清理、真实本机会话代理、断线活动跟踪、原始二进制透传，以及浏览器生图/视频页签与 CSRF 请求。
本机 integration CDP 已知不可用，不执行 tests/integration。未启动任何 Colab VM。

单元测试基线：562 通过、1 个既有失败。完成后：653 通过、同一个既有失败（两个原版 LLM 预设待补 measured，不修改）。新增 91 项测试；Ruff 新模块/新增测试检查、JS 语法检查、git diff --check 均通过。
