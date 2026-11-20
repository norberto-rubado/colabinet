"""Cost guards as pure functions (spec 5.4). The supervisor acts on the decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .billing import session_cu

IDLE_STATES = {"tunnelling", "ready", "engine_down"}
NO_GUARD_STATES = {"stopping", "stopped", "failed"}
IDLE_WARN_S = 5 * 60
DAILY_WARN_FRACTION = 0.8


@dataclass(frozen=True)
class Decision:
    action: str = "none"  # none | warn | stop
    reason: str | None = None
    message: str = ""


def evaluate(s: dict[str, Any], now: float, activity_last: float, grace_s: float) -> Decision:
    """One session. `activity_last` is Activity.effective(now) merged with the row's stored last_activity."""
    state = s["state"]
    if state in NO_GUARD_STATES:
        return Decision()
    if s.get("cu_cap") and session_cu(s, now) >= s["cu_cap"]:
        return Decision("stop", "cu_cap", f"已达到 CU 上限（{s['cu_cap']:.2f} CU）")
    if s.get("deadline") and now >= s["deadline"]:
        return Decision("stop", "deadline", "已到定时停止时间")
    if state == "setup_failed" and now - s["state_since"] >= grace_s:
        return Decision("stop", "setup_failed", "设置失败且无人重试")
    if state == "gpu_mismatch" and now - s["state_since"] >= grace_s:
        return Decision("stop", "gpu_mismatch", "分配到的 GPU 不符且无人选择继续")
    idle_min = s.get("idle_min") or 0
    if state in IDLE_STATES and idle_min > 0:
        last = max(activity_last, s.get("last_activity") or 0.0)
        idle = now - last
        limit = idle_min * 60
        if idle >= limit:
            return Decision("stop", "idle", f"已闲置 {idle_min} 分钟")
        if idle >= limit - IDLE_WARN_S:
            return Decision("warn", "idle", f"{max(1, round((limit - idle) / 60))} 分钟后闲置自动停止")
    return Decision()


def evaluate_daily(spend_today: float, cap: float) -> Decision:
    """Account-wide daily CU cap. cap <= 0 means off."""
    if cap <= 0:
        return Decision()
    if spend_today >= cap:
        return Decision("stop", "daily_cap", f"今日消耗 {spend_today:.2f} CU，已达到每日上限 {cap:.2f} CU")
    if spend_today >= DAILY_WARN_FRACTION * cap:
        return Decision("warn", "daily_cap", f"今日消耗 {spend_today:.2f} CU，已超过每日上限 {cap:.2f} CU 的 80%")
    return Decision()
