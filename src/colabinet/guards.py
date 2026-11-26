"""Cost guards as pure functions (spec 5.4). The supervisor acts on the decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .billing import session_cu

IDLE_STATES = {"tunnelling", "ready", "engine_down"}
NO_GUARD_STATES = {"stopping", "stopped", "failed"}
IDLE_WARN_S = 5 * 60
DAILY_WARN_FRACTION = 0.8
CAP_WARN_FRACTION = 0.8
# Past the per-session CU cap an in-flight inference request may finish for up to this long (billed at the session
# rate); after that the cap stops the VM even mid-request.
CAP_BUSY_GRACE_S = 30 * 60


@dataclass(frozen=True)
class Decision:
    action: str = "none"  # none | warn | stop
    reason: str | None = None
    message: str = ""


def cap_grace_cu(s: dict[str, Any]) -> float:
    return s["rate_cu_h"] * CAP_BUSY_GRACE_S / 3600


def evaluate(s: dict[str, Any], now: float, activity_last: float, grace_s: float, busy: bool = False) -> Decision:
    """One session. `activity_last` is Activity.effective(now) merged with the row's stored last_activity. `busy`: an
    inference request is in flight, which defers the CU-cap stop by at most CAP_BUSY_GRACE_S worth of CU."""
    state = s["state"]
    if state in NO_GUARD_STATES:
        return Decision()
    cap = s.get("cu_cap")
    if cap and session_cu(s, now) >= cap and not (busy and session_cu(s, now) < cap + cap_grace_cu(s)):
        return Decision("stop", "cu_cap", f"已达到 CU 上限（{cap:.2f} CU）")
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


def evaluate_cap(s: dict[str, Any], now: float, busy: bool = False) -> Decision:
    """Per-session CU cap warning (the stop itself is in evaluate()): from CAP_WARN_FRACTION of the cap, and while an
    in-flight request holds the session past the cap."""
    cap = s.get("cu_cap")
    if s["state"] in NO_GUARD_STATES or not cap:
        return Decision()
    cu = session_cu(s, now)
    rate = s["rate_cu_h"] or 0.0
    if cu >= cap:
        if not busy:
            return Decision()  # evaluate() stops it
        left_min = max(1, round((cap + cap_grace_cu(s) - cu) / rate * 60)) if rate else 0
        return Decision("warn", "cu_cap", f"已达到 CU 上限 {cap:.2f}，正在等待进行中的请求完成，最多再 {left_min} 分钟")
    if cu >= CAP_WARN_FRACTION * cap:
        left_min = max(1, round((cap - cu) / rate * 60)) if rate else 0
        return Decision("warn", "cu_cap", f"已用 {cu:.2f} / {cap:.2f} CU，按 {rate:.2f} CU/h 约 {left_min} 分钟后停止")
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
