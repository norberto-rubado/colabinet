"""Per-session CU cap: early warning, in-flight deferral, live extension."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from colabinet.activity import Activity
from colabinet.guards import CAP_BUSY_GRACE_S, evaluate, evaluate_cap
from colabinet.supervisor import CreateRefused
from tests.unit.test_presets_guards_alerts import sess
from tests.unit.test_review_r1_lifecycle import comfy_guard

H = 3600.0


def test_cap_warns_from_80_percent_with_eta():
    s = sess(cu_cap=10.0, rate_cu_h=10.0, idle_min=0)
    assert evaluate_cap(s, now=0.79 * H).action == "none"
    d = evaluate_cap(s, now=0.8 * H)
    assert (d.action, d.reason) == ("warn", "cu_cap") and "12 分钟后停止" in d.message
    assert evaluate_cap(sess(cu_cap=None), now=10 * H).action == "none"
    assert evaluate_cap(sess(cu_cap=10.0, state="stopped"), now=10 * H).action == "none"


def test_cap_stop_deferred_only_while_busy_and_within_grace():
    s = sess(cu_cap=10.0, rate_cu_h=10.0, idle_min=0)
    assert evaluate(s, now=H, activity_last=H, grace_s=600).reason == "cu_cap"  # idle at the cap: stop
    assert evaluate(s, now=H, activity_last=H, grace_s=600, busy=True).action == "none"
    w = evaluate_cap(s, now=H + 60, busy=True)
    assert w.action == "warn" and "进行中的请求" in w.message and "29 分钟" in w.message
    late = H + CAP_BUSY_GRACE_S
    assert evaluate(s, now=late, activity_last=late, grace_s=600, busy=True).reason == "cu_cap"


def cap_guard(tmp_path):
    sup, row, clock, client = comfy_guard(tmp_path, lambda r: httpx.Response(200, json={}))
    row.update(cu_cap=10.0, rate_cu_h=10.0, created_at=clock.t, idle_min=0)
    sup.store.current_run = AsyncMock(return_value={"engine": "llamacpp"})
    sup.store.session = AsyncMock(side_effect=lambda sid: dict(row))
    sup.account_id = None
    sup._own_account = lambda r: None
    return sup, row, clock, client


async def test_guard_raises_warning_then_defers_stop_for_inflight_request(tmp_path):
    sup, row, clock, client = cap_guard(tmp_path)
    async with client:
        clock.t += 0.85 * H
        await sup.guards_once()
        assert [a.kind for a in sup.alerts.list()] == ["cu_cap_warn"]
        assert sup.alerts.list()[0].actions == ["extend-cap"]
        sup._activity[1].begin(7, clock.t)
        clock.t += 0.2 * H  # past the cap, request still running
        await sup.guards_once()
        assert not sup.stop.called
        assert "进行中的请求" in sup.alerts.list()[0].message
        sup._activity[1].end(7, clock.t, useful=True)
        clock.t += 30
        await sup.guards_once()
        assert sup.stop.call_args.args == (1, "cu_cap")


async def test_extend_cap_from_current_spend_and_clears_warning(tmp_path):
    sup, row, clock, client = cap_guard(tmp_path)
    sup._update = AsyncMock(side_effect=lambda sid, **f: row.update(f))
    sup._event = AsyncMock()
    async with client:
        clock.t += 0.9 * H
        await sup.guards_once()
        assert sup.alerts.list()
        assert await sup.extend_cap(1, 10) == 20.0
        assert not sup.alerts.list()
        await sup.guards_once()
        assert not sup.alerts.list()  # 9 / 20 is under 80%
        clock.t += 1.5 * H  # 24 CU spent, cap 20: extending starts from the spend, not the stale cap
        assert await sup.extend_cap(1, 10) == 34.0
        with pytest.raises(CreateRefused):
            await sup.extend_cap(1, 0)
        row["state"] = "stopped"
        with pytest.raises(CreateRefused):
            await sup.extend_cap(1, 10)
