"""Regression tests for Gemini review round 4 (R4-01, R4-02): external-runtime stop decisions."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from colabinet.alerts import Alerts, NullNotifier
from colabinet.cli_parse import make_listing
from colabinet.colab_cli import CmdResult
from colabinet.config import REPO_ROOT, Clock, Settings
from colabinet.db import Database
from colabinet.presets import load_presets
from colabinet.proxy import ProxyManager
from colabinet.store import Store
from colabinet.supervisor import CreateRequest, Supervisor

L4 = "[managed-l4] gpu-l4-1 | Hardware: L4 | Shape: Standard | Variant: GPU\n"
G4 = "[ext-g4] gpu-g4-ep1 | Hardware: G4 | Shape: Standard | Variant: GPU\n"
TWO = "Current balance: 200.00 compute units\nUsage rate: 10.44/hr\nActive assignments: 2\n"
NONE = "[colab] No active sessions found on server.\n"
ZERO = "Current balance: 200.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n"
ONE = "Current balance: 200.00 compute units\nUsage rate: 8.90/hr\nActive assignments: 1\n"


class CLI:
    enabled = True

    def __init__(self, clock):
        self.clock, self.sessions_out, self.usage_out = clock, "", ""
        self.stops: list[str] = []
        self.stop_rc = 0

    def command(self):
        return ["colab"]

    async def listing(self):
        return make_listing(0, self.sessions_out, 0, self.usage_out, self.clock.now())

    async def exec_code(self, name, code, timeout=None):
        return CmdResult(0, "1\n")

    async def stop(self, name):
        self.stops.append(name)
        return CmdResult(self.stop_rc, "boom" if self.stop_rc else "[colab] Session terminated.")

    async def spawn_new(self, name, gpu_class):
        fut = asyncio.get_running_loop().create_future()
        return SimpleNamespace(pid=99999, returncode=None, communicate=lambda: fut)


@pytest.fixture
async def world(tmp_path):
    base = Clock()
    clock = SimpleNamespace(t=1000.0, day_bounds=base.day_bounds, day_key=base.day_key)
    clock.now = lambda: clock.t
    db = Database(tmp_path / "r4.db")
    db.init()
    store, cli, notifier = Store(db), CLI(clock), NullNotifier()
    proxies = Mock(bound=Mock(return_value=True), ports=Mock(return_value=set()), unbind=AsyncMock())
    sup = Supervisor(settings=Settings(data_dir=tmp_path, daily_cap_cu=0), store=store, cli=cli,
                     tunnels=Mock(stop=AsyncMock()), proxies=proxies or ProxyManager(),
                     alerts=Alerts(notifier, clock.now), clock=clock,
                     presets=load_presets(REPO_ROOT / "presets.toml"), http=object())
    sup._start_driver = Mock()
    yield SimpleNamespace(sup=sup, store=store, cli=cli, clock=clock, notifier=notifier)
    for t in list(sup._new.values()):
        t.cancel()
    await asyncio.gather(*sup._new.values(), return_exceptions=True)
    db.close()


async def managed_l4(store):
    await store.insert_session(
        account_id="default", colab_name="managed-l4", endpoint="gpu-l4-1", gpu_requested="L4", gpu_class="L4",
        proxy_port=3080, hidden_port=23000, vm_port=8090, state="ready", state_since=1000.0, created_at=1000.0,
        ready_at=1000.0, last_activity=1000.0, rate_cu_h=1.54, rate_source="measured", idle_min=30)


def test_r4_01_listing_complete_requires_all_billed_runtimes():
    """R4-01: a listing is `complete` (absence proves a VM is gone) only when it lists as many runtimes as usage
    bills; a partial `colab sessions` answer stays trusted for what it lists."""
    partial = make_listing(0, L4, 0, TWO, 1.0)
    assert partial.trusted and not partial.complete
    assert make_listing(0, L4 + G4, 0, TWO, 1.0).complete
    assert make_listing(0, NONE, 0, ZERO, 1.0).complete
    assert not make_listing(0, L4 + G4, 1, "", 1.0).complete  # usage failed -> never complete


async def test_r4_01_partial_listing_does_not_disarm_the_auto_stop_timer(world):
    """R4-01: when the timer fires and `colab sessions` omits the external VM while usage still bills it, colabinet
    must not mark it already_gone: it stops it by name and the keepalive record stays until that succeeds."""
    w = world
    await managed_l4(w.store)
    w.cli.sessions_out, w.cli.usage_out = L4 + G4, TWO
    await w.sup.reconcile_once()
    await w.sup.external_keepalive("gpu-g4-ep1", True, stop_after_min=5)
    w.cli.sessions_out = L4  # partial: 1 of 2 billed runtimes listed
    w.clock.t += 301.0
    await w.sup.keepalive_externals_once()
    events = await w.store.db.fetchall("SELECT detail_json FROM events WHERE kind = 'external_stopped'")
    assert not any(json.loads(e["detail_json"]).get("reason") == "already_gone" for e in events)
    row = await w.store.external("gpu-g4-ep1")
    assert row["keepalive"] == 1 and row["stop_after_min"] == 5  # still guarded; retried next tick
    w.cli.sessions_out = L4 + G4  # the listing recovers -> the next tick really stops it
    w.clock.t += 15.0
    await w.sup.keepalive_externals_once()
    assert w.cli.stops == ["ext-g4"]
    assert (await w.store.external("gpu-g4-ep1"))["keepalive"] == 0


async def test_r4_01_failed_stop_with_partial_after_listing_keeps_the_guard(world):
    """R4-01: `colab stop` fails and the follow-up listing is partial: that is not proof the VM is gone, so the
    keepalive stays, the auto-stop alert is raised and the stop is retried."""
    w = world
    await managed_l4(w.store)
    w.cli.sessions_out, w.cli.usage_out = L4 + G4, TWO
    await w.sup.reconcile_once()
    await w.sup.external_keepalive("gpu-g4-ep1", True, stop_after_min=5)
    w.cli.stop_rc = 1
    calls = 0

    async def full_then_partial():
        nonlocal calls
        calls += 1
        return make_listing(0, L4 + G4 if calls == 1 else L4, 0, TWO, w.clock.now())

    w.cli.listing = full_then_partial
    w.clock.t += 301.0
    await w.sup.keepalive_externals_once()
    assert w.cli.stops == ["ext-g4"]
    assert (await w.store.external("gpu-g4-ep1"))["keepalive"] == 1
    assert any(a.key == "external_autostop:gpu-g4-ep1" for a in w.sup.alerts.list())


async def test_r4_02_stopped_external_whose_name_is_reused_is_already_gone(world):
    """R4-02: the external VM was stopped elsewhere and its name reused by a new managed session before the next
    reconcile: the timer records already_gone instead of a CRITICAL 'is a managed session' auto-stop failure."""
    w = world
    w.cli.sessions_out, w.cli.usage_out = "[work] gpu-ep1 | Hardware: G4 | Shape: Standard | Variant: GPU\n", ONE
    await w.sup.reconcile_once()
    await w.sup.external_keepalive("gpu-ep1", True, stop_after_min=5)
    w.notifier.sent.clear()
    w.cli.sessions_out, w.cli.usage_out = NONE, ZERO
    await w.sup.create(CreateRequest(name="work", gpu_class="G4", preset_id="comfy-image-uncensored", confirm=True))
    w.clock.t += 301.0
    await w.sup.keepalive_externals_once()
    assert not any(a.key == "external_autostop:gpu-ep1" for a in w.sup.alerts.list())
    assert w.notifier.sent == [] and w.cli.stops == []
    events = await w.store.db.fetchall("SELECT detail_json FROM events WHERE kind = 'external_stopped'")
    assert [json.loads(e["detail_json"])["reason"] for e in events] == ["already_gone"]


async def test_r4_02_unnamed_external_whose_name_is_reused_gets_the_unnamed_warning(world):
    """R4-02: a kept-alive external VM lost its CLI name and the freed name went to a new managed session: the
    timer raises the non-critical external_unnamed warning, not a CRITICAL external_autostop alert."""
    w = world
    w.cli.sessions_out, w.cli.usage_out = "[work2] gpu-ep2 | Hardware: G4 | Shape: Standard | Variant: GPU\n", ONE
    await w.sup.reconcile_once()
    await w.sup.external_keepalive("gpu-ep2", True, stop_after_min=5)
    w.notifier.sent.clear()
    w.cli.sessions_out = "[?] gpu-ep2 | Hardware: G4 | Shape: Standard | Variant: GPU\n"
    await w.sup.create(CreateRequest(name="work2", gpu_class="G4", preset_id="comfy-image-uncensored", confirm=True))
    w.clock.t += 301.0
    await w.sup.keepalive_externals_once()
    keys = {a.key: a for a in w.sup.alerts.list()}
    assert "external_autostop:gpu-ep2" not in keys
    assert "external_unnamed:gpu-ep2" in keys and keys["external_unnamed:gpu-ep2"].level != "critical"
    assert w.cli.stops == []


async def test_r4_02_manual_stop_of_a_managed_name_is_still_refused(world):
    """R4-02 guard: moving name_in_use after the gone/unnamed checks must not let a manual external stop hit a
    live managed session."""
    from colabinet.supervisor import CreateRefused
    w = world
    await managed_l4(w.store)
    w.cli.sessions_out, w.cli.usage_out = L4 + G4, TWO
    await w.sup.reconcile_once()
    with pytest.raises(CreateRefused, match="托管会话"):
        await w.sup.stop_external("managed-l4")
    assert w.cli.stops == []
