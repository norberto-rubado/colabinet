"""外部运行时: runtimes in `colab sessions` that colabinet did not create — snapshot, first_seen persistence, cost
estimate, 接管保活 heartbeat and auto-stop (fake clock + in-process fake CLI), and the /ui/external card."""

import json
import sqlite3

import httpx
import pytest

from colabinet.alerts import Alerts, NullNotifier
from colabinet.cli_parse import Listing, Runtime, Usage, rate_class
from colabinet.colab_cli import CmdResult, ColabCLI
from colabinet.config import SEED_RATES, Settings
from colabinet.db import SCHEMA_V1, SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_VERSION, Database
from colabinet.presets import PresetError
from colabinet.proxy import ProxyManager
from colabinet.store import Store
from colabinet.supervisor import EXTERNAL_BEAT_CODE, EXTERNAL_BEAT_S, Supervisor, external_cu
from tests.fakes import FakeColab

CSRF, TOKEN = "ext-csrf", "ext-token"


class FakeCLI:
    """In-process stand-in for ColabCLI: a mutable listing, recorded exec/stop calls."""

    enabled = True

    def __init__(self, clock, runtimes=()):
        self.clock = clock
        self.runtimes = list(runtimes)
        self.execs: list[tuple[str, str]] = []
        self.stops: list[str] = []
        self.exec_ok = True

    def command(self):
        return ["colab"]

    async def listing(self):
        rate = sum(SEED_RATES.get(rate_class(r.hardware, r.shape) or "", (0, ""))[0] for r in self.runtimes)
        return Listing(list(self.runtimes), Usage(500.0, rate, len(self.runtimes)), True, self.clock.now())

    async def usage(self):
        return (await self.listing()).usage

    async def version(self):
        return "0.8.0"

    async def exec_code(self, name, code, timeout=None):
        self.execs.append((name, code))
        return CmdResult(0, "1\n") if self.exec_ok else CmdResult(1, "[colab] Session 'x' appears to be lost")

    async def stop(self, name):
        self.stops.append(name)
        self.runtimes = [r for r in self.runtimes if r.name != name]
        return CmdResult(0, "[colab] Session terminated.")


def g4(name="g4a", endpoint="gpu-g4-s-1"):
    return Runtime(name, endpoint, "G4", "Standard", "GPU")


def make_sup(store, clock, cli, tmp_path):
    return Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=cli, tunnels=None,
                      proxies=ProxyManager(), alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                      presets=PresetError("unused"), http=object())


@pytest.fixture
def cli(clock):
    return FakeCLI(clock, [g4(), Runtime(None, "gpu-a100-s-2", "A100", "High-RAM", "GPU")])


@pytest.fixture
def sup(store, clock, cli, tmp_path):
    return make_sup(store, clock, cli, tmp_path)


def toast_title(response):
    return json.loads(response.headers["HX-Trigger"])["cb-toast"]["title"]


async def events(store, kind):
    return await store.db.fetchall("SELECT * FROM events WHERE kind = ? ORDER BY id", (kind,))


# ------------------------------------------------------------------------------------------------- snapshot
def test_rate_class_maps_listing_hardware_and_shape():
    assert rate_class("G4", "Standard") == "G4"
    assert rate_class("A100", "High-RAM") == "A100-80" and rate_class("A100", "Standard") == "A100"
    assert rate_class("T4", "High-RAM") == "T4-HIGHMEM" and rate_class("CPU", "Standard") == "CPU"
    assert rate_class("H100", "Standard") is None and rate_class("V5E1", "") is None


async def test_snapshot_built_from_fake_listing_skips_managed_runtimes(sup, store, cli, clock):
    sid = await store.insert_session(colab_name="ours", endpoint="gpu-l4-s-9", gpu_requested="L4", proxy_port=3080,
                                     hidden_port=23000, vm_port=8090, state="ready", state_since=1.0,
                                     created_at=clock.now(), rate_cu_h=1.54, rate_source="measured", idle_min=30)
    cli.runtimes.append(Runtime("ours", "gpu-l4-s-9", "L4", "Standard", "GPU"))
    await sup.reconcile_once()
    state = await sup.state()
    ext = {e["endpoint"]: e for e in state["externals"]}
    assert set(ext) == {"gpu-g4-s-1", "gpu-a100-s-2"} and sid
    named = ext["gpu-g4-s-1"]
    assert (named["name"], named["hardware"], named["shape"], named["rate_class"]) == ("g4a", "G4", "Standard", "G4")
    assert named["first_seen"] == clock.now() and named["rate_cu_h"] == SEED_RATES["G4"][0]
    assert named["cu_spent"] == 0 and named["keepalive"] is False
    unnamed = ext["gpu-a100-s-2"]
    assert unnamed["name"] is None and unnamed["rate_class"] == "A100-80"
    # the existing alerts stay, and now point to the card
    alerts = {a["key"]: a for a in state["alerts"]}
    assert "见外部运行时" in alerts["external:gpu-g4-s-1"]["message"]
    assert alerts["external:gpu-g4-s-1"]["actions"] == ["stop-external"]
    assert "见外部运行时" in alerts["external:gpu-a100-s-2"]["message"]


async def test_runtime_gone_from_listing_leaves_snapshot_and_alerts(sup, cli):
    await sup.reconcile_once()
    cli.runtimes = []
    await sup.reconcile_once()
    assert (await sup.state())["externals"] == []
    assert not [a for a in sup.alerts.list() if a.key.startswith("external:")]


# ------------------------------------------------------------------------------------- first_seen persistence
async def test_first_seen_survives_supervisor_restart(tmp_path, clock):
    db = Database(tmp_path / "persist.db")
    db.init()
    cli = FakeCLI(clock, [g4()])
    first = make_sup(Store(db), clock, cli, tmp_path)
    await first.reconcile_once()
    t0 = clock.now()
    db.close()

    clock.advance(1800)
    db = Database(tmp_path / "persist.db")
    db.init()
    try:
        second = make_sup(Store(db), clock, cli, tmp_path)
        assert (await second.state())["externals"] == []  # nothing until a reconcile lists it
        await second.reconcile_once()
        row = (await second.state())["externals"][0]
        assert row["first_seen"] == t0 and row["last_seen"] == clock.now()
        assert row["up_s"] == 1800
        assert row["cu_spent"] == pytest.approx(SEED_RATES["G4"][0] * 0.5, abs=1e-3)
    finally:
        db.close()


async def test_schema_v5_migrates_a_v4_database(tmp_path):
    path = tmp_path / "v4.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA_V1 + SCHEMA_V2 + SCHEMA_V3 + SCHEMA_V4)
        connection.execute("PRAGMA user_version = 4")
    db = Database(path)
    db.init()
    try:
        assert SCHEMA_VERSION == 5
        assert (await db.fetchone("PRAGMA user_version"))["user_version"] == 5
        cols = {row["name"] for row in await db.fetchall("PRAGMA table_info(external_runtimes)")}
        assert {"endpoint", "name", "hardware", "first_seen", "last_seen", "keepalive", "keepalive_since",
                "stop_after_min"} <= cols
    finally:
        db.close()


async def test_store_keeps_first_seen_and_prunes_old_rows(store):
    await store.see_externals([("ep", "a", "G4")], 100.0)
    await store.see_externals([("ep", "b", "G4")], 200.0)
    row = await store.external("ep")
    assert (row["first_seen"], row["last_seen"], row["name"]) == (100.0, 200.0, "b")
    await store.prune_externals(150.0)
    assert await store.external("ep") is not None
    await store.prune_externals(250.0)
    assert await store.external("ep") is None


# ------------------------------------------------------------------------------------------------ cost math
def test_external_cost_estimate_math():
    assert external_cu(8.9, 1000.0, 1000.0 + 3600) == pytest.approx(8.9)
    assert external_cu(8.9, 1000.0, 1000.0 + 900) == pytest.approx(2.225)
    assert external_cu(5.3, 2000.0, 1000.0) == 0.0  # clock skew never gives a negative cost


async def test_cost_uses_learned_rate_table(sup, store, clock):
    await store.set_rate("G4", 10.05, "learned", clock.now())
    await sup.reconcile_once()
    clock.advance(1200)
    row = next(e for e in await sup.externals() if e["name"] == "g4a")
    assert row["rate_cu_h"] == 10.05 and row["rate_source"] == "learned"
    assert row["cu_spent"] == pytest.approx(10.05 / 3, abs=1e-3) and row["up_s"] == 1200


async def test_unknown_hardware_has_no_rate(sup, cli):
    cli.runtimes = [Runtime("h", "gpu-h100-s-3", "H100", "Standard", "GPU")]
    await sup.reconcile_once()
    row = (await sup.externals())[0]
    assert row["rate_cu_h"] is None and row["cu_spent"] is None


# ------------------------------------------------------------------------------------------------- keepalive
async def test_keepalive_heartbeats_every_60s_with_fake_clock(sup, store, cli, clock):
    await sup.reconcile_once()
    await sup.keepalive_externals_once()
    assert cli.execs == []  # nothing until 接管保活 is on
    row = await sup.external_keepalive("gpu-g4-s-1", True, 0)
    assert row["keepalive"] and row["stop_after_min"] == 0 and row["stop_at"] is None
    await sup.keepalive_externals_once()
    assert cli.execs == [("g4a", EXTERNAL_BEAT_CODE)] and "print(1)" in EXTERNAL_BEAT_CODE
    clock.advance(30)
    await sup.keepalive_externals_once()
    assert len(cli.execs) == 1  # not due yet
    clock.advance(EXTERNAL_BEAT_S - 30)
    await sup.keepalive_externals_once()
    assert len(cli.execs) == 2
    assert (await sup.externals())[0]["last_beat_ok"] is True
    # off: no more beats, persisted flag cleared
    await sup.external_keepalive("gpu-g4-s-1", False)
    clock.advance(EXTERNAL_BEAT_S * 3)
    await sup.keepalive_externals_once()
    assert len(cli.execs) == 2 and not (await store.external("gpu-g4-s-1"))["keepalive"]
    kinds = [e["detail_json"] for e in await events(store, "external_keepalive")]
    assert len(kinds) == 2 and '"enabled": true' in kinds[0] and '"enabled": false' in kinds[1]


async def test_keepalive_default_stop_after_comes_from_settings(sup, store):
    await sup.update_settings(idle_min_default=45)
    await sup.reconcile_once()
    row = await sup.external_keepalive("gpu-g4-s-1", True)
    assert row["stop_after_min"] == 45 and row["stop_at"] == row["keepalive_since"] + 45 * 60
    assert (await store.external("gpu-g4-s-1"))["stop_after_min"] == 45


async def test_keepalive_flag_survives_restart(tmp_path, clock):
    db = Database(tmp_path / "ka.db")
    db.init()
    cli = FakeCLI(clock, [g4()])
    first = make_sup(Store(db), clock, cli, tmp_path)
    await first.reconcile_once()
    await first.external_keepalive("gpu-g4-s-1", True, 0)
    db.close()
    db = Database(tmp_path / "ka.db")
    db.init()
    try:
        second = make_sup(Store(db), clock, cli, tmp_path)
        await second.reconcile_once()
        await second.keepalive_externals_once()
        assert cli.execs == [("g4a", EXTERNAL_BEAT_CODE)]
    finally:
        db.close()


async def test_unnamed_runtime_cannot_be_kept_alive(sup):
    from colabinet.supervisor import CreateRefused

    await sup.reconcile_once()
    with pytest.raises(CreateRefused):
        await sup.external_keepalive("gpu-a100-s-2", True, 0)
    with pytest.raises(CreateRefused):
        await sup.external_keepalive("not-listed", True, 0)


async def test_failed_heartbeats_raise_a_warning(sup, cli, clock):
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-s-1", True, 0)
    cli.exec_ok = False
    for _ in range(3):
        await sup.keepalive_externals_once()
        clock.advance(EXTERNAL_BEAT_S)
    alert = next(a for a in sup.alerts.list() if a.key == "external_keepalive:gpu-g4-s-1")
    assert "见外部运行时" in alert.message and "3 次" in alert.message
    cli.exec_ok = True
    await sup.keepalive_externals_once()
    assert not [a for a in sup.alerts.list() if a.key == "external_keepalive:gpu-g4-s-1"]


async def test_auto_stop_after_n_minutes_goes_through_stop_external(sup, store, cli, clock):
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-s-1", True, 10)
    calls = []
    real = sup.stop_external

    async def spy(name, expected_account=None, **kw):
        calls.append((name, kw))
        return await real(name, expected_account, **kw)

    sup.stop_external = spy
    clock.advance(9 * 60)
    await sup.keepalive_externals_once()
    assert cli.stops == [] and len(cli.execs) == 1
    clock.advance(60)
    await sup.keepalive_externals_once()
    assert calls == [("g4a", {"endpoint": "gpu-g4-s-1", "reason": "keepalive_timer"})]
    assert cli.stops == ["g4a"] and len(cli.execs) == 1  # stopped, not beaten
    stopped = await events(store, "external_stopped")
    assert len(stopped) == 1 and '"reason": "keepalive_timer"' in stopped[0]["detail_json"]
    assert not (await store.external("gpu-g4-s-1"))["keepalive"]
    assert [e["name"] for e in await sup.externals()] == [None]


async def test_auto_stop_refuses_a_recreated_runtime_with_the_same_name(sup, cli, clock):
    await sup.reconcile_once()
    await sup.external_keepalive("gpu-g4-s-1", True, 1)
    cli.runtimes = [g4(endpoint="gpu-g4-s-77"), *cli.runtimes[1:]]  # same name, another VM, not yet reconciled
    clock.advance(120)
    await sup.keepalive_externals_once()
    assert cli.stops == []
    assert any(a.key == "external_autostop:gpu-g4-s-1" for a in sup.alerts.list())  # own key since R2-03


async def test_heartbeat_uses_the_cli_exec_helper(tmp_path):
    fake = FakeColab(tmp_path)
    try:
        cli = ColabCLI(str(fake.bin), now=lambda: 1.0)
        proc = await cli.spawn_new("g4a", "G4")
        await proc.communicate()
        res = await cli.exec_code("g4a", EXTERNAL_BEAT_CODE)
        assert res.ok and res.out.strip() == "1"
        assert fake.read()["calls"][-1][:3] == ["exec", "-s", "g4a"]
        assert not (await cli.exec_code("missing", EXTERNAL_BEAT_CODE)).ok
    finally:
        fake.kill_all()


# ------------------------------------------------------------------------------------------------------- UI
@pytest.fixture
async def client(sup):
    from colabinet.api import create_app

    app = create_app(sup, TOKEN, 7860, csrf=CSRF)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as c:
        yield c


async def test_external_partial_renders_table(sup, client, clock):
    await sup.reconcile_once()
    clock.advance(3600)
    body = (await client.get("/ui/external")).text
    for header in ("名称", "硬件", "已运行", "估算费率", "估算已花费", "操作"):
        assert f"<th" in body and header in body
    assert "g4a" in body and "G4" in body and "1 小时 00 分" in body
    assert f"{SEED_RATES['G4'][0]:.2f} CU/h" in body and f"≈ {SEED_RATES['G4'][0]:.2f} CU" in body
    assert "接管保活" in body and 'hx-post="/ui/actions/external/g4a/stop"' in body
    assert "hx-confirm=\"要停止外部运行时 g4a" in body
    assert "（未命名）" in body and "请在 Colab 网页中停止" in body


async def test_external_partial_empty_state(sup, cli, client):
    cli.runtimes = []
    await sup.reconcile_once()
    assert "没有外部运行时" in (await client.get("/ui/external")).text


async def test_overview_and_sessions_pages_include_the_card(sup, client):
    await sup.reconcile_once()
    for path in ("/", "/sessions"):
        body = (await client.get(path)).text
        assert "外部运行时" in body and 'hx-get="/ui/external"' in body and "g4a" in body, path


async def test_keepalive_toggle_action(sup, store, client):
    await sup.reconcile_once()
    headers = {"X-CSRF-Token": CSRF}
    r = await client.post("/ui/actions/external-keepalive", headers=headers,
                          data={"endpoint": "gpu-g4-s-1", "enabled": "1", "stop_after_min": "20", "account_id": "default"})
    assert r.status_code == 200 and "保活中" in r.text and "取消接管保活" in r.text
    assert "已接管保活" in toast_title(r)
    row = await store.external("gpu-g4-s-1")
    assert row["keepalive"] == 1 and row["stop_after_min"] == 20
    r = await client.post("/ui/actions/external-keepalive", headers=headers,
                          data={"endpoint": "gpu-g4-s-1", "enabled": "0", "account_id": "default"})
    assert r.status_code == 200 and not (await store.external("gpu-g4-s-1"))["keepalive"]
    r = await client.post("/ui/actions/external-keepalive", headers=headers,
                          data={"endpoint": "gpu-g4-s-1", "enabled": "1", "stop_after_min": "x"})
    assert r.status_code == 409
    r = await client.post("/ui/actions/external-keepalive", headers=headers,
                          data={"endpoint": "gpu-g4-s-1", "enabled": "1", "account_id": "someone-else"})
    assert r.status_code == 409


async def test_stop_action_from_card(sup, store, cli, client):
    await sup.reconcile_once()
    headers = {"X-CSRF-Token": CSRF}
    r = await client.post("/ui/actions/external/g4a/stop", headers=headers,
                          data={"account_id": "default", "endpoint": "gpu-g4-s-1", "card": "1"})
    assert r.status_code == 409 and cli.stops == []  # needs confirm
    r = await client.post("/ui/actions/external/g4a/stop", headers=headers,
                          data={"confirm": "yes", "account_id": "default", "endpoint": "gpu-other", "card": "1"})
    assert r.status_code == 409 and cli.stops == []  # stale card: the name is another VM now
    r = await client.post("/ui/actions/external/g4a/stop", headers=headers,
                          data={"confirm": "yes", "account_id": "default", "endpoint": "gpu-g4-s-1", "card": "1"})
    assert r.status_code == 200 and cli.stops == ["g4a"]
    assert toast_title(r) == "已请求 Colab 停止 g4a"
    assert "g4a" not in r.text.split("<tbody>")[1]  # the card re-rendered without it
    stopped = await events(store, "external_stopped")
    assert len(stopped) == 1 and '"endpoint": "gpu-g4-s-1"' in stopped[0]["detail_json"]
    # the alert-panel button (no endpoint, no card) keeps working and answers 204
    cli.runtimes.append(g4("other", "gpu-g4-s-5"))
    await sup.reconcile_once()
    r = await client.post("/ui/actions/external/other/stop", headers=headers,
                          data={"confirm": "yes", "account_id": "default"})
    assert r.status_code == 204 and cli.stops == ["g4a", "other"]
