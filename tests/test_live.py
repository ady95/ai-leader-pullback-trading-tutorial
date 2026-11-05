"""15장 운영 시스템 테스트: 손으로 만든 시나리오에서 운영 엔진 == 백테스트, 주문 변환, 감사 기록, 화면."""
import json
import math

import pandas as pd
import pytest

from pullback.backtest import Rules, run
from pullback.live.audit import AuditLog, verify
from pullback.live.engine import LiveEngine
from pullback.live.risk import ceil_tick, convert, floor_tick, tick, upper_limit
from pullback.live.orders import Order
from pullback.live.sim import SimBroker

R = Rules(sell_tax_by_year={2024: 0.0018}, commission=0.0, slippage=0.0)


def frame(bars, line=100.0):
    rows = []
    for k, b in enumerate(bars):
        o, h, lo, c = b[:4]
        rows.append({"code": "000010", "date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=k),
                     "open": o, "high": h, "low": lo, "close": c, "volume": b[4] if len(b) > 4 else 1000})
    df = pd.DataFrame(rows)
    df["prev_close"] = df["close"].shift(1)
    df["halted"] = df["volume"] == 0
    df["peak_date"] = pd.Timestamp("2024-01-01")
    df["kijun120"], df["env_low"], df["ma60"], df["ma120"] = line, line * 0.8, line * 0.9, line * 0.7
    setup = pd.Series(False, index=df.index)
    setup.iloc[0] = True
    return df, setup


def live_events(df, setup, r=R):
    eng, sim, out = LiveEngine(r), SimBroker(r), []
    for i, x in df.iterrows():
        day = str(x["date"].date())
        bar = {k: x[k] for k in ("open", "high", "low", "close", "prev_close", "halted")}
        out += [(f.kind, round(f.price, 6), f.units, f.reason) for f in sim.run_day(eng, day, {x["code"]: bar})]
        setups = [(x["code"], 100.0, "2024-01-01")] if setup.iloc[i] else []
        eng.end_of_day(day, {x["code"]: bar}, setups)
    return out


def bt_events(df, setup, r=R):
    _, ev = run(df, setup, r)
    ev = ev[ev["reason"] != "data_end"]
    return [(k, round(p, 6), u, rs) for k, p, u, rs in zip(ev["kind"], ev["price"], ev["units"], ev["reason"])]


SCENARIOS = {
    "gap_down": [(110, 112, 105, 108), (95, 99, 90, 97), (97, 98, 96, 97)],
    "buy2_then_tp1_same_day": [(110, 112, 105, 108), (101, 102, 99, 100), (98, 120, 95, 115), (115, 116, 114, 115)],
    "tp1_then_limit_up": [(110, 112, 105, 108), (101, 102, 99, 100), (98, 99, 95, 97),
                          (110, 118, 108, 115), (140, 149, 139, 149)],
    "stop_next_open": [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 95, 92), (90, 91, 88, 89)],
    "locked_limit_down": [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 95, 92),
                          (64.4, 64.4, 64.4, 64.4), (60, 66, 59, 65)],
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_live_engine_matches_backtest(name):
    df, s = frame(SCENARIOS[name])
    assert live_events(df, s) == bt_events(df, s)


def test_buy2_fill_reprices_take_profit():
    # 2차 매수(96) 체결 뒤 1차 익절가는 평균 98 x 1.2 = 117.6 으로 고쳐져야 한다 (장 전 주문은 120)
    df, s = frame(SCENARIOS["buy2_then_tp1_same_day"])
    ev = live_events(df, s)
    assert ("sell", 117.6, 1.0, "tp1") in ev


def test_state_roundtrip(tmp_path):
    df, s = frame(SCENARIOS["tp1_then_limit_up"][:3])
    eng, sim = LiveEngine(R), SimBroker(R)
    for i, x in df.iterrows():
        day = str(x["date"].date())
        bar = {k: x[k] for k in ("open", "high", "low", "close", "prev_close", "halted")}
        sim.run_day(eng, day, {x["code"]: bar})
        eng.end_of_day(day, {x["code"]: bar}, [(x["code"], 100.0, "2024-01-01")] if s.iloc[i] else [])
    eng.save(tmp_path / "s.json")
    back = LiveEngine.load(R, tmp_path / "s.json")
    assert back.s.positions["000010"].units == 2.0
    assert [o.kind for o in back.book["000010"]] == [o.kind for o in eng.book["000010"]]


def test_ticks_and_conversion():
    assert tick(1_999) == 1 and tick(2_000) == 5 and tick(49_950) == 50 and tick(500_000) == 1_000
    assert floor_tick(17_905.9) == 17_900 and ceil_tick(22_382.4) == 22_400
    assert ceil_tick(19_996) == 20_000          # 경계를 넘으면 위 단위
    assert upper_limit(10_000) == 13_000
    o = Order("000010", "buy1", 18_652.4, 1.0, "2026-11-05")
    bo, why = convert(o, prev_close=19_000)
    assert why == "" and bo.price == 18_650 and bo.qty == 53
    big = Order("000010", "buy1", 1_174_880, 1.0, "2026-11-05")
    assert convert(big, prev_close=1_200_000)[1] == "주식 수 0"
    tp = Order("000010", "tp1", 120.0, 0.5, "2026-11-05")
    assert convert(tp, 100, held_qty=1)[0].qty == 1   # 1주는 나눌 수 없어 전량
    ex = Order("000010", "exit_open", math.nan, 2.0, "2026-11-05", "stop")
    assert convert(ex, 100, held_qty=10)[0].order_type == "market"


def test_audit_chain_detects_edit(tmp_path):
    p = tmp_path / "a.jsonl"
    log = AuditLog(p)
    for i in range(5):
        log.append("day", {"i": i})
    assert verify(p) == (True, 0)
    AuditLog(p).append("day", {"i": 5})          # 다시 열어 이어 쓰기
    assert verify(p) == (True, 0)
    lines = p.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["data"]["i"] = 99
    lines[2] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify(p) == (False, 3)


def test_dashboard_renders(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    eng = LiveEngine(R)
    eng.s.last_date = "2026-11-05"
    eng.save(tmp_path / "state.json")
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("day", {"date": "2026-11-05", "orders": []})
    (tmp_path / "plan_2026-11-06.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv("LIVE_DIR", str(tmp_path))
    at = AppTest.from_file("../app/dashboard.py", default_timeout=30).run()
    assert not at.exception
    assert any("2026-11-05" in m.value for m in at.metric)
