"""06-3 손으로 만든 시나리오로 B0 엔진의 규칙을 하나씩 확인한다.

사용:
    uv run pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from pullback.backtest import Rules, run

R = Rules(sell_tax_by_year={2024: 0.0018}, commission=0.0, slippage=0.0)


def frame(bars, setup_days=(0,), line=100.0, peak="2024-01-01"):
    """bars: (open, high, low, close[, volume]) 목록. 0일 장 마감에 1차 매수선 line 이 정해진다."""
    rows = []
    for k, b in enumerate(bars):
        o, h, lo, c = b[:4]
        vol = b[4] if len(b) > 4 else 1000
        rows.append({"code": "000010", "date": pd.Timestamp("2024-03-01") + pd.Timedelta(days=k),
                     "open": o, "high": h, "low": lo, "close": c, "volume": vol})
    df = pd.DataFrame(rows)
    df["prev_close"] = df["close"].shift(1)
    df["halted"] = df["volume"] == 0
    df["peak_date"] = pd.Timestamp(peak)
    df["kijun120"] = line
    df["env_low"] = line * 0.8
    setup = pd.Series(False, index=df.index)
    setup.iloc[list(setup_days)] = True
    return df, setup


def test_gap_down_fills_at_open():
    df, s = frame([(110, 112, 105, 108), (95, 99, 90, 97), (97, 98, 96, 97)])
    trades, ev = run(df, s, R)
    assert ev.iloc[0]["kind"] == "buy1" and ev.iloc[0]["price"] == 95
    assert bool(trades.loc[0, "gap_fill"]) is True


def test_second_buy_then_take_profits():
    bars = [(110, 112, 105, 108),
            (101, 102, 99, 100),      # 1차 100
            (98, 99, 95, 97),         # 2차 96 (평균 98)
            (110, 118, 108, 115),     # 1차 익절 98 x 1.2 = 117.6
            (140, 149, 139, 149)]     # 전일 115 대비 +29.6% -> 상한가 잔여 매도 149
    df, s = frame(bars)
    trades, ev = run(df, s, R)
    kinds = list(zip(ev["kind"], ev["price"].round(2), ev["units"]))
    assert kinds == [("buy1", 100.0, 1.0), ("buy2", 96.0, 1.0), ("sell", 117.6, 1.0), ("sell", 149.0, 1.0)]
    t = trades.iloc[0]
    assert t["exit_reason"] == "tp2_limit_up"
    expected = (117.6 + 149.0) * (1 - 0.0018) / (100 + 96) - 1
    assert t["net_ret"] == pytest.approx(expected)


def test_no_exit_on_entry_day():
    df, s = frame([(110, 112, 105, 108), (101, 130, 99, 125), (125, 126, 124, 125)])
    _, ev = run(df, s, R)
    entry_day = df["date"].iloc[1]                     # 고가 130 (+30%) 이지만 1차 체결일
    assert not ((ev["kind"] == "sell") & (ev["date"] == entry_day)).any()
    assert ev[ev["kind"] == "sell"]["date"].min() == df["date"].iloc[2]


def test_stop_executes_next_open():
    # 1차 100, 2차 96, 손절선 96 x 0.97 = 93.12
    bars = [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 92, 93), (91, 92, 89, 90)]
    df, s = frame(bars)
    trades, ev = run(df, s, R)
    assert trades.iloc[0]["exit_reason"] == "stop"
    assert ev.iloc[-1]["price"] == 91 and ev.iloc[-1]["date"] == df["date"].iloc[3]


def test_halt_defers_stop_until_resume():
    bars = [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 92, 93),
            (93, 93, 93, 93, 0), (93, 93, 93, 93, 0), (70, 75, 65, 72)]
    df, s = frame(bars)
    trades, ev = run(df, s, R)
    assert trades.iloc[0]["exit_reason"] == "stop"
    assert ev.iloc[-1]["price"] == 70 and ev.iloc[-1]["date"] == df["date"].iloc[5]


def test_time_exit_after_max_hold_days():
    bars = [(110, 112, 105, 108), (101, 102, 99, 100)] + [(100, 101, 99, 100)] * 62
    df, s = frame(bars)
    trades, _ = run(df, s, Rules(max_hold_days=60, commission=0, slippage=0))
    t = trades.iloc[0]
    assert t["exit_reason"] == "time" and t["hold_days"] == 61


def test_strict_touch_needs_trade_below_limit():
    df, s = frame([(110, 112, 105, 108), (101, 102, 100, 101), (101, 102, 100, 101)])
    loose, _ = run(df, s, R)
    strict, _ = run(df, s, Rules(strict_touch=True, commission=0, slippage=0))
    assert len(loose) == 1 and strict.empty


def test_one_trade_per_wave():
    bars = [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 92, 93), (91, 92, 89, 90),
            (108, 110, 105, 108), (101, 102, 99, 100), (100, 100, 100, 100)]
    df, s = frame(bars, setup_days=(0, 4))
    trades, _ = run(df, s, R)
    assert len(trades) == 1


def test_costs_reduce_return():
    bars = [(110, 112, 105, 108), (101, 102, 99, 100), (100, 101, 99, 100)]
    df, s = frame(bars)
    t0, _ = run(df, s, R)
    t1, _ = run(df, s, Rules(sell_tax_by_year={2024: 0.0018}, commission=0.00015, slippage=0.001))
    assert t1.iloc[0]["net_ret"] < t0.iloc[0]["net_ret"]
    assert np.isclose(t0.iloc[0]["gross_ret"], 0.0)


def test_locked_limit_down_defers_stop():
    # 손절 결정 다음 날 점하한가(93 -> 65.1, 고가=저가) -> 매도 못 하고 그다음 날 시가 60에 청산
    bars = [(110, 112, 105, 108), (101, 102, 99, 100), (97, 98, 92, 93),
            (65.1, 65.1, 65.1, 65.1), (60, 66, 58, 62)]
    df, s = frame(bars)
    trades, ev = run(df, s, R)
    assert "defer" in set(ev["kind"])
    assert ev[ev["kind"] == "sell"].iloc[-1]["price"] == 60
