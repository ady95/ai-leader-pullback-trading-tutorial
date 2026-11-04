"""06장 B0 일봉 백테스트 엔진.

종목마다 하루씩 앞으로 진행하며 신호 -> 주문 -> 체결 -> 잔고를 이벤트로 기록한다.
하루 안의 순서는 다음으로 고정한다(불리한 쪽을 먼저 가정).
  1) 전날 장 마감에 결정한 시가 청산(손절·시간 청산)
  2) 지정가 매수(1차·2차)
  3) 지정가 매도(1차 익절 -> 상한가 잔여 매도)
  4) 장 마감 판정(손절·시간 청산 결정, 다음 날 1차 매수 주문 가격 계산)
1차 체결일에는 매도하지 않는다(일봉으로는 매수와 매도의 선후를 알 수 없으므로).

사용:
    from pullback.backtest import run
    trades, events = run(df, flags_df, spec)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Rules:
    second_drop: float = 0.04
    second_units: float = 1.0
    take_profit_1: float = 0.20
    take_profit_1_fraction: float = 0.5
    limit_up_ret: float = 0.295
    stop_zone_buffer: float = 0.03
    max_hold_days: int = 60
    commission: float = 0.00015
    slippage: float = 0.001
    sell_tax_by_year: dict = field(default_factory=dict)
    sell_tax_default: float = 0.0020
    strict_touch: bool = False      # True면 지정가 매수는 저가가 주문가보다 '낮아야' 체결
    # 07-3 대안 비교용 (기본값 = B0)
    first_line: str = "b0"          # b0 | ma60 | ma120_env
    use_stop: bool = True
    tp_mode: str = "b0"             # b0 | fixed
    tp_fixed: float = 0.25
    second_require_ma60: bool = False   # 원문 문구: 2차 매수가가 전날 60일선 ±3% 안일 때만
    second_skip_gap: bool = False       # 12-2 C1: 시가가 2차 매수가 아래인 날(갭 체결)에는 2차 매수를 하지 않는다
    # 13장 청산 대안 (기본값 = B0)
    stop_mode: str = "b0"               # b0 | atr: 손절선 = 1차 체결가 x (1 - atr_k x 신호일 ATR/종가)
    atr_k: float = 0.669
    trail_on: float = 0.0               # > 0 이면 최고 고가가 평균 단가 x (1 + trail_on) 이상이 된 뒤 트레일링
    noprog_day: int = 0                 # > 0 이면 그 거래일째 종가가 평균 단가 아래일 때 청산

    @classmethod
    def from_spec(cls, spec: dict, **override) -> "Rules":
        e, x, c, l = spec["entry"], spec["exit"], spec["costs"], spec["leader"]
        kw = dict(second_drop=e["second_drop"], second_units=e["second_units"],
                  take_profit_1=x["take_profit_1"], take_profit_1_fraction=x["take_profit_1_fraction"],
                  limit_up_ret=l["limit_up_ret"], stop_zone_buffer=x["stop_zone_buffer"],
                  max_hold_days=x["max_hold_days"], commission=c["commission_one_way"],
                  slippage=c["slippage_one_way"],
                  sell_tax_by_year={int(k): v for k, v in c.get("sell_tax_by_year", {}).items()},
                  sell_tax_default=c["sell_tax"])
        kw.update(override)
        return cls(**kw)

    def tax(self, year: int) -> float:
        return self.sell_tax_by_year.get(year, self.sell_tax_default)


LINES = {"b0": ["kijun120", "env_low"], "ma60": ["ma60"], "ma120_env": ["ma120", "env_low"]}


def first_buy_price(df: pd.DataFrame, kind: str = "b0") -> pd.Series:
    """t일 장 마감에 계산하는 1차 매수선: 종가 아래에 있는 후보 지지선 중 가장 높은 것 (B0: 기준선·엔벨로프 하단)."""
    cols = df[LINES[kind]]
    return cols.where(cols.lt(df["close"], axis=0)).max(axis=1)


def _touch(low: float, price: float, strict: bool) -> bool:
    return low < price if strict else low <= price


def simulate_stock(x: pd.DataFrame, setup: np.ndarray, r: Rules) -> tuple[list[dict], list[dict]]:
    """한 종목의 일봉(x, 날짜 오름차순)과 후보일 여부로 거래를 만든다."""
    d = x["date"].to_numpy()
    o, h, lo, c = (x[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    pc = x["prev_close"].to_numpy(float)
    halted = x["halted"].to_numpy(bool)
    peak = x["peak_date"].to_numpy()
    line = first_buy_price(x, r.first_line).to_numpy(float)
    ma60 = x["ma60"].to_numpy(float)
    atr_pct = (x["atr"] / x["close"]).to_numpy(float) if "atr" in x else np.full(len(x), np.nan)
    code = x["code"].iloc[0]

    trades, events = [], []
    used_waves: set = set()
    order = None          # 다음 날 1차 매수 주문 (price, wave)
    pos = None            # 보유 상태
    n = len(x)

    def ev(i, kind, price, units, reason=""):
        events.append({"code": code, "date": d[i], "kind": kind, "price": price, "units": units, "reason": reason})

    def close_trade(i, reason):
        t = pos["trade"]
        t.update(exit_date=d[i], exit_reason=reason, hold_days=i - pos["entry_i"])
        trades.append(t)

    for i in range(n):
        # 점하한가: 고가=저가이고 전일 대비 -29.5% 이하. 매도 주문이 체결되지 않는 것으로 본다
        locked_down = h[i] == lo[i] and pc[i] > 0 and c[i] / pc[i] - 1 <= -r.limit_up_ret
        # ---- 장중 ----
        if pos is not None and pos["exit_next_open"] and locked_down and not halted[i]:
            ev(i, "defer", c[i], pos["units"], "locked_limit_down")
        elif pos is not None and not halted[i]:
            if pos["exit_next_open"]:
                price = o[i]
                pos["sells"].append((price, pos["units"], d[i]))
                ev(i, "sell", price, pos["units"], pos["exit_next_open"])
                pos["units"] = 0.0
                close_trade(i, pos["exit_next_open"])
                pos = None
            else:
                entry_day = i == pos["entry_i"]
                near60 = (not r.second_require_ma60) or (i > 0 and abs(ma60[i - 1] / pos["second_price"] - 1) <= 0.03)
                gap_ok = not (r.second_skip_gap and o[i] < pos["second_price"])
                if (not entry_day and r.second_units > 0 and near60 and gap_ok and not pos["second_done"]
                        and _touch(lo[i], pos["second_price"], r.strict_touch)):
                    price = min(o[i], pos["second_price"])
                    pos["buys"].append((price, r.second_units, d[i]))
                    pos["units"] += r.second_units
                    pos["second_done"] = True
                    pos["trade"]["second_date"] = d[i]
                    ev(i, "buy2", price, r.second_units)
                if not entry_day and r.tp_mode == "fixed":
                    avg = sum(p * u for p, u, _ in pos["buys"]) / sum(u for _, u, _ in pos["buys"])
                    tp = avg * (1 + r.tp_fixed)
                    if h[i] >= tp:
                        price = max(o[i], tp)
                        pos["sells"].append((price, pos["units"], d[i]))
                        ev(i, "sell", price, pos["units"], "tp_fixed")
                        pos["units"] = 0.0
                        close_trade(i, "tp_fixed")
                        pos = None
                elif not entry_day:
                    avg = sum(p * u for p, u, _ in pos["buys"]) / sum(u for _, u, _ in pos["buys"])
                    tp1 = avg * (1 + r.take_profit_1)
                    if not pos["tp1_done"] and h[i] >= tp1:
                        price = max(o[i], tp1)
                        u = pos["units"] * r.take_profit_1_fraction
                        pos["sells"].append((price, u, d[i]))
                        pos["units"] -= u
                        pos["tp1_done"] = True
                        ev(i, "sell", price, u, "tp1")
                    if pos["tp1_done"] and pos["units"] > 0 and pc[i] > 0 and h[i] / pc[i] - 1 >= r.limit_up_ret:
                        price = h[i]
                        pos["sells"].append((price, pos["units"], d[i]))
                        ev(i, "sell", price, pos["units"], "tp2_limit_up")
                        pos["units"] = 0.0
                        close_trade(i, "tp2_limit_up")
                        pos = None

        if pos is None and order is not None and not halted[i]:
            price_lim, wave = order
            if _touch(lo[i], price_lim, r.strict_touch):
                price = min(o[i], price_lim)
                used_waves.add(wave)
                pos = {"entry_i": i, "units": 1.0, "buys": [(price, 1.0, d[i])], "sells": [],
                       "second_price": price * (1 - r.second_drop), "second_done": False,
                       "tp1_done": False, "exit_next_open": "", "max_high": 0.0,
                       "atr_pct": atr_pct[i - 1] if i > 0 else np.nan,
                       "trade": {"code": code, "wave_peak": wave, "entry_date": d[i], "entry_price": price,
                                 "gap_fill": bool(o[i] <= price_lim), "second_date": pd.NaT}}
                ev(i, "buy1", price, 1.0)
        order = None

        # ---- 장 마감 ----
        if pos is not None:
            stop_line = pos["second_price"] * (1 - r.stop_zone_buffer)
            if r.stop_mode == "atr" and not np.isnan(pos["atr_pct"]):
                stop_line = pos["buys"][0][0] * (1 - r.atr_k * pos["atr_pct"])
            if i > pos["entry_i"]:
                pos["max_high"] = max(pos["max_high"], h[i])
            avg = sum(p * u for p, u, _ in pos["buys"]) / sum(u for _, u, _ in pos["buys"])
            trail = (r.trail_on > 0 and not np.isnan(pos["atr_pct"]) and pos["max_high"] >= avg * (1 + r.trail_on)
                     and c[i] < pos["max_high"] * (1 - r.atr_k * pos["atr_pct"]))
            if not halted[i]:
                if r.use_stop and c[i] < stop_line:
                    pos["exit_next_open"] = "stop"
                elif trail:
                    pos["exit_next_open"] = "trail"
                elif r.noprog_day > 0 and i - pos["entry_i"] == r.noprog_day and c[i] < avg:
                    pos["exit_next_open"] = "noprog"
                elif i - pos["entry_i"] >= r.max_hold_days:
                    pos["exit_next_open"] = "time"
        elif setup[i] and not np.isnan(line[i]) and peak[i] not in used_waves:
            order = (line[i], peak[i])

    if pos is not None:            # 데이터 끝: 폐지 종목이면 마지막 종가, 아니면 평가 종료
        last = n - 1
        reason = "data_end"
        pos["sells"].append((c[last], pos["units"], d[last]))
        ev(last, "sell", c[last], pos["units"], reason)
        close_trade(last, reason)
    return trades, events


def _pnl(trade_buys, trade_sells, r: Rules) -> tuple[float, float, float]:
    cost = sum(p * u * (1 + r.commission + r.slippage) for p, u, _ in trade_buys)
    gross_cost = sum(p * u for p, u, _ in trade_buys)
    gross = sum(p * u for p, u, _ in trade_sells)
    net = sum(p * u * (1 - r.commission - r.slippage - r.tax(pd.Timestamp(dt).year)) for p, u, dt in trade_sells)
    return cost, gross / gross_cost - 1, net / cost - 1


def run(df: pd.DataFrame, setup: pd.Series, r: Rules) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전 종목 실행. trades 표(거래당 한 행)와 events 표(체결 기록)를 돌려준다."""
    cols = ["code", "date", "open", "high", "low", "close", "prev_close", "halted",
            "peak_date", "kijun120", "env_low", "ma60", "ma120"] + (["atr"] if "atr" in df else [])
    all_trades, all_events = [], []
    for code, x in df[cols].groupby("code", sort=True):
        s = setup.loc[x.index].to_numpy(bool)
        if not s.any():
            continue
        t, e = simulate_stock(x.sort_values("date"), s, r)
        all_trades += t
        all_events += e
    events = pd.DataFrame(all_events)
    trades = pd.DataFrame(all_trades)
    if trades.empty:
        return trades, events
    # 체결 기록으로 손익 계산
    rows = []
    for (code, entry), g in events.assign(entry=events.groupby("code")["kind"].transform(
            lambda k: (k == "buy1").cumsum())).groupby(["code", "entry"]):
        buys = [(p, u, dt) for p, u, dt, k in zip(g["price"], g["units"], g["date"], g["kind"]) if k.startswith("buy")]
        sells = [(p, u, dt) for p, u, dt, k in zip(g["price"], g["units"], g["date"], g["kind"]) if k == "sell"]
        invested, gross, net = _pnl(buys, sells, r)
        rows.append({"code": code, "entry_date": g.loc[g["kind"] == "buy1", "date"].iloc[0],
                     "units_max": sum(u for _, u, _ in buys), "invested": invested,
                     "gross_ret": gross, "net_ret": net})
    pnl = pd.DataFrame(rows)
    trades = trades.merge(pnl, on=["code", "entry_date"], how="left")
    return trades.sort_values(["entry_date", "code"]).reset_index(drop=True), events
