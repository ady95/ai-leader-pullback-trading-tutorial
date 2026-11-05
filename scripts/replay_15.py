"""15-1 운영 엔진 재생: 2020-01-02 ~ 2026-11-03을 하루씩 돌려 백테스트와 이벤트를 대조한다.

매일 (1) 모의 체결기가 전날 밤 주문을 그날 일봉에 체결시키고 (2) 운영 엔진이 장 마감 판정과
다음 날 주문을 만든다. 엔진 상태는 마지막에 data/live/state.json 으로 저장하고(15-2 배치가 이어 받음),
매일의 결정은 data/live/replay_audit.jsonl 에 해시 체인으로 남긴다(15-3).

사용:
    uv run python scripts/replay_15.py              # 대조 + 주문 변환 통계 (약 1분)
    uv run python scripts/replay_15.py --mutate tp1 # 일부러 틀린 엔진: 대조가 차이를 잡는지 확인
"""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from pullback.backtest import Rules, first_buy_price, run
from pullback.definitions import build, flags, load_daily
from pullback.live.audit import AuditLog, verify
from pullback.live.engine import LiveEngine
from pullback.live.risk import ceil_tick, floor_tick, upper_limit
from pullback.live.sim import SimBroker
from pullback.paths import DATA
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
LIVE = DATA / "live"
LIVE_ENGINE = LiveEngine
BAR = ["open", "high", "low", "close", "prev_close", "halted"]


class NoReprice(LiveEngine):
    """틀린 엔진 1: 2차 매수 체결 뒤 1차 익절 주문을 고치지 않는다 (장 전에 낸 주문 그대로)."""
    def on_fill(self, f):
        if f.kind == "buy2":
            p = self.s.positions[f.code]
            p.buys.append([f.price, f.units, f.date])
            p.units += f.units
            p.second_done = True
            return []
        return super().on_fill(f)


class HoldFromEntry(LiveEngine):
    """틀린 엔진 2: 보유 일수를 진입일부터 센다 (하루 일찍 시간 청산)."""
    def end_of_day(self, day, bars, setups):
        for code, p in self.s.positions.items():
            if p.entry_date == day and code in bars:
                p.days_held += 1
        return super().end_of_day(day, bars, setups)


MUTANTS = {"tp1": NoReprice, "hold": HoldFromEntry}


def replay(df: pd.DataFrame, setup: pd.Series, r: Rules, start: str, end: str, digest: str,
           audit_path: Path, engine_cls=LiveEngine) -> tuple[LiveEngine, pd.DataFrame, list[dict]]:
    d = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    d["line"] = first_buy_price(d)
    d["setup"] = setup.loc[d.index].to_numpy(bool)
    d = d.sort_values(["date", "code"]).reset_index(drop=True)
    dates = d["date"].to_numpy()
    days = np.unique(dates)
    bounds = np.searchsorted(dates, days, side="left").tolist() + [len(d)]
    codes = d["code"].to_numpy()
    vals = {k: d[k].to_numpy() for k in BAR}
    eng, sim = engine_cls(r), SimBroker(r)
    fills, log_rows = [], []
    if audit_path.exists():
        audit_path.unlink()
    log = AuditLog(audit_path)
    for j, day in enumerate(days):
        lo_i, hi_i = bounds[j], bounds[j + 1]
        idx = pd.Index(codes[lo_i:hi_i])
        ds = str(pd.Timestamp(day).date())

        def bars_for(cs):
            cs = list(cs)
            pos = idx.get_indexer(cs)
            return {c: {k: vals[k][lo_i + p] for k in BAR} for c, p in zip(cs, pos) if p >= 0}

        f = sim.run_day(eng, ds, bars_for(eng.book.keys()))
        fills += [x.as_dict() for x in f]
        today = d.iloc[lo_i:hi_i]
        su = today[today["setup"]]
        setups = [(c, float(ln), str(pd.Timestamp(pk).date())) for c, ln, pk in
                  zip(su["code"], su["line"], su["peak_date"])]
        need = set(eng.s.positions) | set(eng.book)
        orders = eng.end_of_day(ds, bars_for(need), setups)
        log.append("day", {"date": ds, "spec": digest[:16], "setups": len(setups),
                           "fills": [x.as_dict() for x in f], "orders": [o.as_dict() for o in orders],
                           "positions": len(eng.s.positions)})
        log_rows.append({"date": ds, "setups": len(setups), "orders": len(orders), "fills": len(f),
                         "positions": len(eng.s.positions), "carried": len(eng.carried)})
    return eng, pd.DataFrame(fills), log_rows


def compare(bt: pd.DataFrame, lv: pd.DataFrame, tol: float) -> tuple[int, pd.DataFrame]:
    cols = ["code", "date", "kind", "price", "units", "reason"]
    a = bt[cols].copy()
    a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
    b = lv[cols].copy()
    for x in (a, b):
        x["reason"] = x["reason"].fillna("")
        x["n"] = x.groupby("code").cumcount()
    m = a.merge(b, on=["code", "n"], how="outer", suffixes=("_bt", "_lv"), indicator=True)
    bad = m["_merge"] != "both"
    both = ~bad
    for k in ("date", "kind", "reason"):
        bad |= both & (m[k + "_bt"] != m[k + "_lv"])
    for k in ("price", "units"):
        bad |= both & ~np.isclose(m[k + "_bt"].astype(float), m[k + "_lv"].astype(float), rtol=tol, atol=0)
    return int(bad.sum()), m[bad]


def conversion_stats(fl: pd.DataFrame, df: pd.DataFrame, start: str, first_money: float) -> None:
    """백테스트 체결을 실제 주문 규칙(호가 단위·정수 주식 수)으로 옮기면 무엇이 달라지나 (서술 통계)."""
    bars = df[["code", "date", "low", "high"]].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
    x = fl[fl["date"] >= start].merge(bars, on=["code", "date"], how="left")
    print(f"\n== 주문 변환 차이 ({start} ~, 수정주가 기준 근사)")
    for kind in ("buy1", "buy2"):
        b = x[x["kind"] == kind]
        lost = b["low"] > b["order_price"].map(floor_tick)
        print(f"{kind}: 체결 {len(b)}건 중 호가 내림으로 체결되지 않았을 주문 {int(lost.sum())}건 ({lost.mean():.2%})")
    t1 = x[(x["kind"] == "sell") & (x["reason"] == "tp1")]
    lost = t1["high"] < t1["order_price"].map(ceil_tick)
    print(f"tp1: 체결 {len(t1)}건 중 호가 올림으로 체결되지 않았을 주문 {int(lost.sum())}건 ({lost.mean():.2%})")
    lu = x[(x["kind"] == "sell") & (x["reason"] == "tp2_limit_up")]
    lost = lu["high"] < lu["order_price"].map(upper_limit)
    print(f"tp2_limit_up: 체결 {len(lu)}건 중 고가가 상한가(근사)에 못 미친 경우 {int(lost.sum())}건 ({lost.mean():.2%})")
    b1 = x[x["kind"] == "buy1"].copy()
    b1["qty"] = (first_money // b1["order_price"].map(floor_tick)).astype(int)
    short = 1 - b1["qty"] * b1["price"] / first_money
    odd = b1["qty"] % 2 == 1
    print(f"1차 주식 수: 중앙값 {b1['qty'].median():.0f}주, 1주 이하 {int((b1['qty'] <= 1).sum())}건, "
          f"0주 {int((b1['qty'] == 0).sum())}건, 홀수 {int(odd.sum())}건 ({odd.mean():.1%})")
    print(f"1차 실제 금액의 100만 원 대비 부족분: 중앙값 {short.median():.2%}, 95백분위 {short.quantile(0.95):.2%}, "
          f"최대 {short.max():.2%}")


def max_exposure(fl: pd.DataFrame, first_money: float) -> tuple[float, str]:
    """복리 없는 계좌에서 동시에 투입된 매수 원가 합계의 최댓값 (원). 매도는 평균 원가로 줄인다."""
    held: dict[str, list[float]] = {}     # code -> [원가 합, 단위, 1단위당 주식 배율]
    out = []
    for rec in fl[fl["kind"].isin(["buy1", "buy2", "sell"])].itertuples():
        if rec.kind == "buy1":
            held[rec.code] = [0.0, 0.0, first_money / rec.price]
        elif rec.code not in held:          # 시작일 이전에 진입한 거래는 세지 않는다
            continue
        h = held[rec.code]
        if rec.kind.startswith("buy"):
            cost = rec.price * h[2] * rec.units
            h[0] += cost
            h[1] += rec.units
            out.append((rec.date, cost))
        else:
            dc = h[0] * rec.units / h[1]
            h[0] -= dc
            h[1] -= rec.units
            out.append((rec.date, -dc))
    s = pd.DataFrame(out, columns=["date", "d"]).groupby("date")["d"].sum().sort_index().cumsum()
    return float(s.max()), str(s.idxmax())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", choices=sorted(MUTANTS))
    a = ap.parse_args()
    cfg = tomllib.loads(Path("configs/live_15.toml").read_text(encoding="utf-8"))
    pc = cfg["checks"]["parity"]
    spec, digest = load(cfg["strategy"]["spec"])
    p = to_params(spec)
    t0 = time.perf_counter()
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    r = Rules.from_spec(spec)
    t1 = time.perf_counter()
    trades, ev = run(df, f["setup"], r)
    t2 = time.perf_counter()
    LIVE.mkdir(parents=True, exist_ok=True)
    audit = LIVE / ("replay_audit.jsonl" if not a.mutate else f"mutant_{a.mutate}.jsonl")
    eng, fl, days = replay(df, f["setup"], r, pc["replay_start"], pc["replay_end"], digest, audit,
                           MUTANTS.get(a.mutate, LIVE_ENGINE))
    t3 = time.perf_counter()
    print(f"spec {digest[:16]} 지표 {t1 - t0:.0f}s, 백테스트 {t2 - t1:.0f}s, 운영 엔진 재생 {t3 - t2:.0f}s "
          f"({len(days)}거래일)")

    bt = ev[ev["reason"] != "data_end"].reset_index(drop=True)
    n_bad, bad = compare(bt, fl, pc["price_rel_tol"])
    print(f"\n== 이벤트 대조: 백테스트 {len(bt):,}건 (data_end {int((ev['reason'] == 'data_end').sum())}건 제외), "
          f"운영 엔진 {len(fl):,}건, 불일치 {n_bad}건")
    print(fl.groupby("kind").size().to_dict())
    if n_bad:
        print(bad.head(6)[["code", "n", "date_bt", "kind_bt", "price_bt", "reason_bt",
                           "date_lv", "kind_lv", "price_lv", "reason_lv"]].to_string())
    if a.mutate:
        audit.unlink()
        return
    de = trades[trades["exit_reason"] == "data_end"]
    de_set = {(c, str(pd.Timestamp(e).date())) for c, e in zip(de["code"], de["entry_date"])}
    lv_set = {(c, q.entry_date) for c, q in eng.s.positions.items()}
    print(f"미청산: 백테스트 data_end {len(de_set)}건, 운영 엔진 보유 {len(lv_set)}건, 같음 {de_set == lv_set}")
    dd = pd.DataFrame(days)
    print(f"넘긴 주문(그날 일봉 없음) 연인원 {int(dd['carried'].sum())}건, 하루 최대 주문 {int(dd['orders'].max())}건, "
          f"하루 최대 보유 {int(dd['positions'].max())}종목")
    ok, line = verify(audit)
    print(f"감사 기록 {len(days)}줄, 해시 체인 검증 " + ("통과" if ok else f"실패 (줄 {line})"))
    eng.save(LIVE / "state.json")
    fl.to_parquet(LIVE / "replay_fills.parquet", index=False)
    dd.to_parquet(LIVE / "replay_days.parquet", index=False)
    print(f"상태 저장 {LIVE / 'state.json'} (마지막 날 {eng.s.last_date}, 보유 {len(eng.s.positions)}종목)")

    start = cfg["order_conversion"]["report_period"].split(" ~ ")[0]
    conversion_stats(fl, df, start, cfg["strategy"]["first_money"])
    mx, when = max_exposure(fl[fl["date"] >= start], cfg["strategy"]["first_money"])
    print(f"동시 투입 원가 최대 {mx / 1e8:.2f}억 원 ({when}), 계좌 {cfg['strategy']['capital'] / 1e8:.0f}억 원")


if __name__ == "__main__":
    main()
