"""15-2 장 마감 배치.

    uv run python scripts/batch_15.py --coverage     # 감시 목록이 B0 후보일을 모두 덮는가 (저장소 기준)
    uv run python scripts/batch_15.py --truncation   # 마지막 20거래일: t일까지의 데이터만으로 같은 후보가 나오는가
    uv run python scripts/batch_15.py --live         # 실제 장 마감 데이터로 2026-11-04, 11-05 배치 (주문은 내지 않음)

--live 는 scripts/replay_15.py 가 남긴 data/live/state.json (2026-11-03 장 마감 상태와 11-04 주문)에서
이어 받는다. 체결은 16장 전까지 모의 체결기(백테스트 규칙)로 가정한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd

from pullback.backtest import Rules, first_buy_price
from pullback.definitions import build, flags, load_daily
from pullback.kiwoom import KiwoomClient
from pullback.live import data as D
from pullback.live.audit import AuditLog, verify
from pullback.live.engine import LiveEngine
from pullback.live.risk import cash_check, convert
from pullback.live.sim import SimBroker
from pullback.paths import DATA
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
LIVE = DATA / "live"
BAR = ["open", "high", "low", "close", "prev_close", "halted"]


def code_version() -> str:
    """감사 기록에 남길 코드 버전 (git 커밋). 커밋하지 않은 변경이 있으면 표시한다."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def cfg() -> dict:
    return tomllib.loads(Path("configs/live_15.toml").read_text(encoding="utf-8"))


def coverage(spec: dict) -> None:
    p = to_params(spec)
    raw = load_daily(spec["periods"]["warmup_start"])
    df = build(raw, p)
    f = flags(df, p)
    lu = df["limit_up"].astype(int)
    # 전날까지 최근 60개 일봉 안의 상한가 수 (= 전날 밤 감시 목록) + 오늘 상한가
    prev60 = lu.groupby(df["code"]).transform(lambda s: s.shift(1).rolling(D.WATCH_BARS, min_periods=1).sum())
    covered = (prev60.fillna(0) > 0) | df["limit_up"]
    s = f["setup"] & (df["date"] >= "2021-01-01")
    print(f"== 감시 목록 포함률: B0 후보일 {int(s.sum()):,}행 중 {int((covered & s).sum()):,}행 "
          f"({(covered & s).sum() / s.sum():.4f})")
    # 후보일이 되기 전, 그 파동의 상한가가 '오늘 상한가'로 처음 잡혀야 하는 날
    w = df[df["date"] >= "2021-01-01"]
    size = w.assign(c=(prev60.loc[w.index].fillna(0) > 0) | w["limit_up"]).groupby("date")["c"].sum()
    print(f"감시 목록 크기(종목): 중앙값 {size.median():.0f}, 최대 {size.max():.0f} ({size.idxmax():%Y-%m-%d}), "
          f"전체 종목 {df[df['date'] == df['date'].max()]['code'].nunique():,}")
    by_year = size.groupby(size.index.year).median()
    print("연도별 감시 목록 중앙값", by_year.astype(int).to_dict())
    # ka10017 은 '상한가에 있는 종목'을 준다. 종가 +29.5% 이상인데 상한가가 아닌 날이 있는가
    from pullback.live.risk import upper_limit
    last_adj = raw[raw["adj_flag"].notna()].groupby("code")["date"].max()
    x = df[(df["date"] >= "2025-01-01") & df["limit_up"]]
    off = x["close"] < x["prev_close"].map(upper_limit)
    clean = ~(x["date"] < x["code"].map(last_adj))         # 그 뒤에 수정주가 조정이 없는 날 (가격이 실제 호가와 같다)
    print(f"2025~2026 상한가(종가 +29.5% 이상) {len(x)}건 중 근사 상한가 미만으로 끝난 날 {int(off.sum())}건, "
          f"이후 수정주가 조정이 없는 {int(clean.sum())}건만 보면 {int((off & clean).sum())}건 "
          f"(ka10017 '상한' 목록에 없을 수 있음)")


def truncation(spec: dict, c: dict) -> None:
    tc = c["checks"]["truncation"]
    p = to_params(spec)
    raw = load_daily(spec["periods"]["warmup_start"])
    full = build(raw, p)
    ff = flags(full, p)
    full["line"] = first_buy_price(full)
    days = sorted(full["date"].unique())[-tc["days"]:]
    total_bad, times, n_cand, max_rel, n_nan = 0, [], 0, 0.0, 0
    for day in days:
        ds = str(pd.Timestamp(day).date())
        t0 = time.perf_counter()
        start = pd.Timestamp(day) - pd.Timedelta(days=tc["lookback_calendar_days"])
        win = raw[(raw["date"] >= start) & (raw["date"] <= day)]
        before = win[win["date"] < day]
        watch = D.watchlist(before, str((pd.Timestamp(day) - pd.Timedelta(days=1)).date()), p)
        today = win[win["date"] == day]
        pc = win[win["date"] < day].groupby("code")["close"].last()
        lu_today = set(today.loc[(today["close"] / today["code"].map(pc) - 1 >= p.limit_up_ret)
                                 & (today["volume"] > 0), "code"])
        codes = watch | lu_today
        setups, _ = D.signals(win[win["code"].isin(codes)], ds, p)
        times.append(time.perf_counter() - t0)
        ref = full[(full["date"] == day) & ff["setup"]]
        a = {(cd, str(pd.Timestamp(pk).date())): ln for cd, ln, pk in zip(ref["code"], ref["line"], ref["peak_date"])}
        b = {(cd, pk): ln for cd, ln, pk in setups}
        keys_bad = len(set(a) ^ set(b))
        def rel_diff(x, y):
            if x != x or y != y:                 # 1차 매수선 없음(NaN): 둘 다 없어야 같다
                return 0.0 if (x != x and y != y) else float("inf")
            return abs(y / x - 1)
        rel = [rel_diff(a[k], b[k]) for k in set(a) & set(b)]
        n_nan += sum(1 for v in a.values() if v != v)
        price_bad = sum(x > tc["price_rel_tol"] for x in rel)
        max_rel = max(max_rel, max(rel, default=0.0))
        total_bad += keys_bad + price_bad
        n_cand += len(a)
        print(f"{ds}: 감시 {len(codes)}종목, 후보 전체 데이터 {len(a)} / 배치 {len(b)}, 종목·파동 불일치 {keys_bad}, "
              f"가격 불일치 {price_bad}, 최대 상대 차이 {max(rel, default=0.0):.1e}, {times[-1]:.1f}s")
    print(f"== 절단 배치 {len(days)}일: 후보 {n_cand}건 (1차 매수선 없음 {n_nan}건), 불일치 {total_bad}건 (가격 최대 상대 차이 {max_rel:.1e}), 배치 시간 중앙값 "
          f"{pd.Series(times).median():.1f}s, 최대 {max(times):.1f}s (기준 {tc['max_batch_seconds']}s)")


def live(spec: dict, c: dict) -> None:
    lc = c["checks"]["live_run"]
    p = to_params(spec)
    r = Rules.from_spec(spec)
    eng = LiveEngine.load(r, LIVE / "state.json")
    assert eng.s.last_date == "2026-11-03", eng.s.last_date
    log = AuditLog(LIVE / "audit.jsonl")
    t0 = time.perf_counter()
    kw = KiwoomClient()
    store = load_daily("2025-06-01")
    watch = D.watchlist(store, eng.s.last_date, p)
    lu4 = D.limit_up_today(kw, previous_day=True)     # 11-05 밤에 부르면 11-04 상한가 (ka10017 전일 상한)
    lu5 = D.top_gainers(kw, p.limit_up_ret)            # 11-05 종가 +29.5% 이상 (ka10027)
    lu5_17 = D.limit_up_today(kw)
    print(f"11-05 상한가: ka10027(+29.5% 이상) {len(lu5)}종목, ka10017(상한) {len(lu5_17)}종목, "
          f"ka10017에만 {sorted(set(lu5_17['code']) - set(lu5['code']))}, ka10027에만 {sorted(set(lu5['code']) - set(lu5_17['code']))}")
    new = set(lu4.loc[lu4["common"], "code"]) | set(lu5.loc[lu5["common"], "code"])
    held = set(eng.s.positions) | set(eng.book)
    codes = sorted(watch | new | held)
    print(f"감시 목록: 저장소 {len(watch)} + 11-04 상한가 {len(lu4)}({int(lu4['common'].sum())} 보통주) "
          f"+ 11-05 상한가 {len(lu5)}({int(lu5['common'].sum())} 보통주) + 보유·주문 {len(held)} = {len(codes)}종목")
    calls0 = kw.calls
    rows = D.refresh(kw, codes, lc["dates"][-1])
    t_fetch = time.perf_counter() - t0
    print(f"수집: {len(codes)}종목, 호출 {kw.calls - calls0}회, 재시도 {kw.retries}, {t_fetch:.0f}s, "
          f"빈 종목 {sum(1 for v in rows.values() if v == 0)}")
    df = D.load_live(codes)
    # 데이터 점검: 저장소의 2026-11-03 종가와 새로 받은 값이 다른가 (그 사이 수정주가 조정)
    a = store[store["date"] == "2026-11-03"].set_index("code")["close"]
    b = df[df["date"] == "2026-11-03"].set_index("code")["close"]
    common = a.index.intersection(b.index)
    diff = common[(a.loc[common] - b.loc[common]).abs() > 1e-6]
    print(f"11-03 종가 대조: {len(common)}종목 중 다른 값 {len(diff)} {list(diff)[:10]}")
    for cd in diff:
        if cd in eng.s.positions:
            print(f"  [경고] 보유 종목 {cd} 수정주가 변경: 엔진 가격 기준과 다름")
    df = df.sort_values(["code", "date"])
    df["prev_close"] = df.groupby("code")["close"].shift(1)
    df["halted"] = df["volume"] == 0
    sim = SimBroker(r)
    for day in lc["dates"]:
        today = df[df["date"] == day].set_index("code")
        bars = {cd: {k: today.at[cd, k] for k in BAR} for cd in today.index}
        fills = sim.run_day(eng, day, {cd: bars[cd] for cd in eng.book if cd in bars})
        t1 = time.perf_counter()
        setups, _ = D.signals(df.drop(columns=["prev_close", "halted"]), day, p)
        t_sig = time.perf_counter() - t1
        need = set(eng.s.positions) | set(eng.book)
        orders = eng.end_of_day(day, {cd: bars[cd] for cd in need if cd in bars}, setups)
        print(f"\n[{day}] 일봉 {len(today)}종목, 체결 {len(fills)}건, 후보 {len(setups)}건, 신호 계산 {t_sig:.1f}s, "
              f"다음 날 주문 의도 {len(orders)}건, 보유 {len(eng.s.positions)}종목")
        for x in fills:
            print("  체결", x.code, x.kind, round(x.price, 2), x.units, x.reason)
        used = df[(df["date"] <= day)][["code", "date", "close", "volume"]]
        data_fp = hashlib.sha256(pd.util.hash_pandas_object(used, index=False).values.tobytes()).hexdigest()[:16]
        log.append("day", {"date": day, "spec": spec_digest[:16], "code_version": code_version(),
                           "data_fp": data_fp, "mode": "paper_sim",
                           "setups": [list(s_) for s_ in setups], "fills": [x.as_dict() for x in fills],
                           "orders": [o.as_dict() for o in orders], "positions": sorted(eng.s.positions)})
    # 주문 의도 -> 증권사 주문 (주식 수·호가). 보유 수량은 1단위 = floor(100만 / 1차 체결가) 주로 가정
    nxt = "2026-11-06"
    last = df[df["date"] == lc["dates"][-1]].set_index("code")["close"]
    plan, rejected = [], []
    for o in orders:
        pos = eng.s.positions.get(o.code)
        q1 = int(c["strategy"]["first_money"] // pos.buys[0][0]) if pos else 0
        held_q = int(round(q1 * pos.units)) if pos else 0
        bo, why = convert(o, float(last.get(o.code, float("nan"))), held_q, q1, c["strategy"]["first_money"])
        (plan.append(bo) if bo else rejected.append((o.code, o.kind, why)))
    invested = sum(b[0] * c["strategy"]["first_money"] / q.buys[0][0] * b[1]
                   for q in eng.s.positions.values() for b in q.buys)
    ok_orders, cash_rej = cash_check(plan, c["strategy"]["capital"] - invested)
    out = LIVE / f"plan_{nxt}.json"
    out.write_text(json.dumps([vars(x) for x in ok_orders], ensure_ascii=False, indent=1), encoding="utf-8")
    log.append("plan", {"for": nxt, "orders": [vars(x) for x in ok_orders], "rejected": rejected,
                        "cash_rejected": [vars(x) for x in cash_rej]})
    eng.save(LIVE / "state_live.json")
    print(f"\n== {nxt} 주문 계획: {len(ok_orders)}건 (거절 {len(rejected)}건 {rejected[:5]}, 현금 부족 {len(cash_rej)}건)")
    for x in ok_orders:
        print(f"  {x.code} {x.kind:13s} {x.side:4s} {x.order_type:6s} {x.price:>10,.0f}원 {x.qty:>5d}주  {x.key}")
    print(f"전체 소요 {time.perf_counter() - t0:.0f}s (기준 {lc['max_minutes']}분), 감사 기록 검증 {verify(LIVE / 'audit.jsonl')}")


def main() -> None:
    global spec_digest
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--truncation", action="store_true")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    c = cfg()
    spec, spec_digest = load(c["strategy"]["spec"])
    if a.coverage:
        coverage(spec)
    if a.truncation:
        truncation(spec, c)
    if a.live:
        live(spec, c)


spec_digest = ""

if __name__ == "__main__":
    main()
