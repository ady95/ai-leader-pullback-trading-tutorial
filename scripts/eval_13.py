"""13장 A3 청산 구성요소 (X1 고정 목표, X2 낮은 1차 익절, X3 ATR 손절, X4 트레일링, T1 시간 청산, T2 무진전 청산).

사용:
    uv run python scripts/eval_13.py              # 성과, 진단, 무작위 기준, 채택 판정
    uv run python scripts/eval_13.py --x3-draws 30   # X3 무작위 대조 횟수를 줄여 빠르게 확인
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_12 import CAPITAL, UNIT, account, in_per, money, pct_rank, rodc, signal_index  # noqa: E402

from pullback.backtest import Rules, _pnl, run  # noqa: E402
from pullback.definitions import build, flags, load_daily  # noqa: E402
from pullback.spec import load, to_params  # noqa: E402

SPEC = tomllib.loads(open("configs/a3_exit.toml", encoding="utf-8").read())
PER = {k: tuple(v) for k, v in SPEC["metrics"]["periods"].items()}
NAMES = ["X1", "X2", "X3", "X4", "T1", "T2"]


def variants(r: Rules) -> dict[str, Rules]:
    return {"B0": r,
            "X1": replace(r, tp_mode="fixed", tp_fixed=0.25),
            "X2": replace(r, take_profit_1=0.10),
            "X3": replace(r, stop_mode="atr"),
            "X4": replace(r, trail_on=0.10),
            "T1": replace(r, max_hold_days=20),
            "T2": replace(r, noprog_day=10)}


class Paths:
    """종목별 일봉 배열. 거래의 진입·청산 위치를 찾는다."""

    def __init__(self, df: pd.DataFrame):
        self.by = {}
        for code, x in df[["code", "date", "open", "high", "halted"]].groupby("code", sort=False):
            x = x.sort_values("date")
            self.by[code] = (x["date"].to_numpy(), x["open"].to_numpy(float), x["high"].to_numpy(float),
                             x["halted"].to_numpy(bool))

    def idx(self, code, date) -> int:
        d = self.by[code][0]
        return int(np.searchsorted(d, np.datetime64(date)))


def mfe(t: pd.DataFrame, e: pd.DataFrame, P: Paths) -> np.ndarray:
    """보유 중 최대 유리 변동: 1차 체결 다음 날 ~ 청산 전날 고가와 매도 체결가 중 최고 / 최종 평균 단가 - 1."""
    out = np.full(len(t), np.nan)
    eg = {k: g for k, g in e.groupby("code")}
    for k, row in enumerate(t.itertuples()):
        d, o, h, _ = P.by[row.code]
        a, b = P.idx(row.code, row.entry_date), P.idx(row.code, row.exit_date)
        g = eg[row.code]
        g = g[(g["date"] >= row.entry_date) & (g["date"] <= row.exit_date)]
        buys = g[g["kind"].str.startswith("buy")]
        avg = (buys["price"] * buys["units"]).sum() / buys["units"].sum()
        hi = h[a + 1:b].max() if b > a + 1 else -np.inf
        sp = g.loc[g["kind"] == "sell", "price"].max()
        out[k] = max(hi, sp) / avg - 1
    return out


def truncations(xd: pd.DataFrame, e: pd.DataFrame, P: Paths, r: Rules) -> dict:
    """B0 거래마다 '장 마감 j일째(진입일 = 0)에 청산 결정, 다음 날 시가 전량 매도' 했을 때의 (net, money, hold)."""
    out = {}
    eg = {k: g for k, g in e.groupby("code")}
    for k, row in enumerate(xd.itertuples()):
        d, o, _, halted = P.by[row.code]
        a = P.idx(row.code, row.entry_date)
        g = eg[row.code]
        g = g[(g["date"] >= row.entry_date) & (g["date"] <= row.exit_date)]
        for j in range(int(row.hold_days) - 1):
            cut = d[a + j]
            buys = [(p, u, dt) for p, u, dt, kd in zip(g["price"], g["units"], g["date"], g["kind"])
                    if kd.startswith("buy") and dt <= cut]
            sells = [(p, u, dt) for p, u, dt, kd in zip(g["price"], g["units"], g["date"], g["kind"])
                     if kd == "sell" and dt <= cut]
            rem = sum(u for _, u, _ in buys) - sum(u for _, u, _ in sells)
            x = a + j + 1
            while x < len(d) - 1 and halted[x]:
                x += 1
            if rem > 1e-12:
                sells = sells + [(o[x], rem, d[x])]
            inv, _, net = _pnl(buys, sells, r)
            out[(k, j)] = (net, inv * UNIT / row.entry_price, x - a)
    return out


def timing_benchmark(xd, cand, trunc, rng, draws, frac=1.0, hold_col="hold_days"):
    """결과가 바뀐 거래 수만큼 무작위로 골라, 바뀐 거래들의 청산(또는 절반 매도) 시점을 섞어 준다."""
    key = ["code", "entry_date"]
    c = pd.DataFrame({"code": cand["code"], "entry_date": cand["entry_date"], "net_c": cand["net_ret"],
                      "hold_c": cand["hold_days"], "h_c": cand[hold_col]})
    m = xd[key + ["net_ret", "hold_days"]].merge(c, on=key, how="inner")
    changed = (m["net_ret"] - m["net_c"]).abs().gt(1e-9) | (m["hold_days"] != m["hold_c"])
    hs = m.loc[changed, "h_c"].to_numpy(int)
    n = len(hs)
    base_net = xd["net_ret"].to_numpy()
    base_m = money(xd).to_numpy()
    base_h = xd["hold_days"].clip(lower=1).to_numpy(float)
    dist = []
    for _ in range(draws):
        net, mm, hh = base_net.copy(), base_m.copy(), base_h.copy()
        pnl, md = net * mm, mm * hh
        for k, h in zip(rng.choice(len(xd), n, replace=False), rng.permutation(hs)):
            j = h - 1
            if (k, j) not in trunc:
                continue
            tn, tm, th = trunc[(k, j)]
            pnl[k] = (1 - frac) * pnl[k] + frac * tn * tm
            md[k] = (1 - frac) * md[k] + frac * tm * max(th, 1)
        dist.append(pnl.sum() / md.sum() * 250)
    return n, dist


def report_costs(t: pd.DataFrame, r: Rules) -> tuple[float, float]:
    """거래 비용 합(원)과 연간 매도 체결 기준 회전 금액."""
    m = money(t)
    sell_tax = t["exit_date"].dt.year.map(r.tax)
    cost = m * (r.commission + r.slippage) + m * (1 + t["gross_ret"]) * (r.commission + r.slippage + sell_tax)
    return float(cost.sum()), float(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x3-draws", type=int, default=None)
    a = ap.parse_args()
    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    setup = f["setup"] & (df["date"] <= "2025-12-31")
    base = Rules.from_spec(spec)
    P = Paths(df)
    rng = np.random.default_rng(SPEC["random_benchmark"]["seed"])
    draws = 2000
    x3_draws = a.x3_draws or 300
    verdict = {}

    for cost_label, r in [("기본 비용", base), ("슬리피지 0.3%", replace(base, slippage=0.003))]:
        print(f"\n######## {cost_label}")
        runs = {k: run(df, setup, v) for k, v in variants(r).items()}
        runs = {k: (t.sort_values(["entry_date", "code"]).reset_index(drop=True), e) for k, (t, e) in runs.items()}
        res = {}
        for name, (t, e) in runs.items():
            for per in PER:
                res[(name, per)] = account(in_per(t, per), e, df, r)
        for per in PER:
            print(f"\n [{per}]")
            for name in runs:
                s = res[(name, per)]
                print(f"  {name:<3} 거래 {s['n']:>4}  평균 {s['mean']:+.4f}  투입연환산 {s['rodc']:+.4f}  계좌 {s['pnl']:+.4f}"
                      f"  MDD {s['mdd']:+.4f}  평균투입 {s['exp'] / 1e6:6.1f}백만  MDD/투입 {s['mdd_exp']:+.3f}"
                      f"  최악 {s['worst']:+.4f} ({s['worst_won'] / 1e4:,.0f}만)")
        verdict[cost_label] = res
        if cost_label != "기본 비용":
            continue

        print("\n== 진단 (기본 비용)")
        for per in PER:
            print(f" [{per}]")
            for name, (t, e) in runs.items():
                x = in_per(t, per)
                reasons = x["exit_reason"].value_counts().to_dict()
                cost, turn = report_costs(x, r)
                yrs = 4 if per == "dev" else 1
                print(f"  {name:<3} 보유 평균 {x['hold_days'].mean():5.1f}일 중앙 {x['hold_days'].median():4.1f}"
                      f"  연간 매수 금액 {turn / yrs / 1e8:5.2f}억  연간 비용 {cost / yrs / 1e4:6.0f}만"
                      f"  청산 사유 {reasons}")
        print("\n== MFE (B0, 보유 중 최대 유리 변동 / 최종 평균 단가)")
        b0_t, b0_e = runs["B0"]
        b0_t["mfe"] = mfe(b0_t, b0_e, P)
        for per in PER:
            x = in_per(b0_t, per)
            print(f" [{per}] 거래 {len(x)}  MFE 중앙 {x['mfe'].median():+.4f}  >=10% {(x['mfe'] >= .10).mean():.3f}"
                  f"  >=20% {(x['mfe'] >= .20).mean():.3f}  >=25% {(x['mfe'] >= .25).mean():.3f}")
            for lo_, hi_ in [(-1, .05), (.05, .10), (.10, .20), (.20, .30), (.30, 9)]:
                y = x[(x["mfe"] >= lo_) & (x["mfe"] < hi_)]
                if len(y):
                    print(f"    MFE [{lo_:+.2f},{hi_:+.2f}) {len(y):>4}건  실현(세전) 평균 {y['gross_ret'].mean():+.4f}"
                          f"  실현/MFE 중앙 {(y['gross_ret'] / y['mfe']).median() if lo_ >= .05 else float('nan'):+.3f}"
                          f"  손절 비율 {(y['exit_reason'] == 'stop').mean():.3f}")
        print(" [dev] 청산 사유별 MFE 중앙:", in_per(b0_t, "dev").groupby("exit_reason")["mfe"].median().round(4).to_dict())

        print(f"\n== 무작위 기준 (개발 구간 투입 연환산, 시점 {draws}회 / X3 {x3_draws}회)")
        xd = in_per(b0_t, "dev").reset_index(drop=True)
        trunc = truncations(xd, b0_e, P, r)
        rand = {}
        for name in ["X1", "X4", "T1", "T2"]:
            n, rand[name] = timing_benchmark(xd, in_per(runs[name][0], "dev"), trunc, rng, draws)
            print(f"  {name}: 결과가 바뀐 거래 {n}건")
        # X2: 1차 익절(+10%) 매도일을 '절반 매도' 시점으로
        x2_t, x2_e = runs["X2"]
        tp1 = x2_e[x2_e["reason"] == "tp1"][["code", "date"]].rename(columns={"date": "tp1_date"})
        x2d = in_per(x2_t, "dev").merge(tp1, on="code", how="left")
        x2d = x2d[x2d["tp1_date"].isna() | x2d["tp1_date"].between(x2d["entry_date"], x2d["exit_date"])]
        x2d = x2d.sort_values("tp1_date").drop_duplicates(["code", "entry_date"]).copy()
        ent = np.array([P.idx(c, d) for c, d in zip(x2d["code"], x2d["entry_date"])])
        tpi = np.array([P.idx(c, d) if pd.notna(d) else -1 for c, d in zip(x2d["code"], x2d["tp1_date"])])
        x2d["h_tp1"] = np.where(tpi >= 0, tpi - ent, 0)
        x2d = x2d[x2d["h_tp1"] > 0]
        n, rand["X2"] = timing_benchmark(xd, x2d.assign(hold_days=-1), trunc, rng, draws, frac=0.5, hold_col="h_tp1")
        print(f"  X2: +10% 절반 매도가 일어난 거래 {n}건 (절반 매도 근사)")
        # X3: atr_pct 를 B0 거래끼리 섞어 손절선을 다시 정한다
        sig = signal_index(b0_t, df).astype(int)
        dev_sig = sig[b0_t["entry_date"].between(*PER["dev"]).to_numpy()]
        atr0 = df["atr"].copy()
        pct = (df["atr"] / df["close"]).to_numpy()
        out = []
        r3 = variants(r)["X3"]
        for _ in range(x3_draws):
            df["atr"] = atr0
            df.loc[dev_sig, "atr"] = rng.permutation(pct[dev_sig]) * df.loc[dev_sig, "close"].to_numpy()
            out.append(rodc(in_per(run(df, setup, r3)[0], "dev")))
        df["atr"] = atr0
        rand["X3"] = out
        for name in NAMES:
            v = res[(name, "dev")]["rodc"]
            dist = rand[name]
            print(f"  {name}: 실제 {v:+.4f}, B0 {res[('B0', 'dev')]['rodc']:+.4f}, 무작위 중앙 {np.median(dist):+.4f},"
                  f" 95백분위 {np.percentile(dist, 95):+.4f}, 백분위 {pct_rank(v, dist):.3f}")
            verdict[("p95", name)] = v > np.percentile(dist, 95)

    print("\n== 채택 판정 (configs/a3_exit.toml [adoption])")
    r0, r3 = verdict["기본 비용"], verdict["슬리피지 0.3%"]
    adopted = []
    for name in NAMES:
        checks = {
            "개발 투입연환산 > B0": r0[(name, "dev")]["rodc"] > r0[("B0", "dev")]["rodc"],
            "개발 MDD/투입 >= B0": r0[(name, "dev")]["mdd_exp"] >= r0[("B0", "dev")]["mdd_exp"],
            "2025 투입연환산 >= B0": r0[(name, "validation")]["rodc"] >= r0[("B0", "validation")]["rodc"],
            "무작위 95백분위 초과": verdict[("p95", name)],
            "슬리피지 0.3% 유지": r3[(name, "dev")]["rodc"] > r3[("B0", "dev")]["rodc"]
            and r3[(name, "validation")]["rodc"] >= r3[("B0", "validation")]["rodc"],
        }
        ok = all(checks.values())
        seen = " (07-3 에서 결과를 본 구성요소)" if name in SPEC["meta"]["seen_components"] else ""
        print(f"  {name}: " + ", ".join(f"{k} {'통과' if v else '실패'}" for k, v in checks.items())
              + f" -> {'채택' if ok else '채택하지 않음'}{seen}")
        if ok:
            adopted.append(name)
    print("  채택:", adopted or "없음")


if __name__ == "__main__":
    main()
