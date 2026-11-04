"""07-2 시장 국면·종목 규모별 B0 성과와 B1(시장 국면 필터).

사용:
    uv run python scripts/eval_07_2.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import bootstrap_mean_ci, equity_curve, trade_stats
from pullback.paths import DATA, UNIVERSE
from pullback.spec import load, to_params
from pullback.theme import add_market_features

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
PERIODS = {"dev": ("2021-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31")}


def in_period(t: pd.DataFrame, name: str) -> pd.DataFrame:
    a, b = PERIODS[name]
    return t[(t["entry_date"] >= a) & (t["entry_date"] <= b)]


def summary(t: pd.DataFrame) -> str:
    s = trade_stats(t["net_ret"])
    return f"n {s['n']:>4}  평균 {s['mean']:+.4f}  중앙 {s['median']:+.4f}  승률 {s['win_rate']:.3f}  PF {s['profit_factor']:.2f}"


def main() -> None:
    spec, digest = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    mk = add_market_features(df)
    df["mkt_gap_ma60"], df["mkt_ret20"], df["market"] = mk["mkt_gap_ma60"], mk["mkt_ret20"], mk["market"]
    g = df.groupby("code")
    df["amt20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["sig_mkt_gap"] = g["mkt_gap_ma60"].shift(1)      # 체결일 전날(신호일) 값
    df["sig_mkt_ret20"] = g["mkt_ret20"].shift(1)
    df["sig_amt20"] = g["amt20"].shift(1)
    r = Rules.from_spec(spec)
    trades, events = run(df, f["setup"], r)
    key = df.set_index(["code", "date"])
    att = key.loc[list(zip(trades["code"], trades["entry_date"])), ["sig_mkt_gap", "sig_mkt_ret20", "sig_amt20", "market"]]
    trades = pd.concat([trades, att.reset_index(drop=True)], axis=1)
    status = pd.read_parquet(UNIVERSE).set_index("code")["status"]
    trades["delisted"] = trades["code"].map(status).eq("delisted")
    th = pd.read_parquet(DATA / "features_b0_theme.parquet")
    sig_date = g["date"].shift(1)
    sd = df.assign(sig_date=sig_date).set_index(["code", "date"])["sig_date"]
    trades["sig_date"] = sd.loc[list(zip(trades["code"], trades["entry_date"]))].to_numpy()
    trades = trades.merge(th[["code", "date", "theme_n_peers", "theme_rank_amt"]].rename(columns={"date": "sig_date"}),
                          on=["code", "sig_date"], how="left")
    trades["month"] = trades["entry_date"].dt.to_period("M")
    q = trades["sig_amt20"].quantile([1 / 3, 2 / 3]).to_numpy()
    trades["size"] = pd.cut(trades["sig_amt20"], [-1, q[0], q[1], float("inf")], labels=["소", "중", "대"])
    print(f"spec {digest[:16]}, 거래대금 규모 경계(20일 평균) {q[0] / 1e8:,.0f}억 / {q[1] / 1e8:,.0f}억")

    groups = {
        "시장 지수 vs 60일선": trades["sig_mkt_gap"].ge(0).map({True: "지수>=60일선", False: "지수<60일선"}),
        "지수 20일 수익률": trades["sig_mkt_ret20"].ge(0).map({True: "+", False: "-"}),
        "시장": trades["market"],
        "거래대금 규모": trades["size"].astype(str),
        "갭 체결": trades["gap_fill"].map({True: "갭", False: "일반"}),
        "테마 대장주": trades["theme_rank_amt"].eq(1) & trades["theme_n_peers"].ge(1),
        "이후 상장폐지": trades["delisted"],
    }
    for name in PERIODS:
        t = in_period(trades, name)
        print(f"\n== {name} B0 전체: {summary(t)}")
        for gname, lab in groups.items():
            print(f"  [{gname}]")
            for k, x in t.groupby(lab.loc[t.index]):
                print(f"    {str(k):<12} {summary(x)}")

    print("\n== B1: 지수 >= 60일선일 때만 1차 주문 (B1 v1.0) 과 대안")
    variants = {"B0": f["setup"], "B1 (지수>=60일선)": f["setup"] & df["mkt_gap_ma60"].ge(0),
                "대안 (지수 20일 수익률>=0)": f["setup"] & df["mkt_ret20"].ge(0)}
    for label, mask in variants.items():
        tt, ee = run(df, mask, r)
        tt["month"] = tt["entry_date"].dt.to_period("M")
        for name in PERIODS:
            x = in_period(tt, name)
            lo, hi = bootstrap_mean_ci(x["net_ret"], x["month"])
            eq = equity_curve(ee, x, df, r, unit_money=1e6, capital=1e8)
            print(f"  {label:<20} {name:<10} {summary(x)}  CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  계좌 {eq['equity'].iloc[-1] / 1e8 - 1:+.4f} MDD {eq['dd'].min():.4f}")


if __name__ == "__main__":
    main()
