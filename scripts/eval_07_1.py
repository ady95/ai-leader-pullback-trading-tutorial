"""07-1 B0 성과: 거래 지표, 신뢰구간, 연도별, 청산 사유, MAE·MFE, 계좌 MDD, 비용 민감도.

사용:
    uv run python scripts/eval_07_1.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import bootstrap_mean_ci, equity_curve, path_stats, trade_stats
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
PERIODS = {"dev": ("2021-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31")}


def fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items())


def main() -> None:
    spec, digest = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    r = Rules.from_spec(spec)
    trades, events = run(df, f["setup"], r)
    trades["month"] = trades["entry_date"].dt.to_period("M")
    print("spec", digest[:16], "거래", len(trades))
    for name, (a, b) in PERIODS.items():
        t = trades[(trades["entry_date"] >= a) & (trades["entry_date"] <= b)]
        lo, hi = bootstrap_mean_ci(t["net_ret"], t["month"])
        print(f"\n== {name} ({a} ~ {b})")
        print(fmt(trade_stats(t["net_ret"])), f"| 평균 95% CI [{lo:.4f}, {hi:.4f}]")
        print("총수익(gross) 평균", round(float(t["gross_ret"].mean()), 4))
        print("청산 사유", t["exit_reason"].value_counts().to_dict(),
              "평균", t.groupby("exit_reason")["net_ret"].mean().round(4).to_dict())
        if name == "dev":
            yr = t.groupby(t["entry_date"].dt.year)["net_ret"].agg(["size", "mean", "median", lambda s: (s > 0).mean()])
            yr.columns = ["n", "mean", "median", "win"]
            print("연도별\n" + yr.round(4).to_string())
        ps = path_stats(t, df)
        print("MAE 분위", ps["mae"].quantile([.1, .25, .5, .75]).round(3).to_dict(),
              "MFE 분위", ps["mfe"].quantile([.25, .5, .75, .9]).round(3).to_dict())
        eq = equity_curve(events, t, df, r, unit_money=1e6, capital=1e8)
        days = pd.Index(sorted(df["date"].unique()))
        conc = pd.Series(0, index=days[(days >= t["entry_date"].min()) & (days <= t["exit_date"].max())])
        for e, x in zip(t["entry_date"], t["exit_date"]):
            conc.loc[e:x] += 1
        print(f"계좌(1억, 1차 매수 100만 원 고정): 최종 손익 {eq['equity'].iloc[-1] - 1e8:,.0f}원"
              f" ({eq['equity'].iloc[-1] / 1e8 - 1:+.4f}), MDD {eq['dd'].min():.4f},"
              f" 최대 노출 {eq['exposure'].max():,.0f}원, 동시 보유 최대 {conc.max()} 중앙 {conc.median():.0f}")

    print("\n== 비용·체결 민감도 (거래당 평균 순수익률)")
    for label, over in [("기본", {}), ("슬리피지 0.3%", {"slippage": 0.003}), ("엄격 체결", {"strict_touch": True}),
                        ("비용 0 (총수익)", {"slippage": 0.0, "commission": 0.0, "sell_tax_by_year": {}, "sell_tax_default": 0.0})]:
        tt, _ = run(df, f["setup"], Rules.from_spec(spec, **over))
        row = []
        for name, (a, b) in PERIODS.items():
            x = tt[(tt["entry_date"] >= a) & (tt["entry_date"] <= b)]["net_ret"]
            row.append(f"{name} {x.mean():+.4f} (n={len(x)})")
        print(f"  {label:<14}", " | ".join(row))


if __name__ == "__main__":
    main()
