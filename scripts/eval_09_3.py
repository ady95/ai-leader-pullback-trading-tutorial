"""09-3 A1(B0 + LightGBM 필터) Walk-forward 매매 평가와 채택 판정.

사용:
    uv run python scripts/train_09.py      # 먼저 예측값 생성
    uv run python scripts/eval_09_3.py
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import replace

import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import bootstrap_mean_ci, equity_curve
from pullback.paths import DATA
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
PER = {"oos": ("2022-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-12-31")}


def evaluate(t, e, df, r, a, b):
    x = t[(t["entry_date"] >= a) & (t["entry_date"] <= b)]
    if x.empty:
        return dict(n=0)
    money = x["invested"] * 1e6 / x["entry_price"]
    eq = equity_curve(e, x, df, r, unit_money=1e6, capital=1e8)
    lo, hi = bootstrap_mean_ci(x["net_ret"], x["entry_date"].dt.to_period("M"))
    return dict(n=len(x), mean=x["net_ret"].mean(), lo=lo, hi=hi, mw=float((x["net_ret"] * money).sum() / money.sum()),
                pnl=eq["equity"].iloc[-1] / 1e8 - 1, mdd=eq["dd"].min())


def line(label, s):
    if s["n"] == 0:
        return f"  {label:<26} 거래 0"
    return (f"  {label:<26} 거래 {s['n']:>4}  평균 {s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]  금액가중 {s['mw']:+.4f}"
            f"  계좌 {s['pnl']:+.4f}  MDD {s['mdd']:+.4f}")


def main() -> None:
    a1 = tomllib.loads(open("configs/a1.toml", encoding="utf-8").read())
    th = a1["decision"]["threshold"]
    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    sc = pd.read_parquet(DATA / "results" / "a1_scores.parquet")
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    score = pd.Series(sc["p"].to_numpy(), index=pd.MultiIndex.from_frame(sc[["code", "date"]]))
    df["p"] = score.reindex(key).to_numpy()
    in_pred = df["date"].between("2022-01-01", "2025-12-31")
    base = Rules.from_spec(spec)
    results = {}
    for cost_label, r in [("기본 비용", base), ("슬리피지 0.3%", replace(base, slippage=0.003))]:
        print(f"\n== {cost_label}")
        b0_t, b0_e = run(df, f["setup"] & in_pred, r)
        a1_t, a1_e = run(df, f["setup"] & in_pred & (df["p"] >= th), r)
        for name, (a, b) in PER.items():
            sb, sa = evaluate(b0_t, b0_e, df, r, a, b), evaluate(a1_t, a1_e, df, r, a, b)
            results[(cost_label, name)] = (sb, sa)
            print(f" [{name} {a[:4]}~{b[:4]}]")
            print(line("B0", sb))
            print(line(f"A1 (p >= {th})", sa))
        # 걸러낸 거래: B0 거래 중 신호일 점수가 임계값 미만
        sig = df.groupby("code")["p"].shift(1)
        b0_t["sig_p"] = pd.Series(sig.to_numpy(), index=key).reindex(
            pd.MultiIndex.from_frame(b0_t[["code", "entry_date"]])).to_numpy()
        for name, (a, b) in PER.items():
            x = b0_t[(b0_t["entry_date"] >= a) & (b0_t["entry_date"] <= b)]
            kept, drop = x[x["sig_p"] >= th], x[x["sig_p"] < th]
            results[(cost_label, name, "filter")] = (kept["net_ret"].mean(), drop["net_ret"].mean())
            print(f"  B0 거래 중 [{name}] 남김 {len(kept)}건 평균 {kept['net_ret'].mean():+.4f} | 걸러냄 {len(drop)}건 평균 {drop['net_ret'].mean():+.4f}")
        if cost_label == "기본 비용":
            print("  A1 거래 연도별", a1_t["entry_date"].dt.year.value_counts().sort_index().to_dict(),
                  "평균", a1_t.groupby(a1_t["entry_date"].dt.year)["net_ret"].mean().round(4).to_dict())

    print("\n== 채택 기준 (configs/a1.toml [adoption])")
    sb_o, sa_o = results[("기본 비용", "oos")]
    sb_v, sa_v = results[("기본 비용", "val")]
    _, ss_o = results[("슬리피지 0.3%", "oos")]
    _, ss_v = results[("슬리피지 0.3%", "val")]
    get = lambda s, k: s.get(k, float("nan"))
    checks = {
        "2022~2024 계좌 손익 > 0": get(sa_o, "pnl") > 0,
        "2022~2024 금액 가중 > 0": get(sa_o, "mw") > 0,
        "2025 거래당 평균 > B0": get(sa_v, "mean") > get(sb_v, "mean"),
        "2025 MDD >= B0": get(sa_v, "mdd") >= get(sb_v, "mdd"),
        "슬리피지 0.3% 유지": get(ss_o, "pnl") > 0 and get(ss_o, "mw") > 0 and get(ss_v, "mean") > get(results[("슬리피지 0.3%", "val")][0], "mean"),
        "2022~2024 거래 >= 100": sa_o.get("n", 0) >= a1["adoption"]["min_trades"],
        "걸러낸 거래 < 남긴 거래 (2022~2024)": results[("기본 비용", "oos", "filter")][1] < results[("기본 비용", "oos", "filter")][0],
        "걸러낸 거래 < 남긴 거래 (2025)": results[("기본 비용", "val", "filter")][1] < results[("기본 비용", "val", "filter")][0],
    }
    for k, v in checks.items():
        print(f"  {'통과' if v else '실패'}  {k}")
    print("  판정:", "A1 채택" if all(checks.values()) else "A1 채택하지 않음 (H6 기각)")


if __name__ == "__main__":
    main()
