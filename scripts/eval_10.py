"""10장 A2(B0 + LLM 공시 악재 필터) 평가.

사용:
    uv run python -m pullback.collect.dart
    uv run python -m pullback.disclosure
    uv run python scripts/eval_10.py                  # 10-1 공시 구조화 통계, 필터 적용 건수 (수익률 없음)
    uv run python scripts/eval_10.py --show-returns   # 10-2·10-3 매매 성과와 채택 판정
    uv run python scripts/eval_10.py --lgbm           # 10-3 LightGBM 결합 진단 (AUC)
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.disclosure import DISC_FEATURES, LABELS, events, order_day_features
from pullback.metrics import bootstrap_mean_ci, equity_curve
from pullback.paths import DATA
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
PER = {"dev": ("2021-01-01", "2024-12-31"), "val": ("2025-01-01", "2025-12-31")}


def describe(ev: pd.DataFrame, lab: pd.DataFrame) -> None:
    in_range = ev["date"].between("2020-01-01", "2025-12-31")
    e = ev[in_range]
    print(f"== 10-1 공시 (2020~2025 접수)\n  공시 {len(e):,}건, 종목 {e['code'].nunique():,}개, 정정 제외 사건 {e['is_event'].sum():,}건"
          f" ({e['is_event'].mean():.1%}), 정규화 제목 {e['title'].nunique():,}개")
    x = e[e["is_event"]]
    print("  사건 기준 LLM impact:", x["impact"].value_counts(normalize=True).round(3).to_dict())
    print("  사건 기준 material 비율:", round(x["material"].astype(bool).mean(), 3))
    print("  사건 기준 범주 상위:", x["category"].value_counts(normalize=True).round(3).head(12).to_dict())
    print("  제목 기준 LLM impact:", lab["impact"].value_counts().to_dict(), "모델", lab["model"].unique().tolist(),
          "프롬프트", lab["prompt"].unique().tolist())
    ct = pd.crosstab(x["impact"], x["keyword_impact"], margins=True)
    print("  LLM(행) x 키워드(열), 사건 수\n" + ct.to_string())
    print(f"  LLM-키워드 일치율(사건): {(x['impact'] == x['keyword_impact']).mean():.3f}")
    for imp in ["악재", "호재"]:
        top = x[x["impact"] == imp]["title"].value_counts().head(12)
        print(f"  LLM '{imp}' 상위 제목:", top.to_dict())
    dis = x[(x["impact"] == "악재") & (x["keyword_impact"] != "악재")]["title"].value_counts().head(8)
    print("  LLM 악재 & 키워드 아님:", dis.to_dict())
    dis = x[(x["impact"] != "악재") & (x["keyword_impact"] == "악재")]["title"].value_counts().head(8)
    print("  키워드 악재 & LLM 아님:", dis.to_dict())


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
        return f"  {label:<14} 거래 0"
    return (f"  {label:<14} 거래 {s['n']:>4}  평균 {s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]  금액가중 {s['mw']:+.4f}"
            f"  계좌 {s['pnl']:+.4f}  MDD {s['mdd']:+.4f}")


def signal_value(trades, df, col):
    """거래의 신호일(체결 전날) 값."""
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    sig = pd.Series(df.groupby("code")[col].shift(1).to_numpy(), index=key)
    return sig.reindex(pd.MultiIndex.from_frame(trades[["code", "entry_date"]])).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-returns", action="store_true")
    ap.add_argument("--lgbm", action="store_true")
    a = ap.parse_args()
    a2 = tomllib.loads(open("configs/a2.toml", encoding="utf-8").read())
    ev, lab = events(), pd.read_parquet(LABELS)
    if not (a.show_returns or a.lgbm):
        describe(ev, lab)

    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    in_eval = df["date"] <= "2025-12-31"          # 07장 B0 와 같이 2020 워밍업 신호부터 (2026 신호는 열지 않음)
    setup = f["setup"] & in_eval
    feats = order_day_features(df, setup, ev)
    feats_kw = order_day_features(df, setup, ev, impact_col="keyword_impact")
    for c in DISC_FEATURES:
        df[c] = feats[c].reindex(df.index)
    df["bad_kw"] = feats_kw["bad_since_peak"].reindex(df.index)

    if a.lgbm:
        return lgbm_check(df)

    print("\n== 필터 적용 건수 (B0 주문일 기준)")
    for name, (s, e) in PER.items():
        m = setup & df["date"].between(s, e)
        print(f"  [{name}] 주문일 {m.sum():,}  LLM 악재로 막힌 날 {df.loc[m, 'bad_since_peak'].mean():.3f}"
              f"  키워드 {df.loc[m, 'bad_kw'].mean():.3f}  상승 구간 재료 {df.loc[m, 'has_material_rise'].mean():.3f}"
              f"  새 재료 {df.loc[m, 'fresh_material'].mean():.3f}")
    base = Rules.from_spec(spec)
    b0_t, b0_e = run(df, setup, base)
    b0_t["sig_bad"] = signal_value(b0_t, df, "bad_since_peak")
    b0_t["sig_kw"] = signal_value(b0_t, df, "bad_kw")
    for name, (s, e) in PER.items():
        x = b0_t[b0_t["entry_date"].between(s, e)]
        print(f"  [{name}] B0 거래 {len(x)}  LLM 필터가 걸렀을 거래 {int((x['sig_bad'] == 1).sum())}"
              f"  키워드 {int((x['sig_kw'] == 1).sum())}")
    if not a.show_returns:
        print("\n(수익률은 --show-returns 로 확인)")
        return

    results = {}
    for cost_label, r in [("기본 비용", base), ("슬리피지 0.3%", replace(base, slippage=0.003))]:
        print(f"\n== {cost_label}")
        bt, be = run(df, setup, r)
        at, ae = run(df, setup & (df["bad_since_peak"] != 1), r)
        kt, ke = run(df, setup & (df["bad_kw"] != 1), r)
        bt["sig_bad"] = signal_value(bt, df, "bad_since_peak")
        bt["sig_kw"] = signal_value(bt, df, "bad_kw")
        for name, (s, e) in PER.items():
            sb, sa, sk = (evaluate(t, ev_, df, r, s, e) for t, ev_ in [(bt, be), (at, ae), (kt, ke)])
            results[(cost_label, name)] = (sb, sa)
            print(f" [{name} {s[:4]}~{e[:4]}]")
            print(line("B0", sb))
            print(line("A2 (LLM)", sa))
            print(line("키워드 필터", sk))
            x = bt[bt["entry_date"].between(s, e)]
            for col, lbl in [("sig_bad", "LLM"), ("sig_kw", "키워드")]:
                kept, drop = x[x[col] != 1], x[x[col] == 1]
                if lbl == "LLM":
                    results[(cost_label, name, "filter")] = (kept["net_ret"].mean(), drop["net_ret"].mean(), len(drop))
                print(f"  {lbl} 남김 {len(kept)}건 평균 {kept['net_ret'].mean():+.4f} | 걸러냄 {len(drop)}건 평균 {drop['net_ret'].mean():+.4f}")
        if cost_label == "기본 비용":
            print("\n== 10-2 재료 신선도 (B0 거래, 신호일 기준)")
            for col in ["has_material_rise", "fresh_material"]:
                bt[col] = signal_value(bt, df, col)
                for name, (s, e) in PER.items():
                    x = bt[bt["entry_date"].between(s, e)]
                    g = x.groupby(x[col].fillna(0))["net_ret"].agg(["size", "mean", lambda v: (v > 0).mean()])
                    print(f"  {col} [{name}]", {k: [int(v["size"]), round(v["mean"], 4), round(v["<lambda_0>"], 3)] for k, v in g.iterrows()})

    print("\n== 채택 기준 (configs/a2.toml [adoption])")
    sb_d, sa_d = results[("기본 비용", "dev")]
    sb_v, sa_v = results[("기본 비용", "val")]
    ssb_d, ss_d = results[("슬리피지 0.3%", "dev")]
    ssb_v, ss_v = results[("슬리피지 0.3%", "val")]
    fd, fv = results[("기본 비용", "dev", "filter")], results[("기본 비용", "val", "filter")]
    g = lambda s, k: s.get(k, float("nan"))
    checks = {
        "2021~2024 계좌 손익 > 0": g(sa_d, "pnl") > 0,
        "2021~2024 금액 가중 > 0": g(sa_d, "mw") > 0,
        "2025 거래당 평균 > B0": g(sa_v, "mean") > g(sb_v, "mean"),
        "2025 MDD >= B0": g(sa_v, "mdd") >= g(sb_v, "mdd"),
        "슬리피지 0.3% 유지": g(ss_d, "pnl") > 0 and g(ss_d, "mw") > 0 and g(ss_v, "mean") > g(ssb_v, "mean")
        and g(ss_v, "mdd") >= g(ssb_v, "mdd"),
        "걸러낸 거래 < 남긴 거래 (2021~2024)": fd[1] < fd[0],
        "걸러낸 거래 < 남긴 거래 (2025)": fv[1] < fv[0],
        "2021~2024 거래 >= 100": sa_d.get("n", 0) >= a2["adoption"]["min_trades"],
        "2021~2024 걸러낸 거래 >= 30": fd[2] >= a2["adoption"]["min_filtered"],
    }
    for k, v in checks.items():
        print(f"  {'통과' if v else '실패'}  {k}")
    print("  판정:", "A2 채택" if all(checks.values()) else "A2 채택하지 않음 (H7 기각)")


def lgbm_check(df: pd.DataFrame) -> None:
    """09장 Walk-forward 에 공시 특징량을 더했을 때 AUC 변화 (진단만)."""
    from pullback.dataset import ALL_FEATURES
    from pullback.model import fit, train_set
    ds = pd.read_parquet(DATA / "dataset_a1.parquet")
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    for c in DISC_FEATURES:
        s = pd.Series(df[c].to_numpy(), index=key)
        ds[c] = s.reindex(pd.MultiIndex.from_frame(ds[["code", "date"]])).to_numpy()
    print("== 10-3 LightGBM 결합 (09장과 같은 분할, y = +20% 먼저)")
    print("  학습 표본 공시 특징량 평균:", ds[DISC_FEATURES].mean().round(3).to_dict())
    pooled = {"A1": ([], []), "A1+공시": ([], [])}
    for last, pred in [("2021", "2022"), ("2022", "2023"), ("2023", "2024"), ("2024", "2025")]:
        tr = train_set(ds, last, pred)
        te = ds[(pd.to_datetime(ds["date"]).dt.year == int(pred)) & ds["y"].notna()]
        yt = (te["y"] == 1).astype(int)
        msg = []
        for name, feats in [("A1", ALL_FEATURES), ("A1+공시", ALL_FEATURES + DISC_FEATURES)]:
            m = fit(tr, "lgbm", feats)
            pr = m.predict_raw(te)
            msg.append(f"{name} {roc_auc_score(yt, pr):.3f}")
            if pred != "2025":
                pooled[name][0].extend(yt)
                pooled[name][1].extend(pr)
        print(f"  {pred}: " + " / ".join(msg))
    for name, (y, pr) in pooled.items():
        print(f"  2022~2024 합산 {name} AUC {roc_auc_score(y, pr):.3f}")
    # 공시 특징량 단독의 방향: 학습 표본 전체에서 y 와의 관계
    d = ds[pd.to_datetime(ds["date"]).dt.year.between(2021, 2024) & ds["y"].notna()]
    for c in ["bad_since_peak", "has_material_rise", "fresh_material"]:
        g = d.groupby(d[c].fillna(0))["y"].agg(["size", lambda v: (v == 1).mean()])
        print(f"  {c}: " + ", ".join(f"{int(k)} -> 표본 {int(v['size'])} 성공률 {v['<lambda_0>']:.3f}" for k, v in g.iterrows()))


if __name__ == "__main__":
    main()
