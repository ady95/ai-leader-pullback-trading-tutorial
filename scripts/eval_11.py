"""11장 B0·B1·A1·A2 동일 조건 비교, 비용 민감도, 무작위 필터 기준.

사용:
    uv run python scripts/train_09.py          # A1 점수 (data/results/a1_scores.parquet)
    uv run python -m pullback.disclosure       # A2 공시 분류
    uv run python scripts/eval_11.py
    uv run python scripts/eval_11.py --ledger   # 11-3 시행 대장만
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.disclosure import events, order_day_features
from pullback.metrics import bootstrap_mean_ci, equity_curve
from pullback.paths import DATA
from pullback.spec import load, to_params
from pullback.theme import add_market_features

sys.stdout.reconfigure(encoding="utf-8")
CFG = tomllib.loads(open("configs/compare_11.toml", encoding="utf-8").read())
WIN = {"2022~2024": tuple(CFG["conditions"]["windows"]["common"]), "2025": tuple(CFG["conditions"]["windows"]["validation"])}
UNIT, CAPITAL = 1e6, 1e8


def money(t: pd.DataFrame) -> pd.Series:
    return t["invested"] * UNIT / t["entry_price"]


def stats(t, e, df, r, a, b):
    x = t[t["entry_date"].between(a, b)]
    if x.empty:
        return dict(n=0)
    m = money(x)
    eq = equity_curve(e, x, df, r, unit_money=UNIT, capital=CAPITAL)
    lo, hi = bootstrap_mean_ci(x["net_ret"], x["entry_date"].dt.to_period("M"))
    pnl = eq["equity"].iloc[-1] - CAPITAL
    span = eq.loc[a:b]
    return dict(n=len(x), mean=x["net_ret"].mean(), lo=lo, hi=hi, win=(x["net_ret"] > 0).mean(),
                mw=float((x["net_ret"] * m).sum() / m.sum()), pnl=pnl / CAPITAL, mdd=eq["dd"].min(),
                worst=x["net_ret"].min(), exp=span["exposure"].mean(), ret_exp=pnl / span["exposure"].mean())


def signal_value(trades, df, col):
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    sig = pd.Series(df.groupby("code")[col].shift(1).to_numpy(), index=key)
    return sig.reindex(pd.MultiIndex.from_frame(trades[["code", "entry_date"]])).to_numpy()


def random_benchmark(x: pd.DataFrame, keep: np.ndarray, rng, draws: int):
    """B0 거래 x 중 keep 와 같은 수(전체·연도별)를 무작위로 골랐을 때의 평균·손익 합 분포."""
    r = x["net_ret"].to_numpy()
    pnl = (x["net_ret"] * money(x)).to_numpy() / CAPITAL
    year = x["entry_date"].dt.year.to_numpy()
    k = int(keep.sum())
    out = {"all": ([], []), "year": ([], [])}
    per_year = {y: int(keep[year == y].sum()) for y in np.unique(year)}
    idx_year = {y: np.flatnonzero(year == y) for y in per_year}
    for _ in range(draws):
        s = rng.choice(len(x), k, replace=False)
        out["all"][0].append(r[s].mean())
        out["all"][1].append(pnl[s].sum())
        s = np.concatenate([rng.choice(idx_year[y], n, replace=False) for y, n in per_year.items() if n])
        out["year"][0].append(r[s].mean())
        out["year"][1].append(pnl[s].sum())
    actual = (r[keep].mean(), pnl[keep].sum())
    res = {}
    for kind, (means, sums) in out.items():
        means, sums = np.array(means), np.array(sums)
        res[kind] = dict(mean_pct=(means < actual[0]).mean(), pnl_pct=(sums < actual[1]).mean(),
                         mean_p95=np.percentile(means, 95), pnl_p50=np.median(sums), pnl_p95=np.percentile(sums, 95))
    return k, actual, res, per_year


def ledger() -> None:
    """지금까지 매매 성과를 계산한 모든 변형. 개발 구간 순위가 검증 구간에서 유지되는가."""
    d = pd.read_csv("configs/trials_ledger.csv", dtype={"section": str})
    d = d[d["section"].str[:2] <= "11"]           # 11-3 시점(11장까지)의 대장. 12장 이후 행은 각 장에서 다룸
    same = d[d["window"] == "dev 2021-2024"]
    print(f"== 11-3 시행 대장: B0 포함 {len(d)}개 (같은 개발 구간 {len(same)}개)")
    print(f"  개발 구간 거래당 평균 > 0 인 변형: {(same['dev_mean_pct'] > 0).sum()}개, 최고 {same['dev_mean_pct'].max():+.2f}%")
    rho = same["dev_mean_pct"].rank().corr(same["val_mean_pct"].rank())
    print(f"  개발-검증 순위 상관 (Spearman) {rho:.3f}")
    s = same.assign(rd=same["dev_mean_pct"].rank(ascending=False), rv=same["val_mean_pct"].rank(ascending=False))
    print(s.sort_values("rd").loc[:, ["id", "dev_mean_pct", "val_mean_pct", "rd", "rv"]].head(6).to_string(index=False))
    top = s.sort_values("rd").head(5)
    print(f"  개발 상위 5개의 검증 평균 {top['val_mean_pct'].mean():+.2f}%, 전체 검증 중앙값 {same['val_mean_pct'].median():+.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    if ap.parse_args().ledger:
        return ledger()
    spec, _ = load("configs/b0.toml")
    a1 = tomllib.loads(open("configs/a1.toml", encoding="utf-8").read())
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    setup = f["setup"] & (df["date"] <= "2025-12-31")
    mk = add_market_features(df)
    df["mkt_gap_ma60"] = mk["mkt_gap_ma60"]
    sc = pd.read_parquet(DATA / "results" / "a1_scores.parquet")
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    df["p"] = pd.Series(sc["p"].to_numpy(), index=pd.MultiIndex.from_frame(sc[["code", "date"]])).reindex(key).to_numpy()
    df["bad"] = order_day_features(df, setup, events())["bad_since_peak"].reindex(df.index)
    # 각 전략의 '주문을 내도 되는가' 플래그 (신호일 기준)
    df["ok_B0"] = 1.0
    df["ok_B1"] = df["mkt_gap_ma60"].ge(0).astype(float)
    df["ok_A1"] = (df["p"] >= a1["decision"]["threshold"]).astype(float)   # 2022 이전은 점수 없음 -> 0
    df["ok_A2"] = (df["bad"] != 1).astype(float)
    masks = {s: setup & df[f"ok_{s}"].eq(1) for s in ["B0", "B1", "A1", "A2"]}

    base = Rules.from_spec(spec)
    print("== 11-1 동일 조건 성과 (기본 비용)")
    runs = {s: run(df, m, base) for s, m in masks.items()}
    for w, (a, b) in WIN.items():
        print(f" [{w}]")
        for s, (t, e) in runs.items():
            x = stats(t, e, df, base, a, b)
            if x["n"] == 0:
                print(f"  {s} 거래 0")
                continue
            print(f"  {s} 거래 {x['n']:>4}  평균 {x['mean']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]  승률 {x['win']:.3f}"
                  f"  금액가중 {x['mw']:+.4f}  계좌 {x['pnl']:+.4f}  MDD {x['mdd']:+.4f}  최악 {x['worst']:+.4f}"
                  f"  평균투입 {x['exp'] / 1e6:,.1f}백만  투입대비 {x['ret_exp']:+.3f}")

    print("\n== 11-1 제외된 거래 (B0 거래를 신호일 플래그로 나눔)")
    b0_t, _ = runs["B0"]
    for s in ["B1", "A1", "A2"]:
        b0_t[f"ok_{s}"] = signal_value(b0_t, df, f"ok_{s}")
    for w, (a, b) in WIN.items():
        x = b0_t[b0_t["entry_date"].between(a, b)]
        for s in ["B1", "A1", "A2"]:
            k, d = x[x[f"ok_{s}"] == 1], x[x[f"ok_{s}"] != 1]
            print(f"  [{w}] {s} 남김 {len(k)}건 {k['net_ret'].mean():+.4f} 승률 {(k['net_ret'] > 0).mean():.3f}"
                  f" | 제외 {len(d)}건 {d['net_ret'].mean():+.4f} 승률 {(d['net_ret'] > 0).mean():.3f}")

    print("\n== 11-2 무작위 필터 기준 (남긴 B0 거래 vs 같은 수 무작위)")
    rng = np.random.default_rng(CFG["random_benchmark"]["seed"])
    draws = CFG["random_benchmark"]["draws"]
    for w, (a, b) in WIN.items():
        x = b0_t[b0_t["entry_date"].between(a, b)].reset_index(drop=True)
        for s in ["B1", "A1", "A2"]:
            keep = (x[f"ok_{s}"] == 1).to_numpy()
            k, act, res, per_year = random_benchmark(x, keep, rng, draws)
            for kind, v in res.items():
                verdict = "구별됨" if act[0] > v["mean_p95"] else "구별 안 됨"
                print(f"  [{w}] {s} {kind:<4} 남김 {k} {per_year if kind == 'year' else ''} 평균 {act[0]:+.4f}"
                      f" (무작위 대비 백분위 {v['mean_pct']:.3f}, 95백분위 {v['mean_p95']:+.4f}) {verdict}"
                      f" | 손익합 {act[1]:+.4f} (백분위 {v['pnl_pct']:.3f}, 중앙 {v['pnl_p50']:+.4f})")

    print("\n== 11-2 비용 민감도 (슬리피지 편도)")
    for slip in CFG["conditions"]["slippage_grid"]:
        r = replace(base, slippage=slip)
        row = []
        for s, m in masks.items():
            t, e = run(df, m, r)
            for w, (a, b) in WIN.items():
                x = stats(t, e, df, r, a, b)
                row.append(f"{s}[{w[:4]}] {x.get('mean', float('nan')):+.4f}/{x.get('pnl', float('nan')):+.3f}")
        print(f"  {slip:.3f}: " + "  ".join(row))


if __name__ == "__main__":
    main()
