"""05-2 시장 상대강도·동반 급등 테마 특징량의 분포와 검증 (수익률은 보지 않는다).

사용:
    uv run python scripts/describe_05_2.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from pullback.definitions import build, flags, load_daily
from pullback.paths import DATA, THEME_DIR, UNIVERSE
from pullback.theme import MARKET_FEATURES, THEME_FEATURES, add_market_features, add_theme_features

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
EVAL_START = "2021-01-01"


def theme_features_for(raw: pd.DataFrame, day: str) -> pd.DataFrame:
    df = build(raw)
    f = flags(df)
    m = f["setup"] & (df["date"] == day)
    return add_theme_features(df, m).set_index(df.loc[m, "code"]).sort_index()


def main() -> None:
    t0 = time.perf_counter()
    raw = load_daily()
    df = build(raw)
    f = flags(df)
    m = f["setup"] & (df["date"] >= EVAL_START)
    mk = add_market_features(df)
    th = add_theme_features(df, m)
    print(f"계산 {time.perf_counter() - t0:.0f}s, 후보일 {int(m.sum())}")
    X = pd.concat([df.loc[m, ["code", "date", "peak_date", "wave_amt_sum"]], mk.loc[m, MARKET_FEATURES], th], axis=1)
    X.drop(columns="peers").to_parquet(DATA / "features_b0_theme.parquet")

    print("\n== 분포 (5%, 25%, 50%, 75%, 95%)")
    print(X[MARKET_FEATURES + THEME_FEATURES].astype(float).quantile([.05, .25, .5, .75, .95]).T.round(3).to_string())
    n = X["theme_n_peers"]
    print("동반 종목 0개 비율", round(float((n == 0).mean()), 3), " 1개 이상", round(float((n >= 1).mean()), 3),
          " 3개 이상", round(float((n >= 3).mean()), 3))
    has = n >= 1
    print("동반 종목이 있을 때 대장주(거래대금 1위) 비율", round(float((X.loc[has, "theme_rank_amt"] == 1).mean()), 3))
    print("시장 지수가 60일선 아래인 후보일 비율", round(float((X["mkt_gap_ma60"] < 0).mean()), 3))

    print("\n== 사례: 원문 종목의 동반 종목 (첫 후보일)")
    names = pd.read_parquet(UNIVERSE).set_index("code")["name"]
    for code in ("001210", "079650", "002990"):
        x = X[X["code"] == code].sort_values("date")
        if x.empty:
            continue
        r = x.iloc[0]
        print(f"  {names[code]} {r['date']:%Y-%m-%d} 동반 {int(r['theme_n_peers'])}개, 거래대금 순위 {int(r['theme_rank_amt'])}:",
              ", ".join(names.get(c, c) for c in r["peers"][:12]))

    print("\n== 키움 테마 스냅샷과 비교 (최근 후보일, 둘 다 스냅샷에 있는 쌍만)")
    snap = pd.concat(pd.read_parquet(p) for p in sorted(THEME_DIR.glob("*.parquet")))
    themes = snap.groupby("code")["thema_grp_cd"].apply(set).to_dict()
    recent = X[X["date"] >= "2026-08-01"]
    hit = tot = 0
    for c, peers in zip(recent["code"], recent["peers"]):
        if c not in themes:
            continue
        for q in peers:
            if q in themes:
                tot += 1
                hit += bool(themes[c] & themes[q])
    in_snap = [c for c in themes]
    rng = np.random.default_rng(0)
    pairs = rng.choice(in_snap, size=(20000, 2))
    base = np.mean([bool(themes[a] & themes[b]) for a, b in pairs if a != b])
    print(f"  동반 종목 쌍 {tot}개 중 같은 키움 테마 {hit}개 ({hit / max(tot, 1):.3f}), 무작위 쌍 기준선 {base:.3f}")
    print(f"  스냅샷 테마 {snap['thema_grp_cd'].nunique()}개, 종목 {snap['code'].nunique()}개")

    print("\n== 절단 검증 (2024-03-15)")
    day = "2024-03-15"
    full = th.loc[df.loc[m & (df['date'] == day)].index].set_index(df.loc[m & (df['date'] == day), "code"]).sort_index()
    cut = theme_features_for(raw[raw["date"] <= day], day)
    diff = np.nanmax(np.abs(full[THEME_FEATURES].astype(float).to_numpy() - cut[THEME_FEATURES].astype(float).to_numpy()))
    print(f"  같은 종목 {full.index.equals(cut.index)}, 테마 특징량 최대 차이 {diff:.2e}")


if __name__ == "__main__":
    main()
