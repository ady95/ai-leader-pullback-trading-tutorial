"""05-1 B0 후보일 특징량의 분포와 중복도 (수익률은 보지 않는다).

사용:
    uv run python scripts/describe_05.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pullback.definitions import build, flags, load_daily
from pullback.features import FEATURES, add_feature_columns, feature_table
from pullback.paths import DATA

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
EVAL_START = "2021-01-01"


def main() -> None:
    df = add_feature_columns(build(load_daily()))
    f = flags(df)
    X = feature_table(df, f, f["setup"] & (df["date"] >= EVAL_START))
    X.to_parquet(DATA / "features_b0.parquet")
    print("후보일", len(X), "종목", X["code"].nunique())

    print("\n== 결측 비율 (0보다 큰 것만)")
    na = X[FEATURES].isna().mean()
    print(na[na > 0].round(4).to_dict())

    print("\n== 분포 (5%, 25%, 50%, 75%, 95%)")
    print(X[FEATURES].quantile([.05, .25, .5, .75, .95]).T.round(3).to_string())

    print("\n== 해석용 통계")
    print("거래대금 수축: 최근 5일 평균 / 파동 최대일 중앙값", round(float(X["amt_contraction"].median()), 3),
          ", 0.1 미만 비율", round(float((X["amt_contraction"] < 0.1).mean()), 3))
    print("거래량 5일/20일 < 1 비율", round(float((X["vol_ratio_5_20"] < 1).mean()), 3))
    print("파동 안 최대 연속 상한가 분포", X["lu_run_max"].value_counts(normalize=True).sort_index().round(3).to_dict())
    print("3연속 이상 비율", round(float((X["lu_run_max"] >= 3).mean()), 3))

    print("\n== 강하게 겹치는 특징량 쌍 (스피어만 |rho| >= 0.7)")
    c = X[FEATURES].corr(method="spearman")
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:] if abs(c.loc[a, b]) >= 0.7]
    for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
        print(f"  {a:<16} {b:<16} {r:+.2f}")


if __name__ == "__main__":
    main()
