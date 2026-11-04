"""05-3 미래 정보 누수 검증.

1) 절단 검증: t일까지의 데이터만으로 다시 계산한 t일 특징량이 전체 데이터로 계산한 값과 같은가
2) 수정주가 누수 규모: 후보일 이후에 수정주가 이벤트가 생긴 종목의 비율

사용:
    uv run python scripts/verify_05_3.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pullback.definitions import build, flags, load_daily
from pullback.features import FEATURES, add_feature_columns, feature_table

sys.stdout.reconfigure(encoding="utf-8")
DATES = ["2022-06-15", "2024-03-15", "2026-10-07"]


def table_at(raw: pd.DataFrame, date: str) -> pd.DataFrame:
    df = add_feature_columns(build(raw))
    f = flags(df)
    return feature_table(df, f, f["setup"] & (df["date"] == date)).set_index("code").sort_index()


def main() -> None:
    raw = load_daily()
    full_df = add_feature_columns(build(raw))
    full_f = flags(full_df)
    for d in DATES:
        full = feature_table(full_df, full_f, full_f["setup"] & (full_df["date"] == d)).set_index("code").sort_index()
        cut = table_at(raw[raw["date"] <= d], d)
        same_set = full.index.equals(cut.index)
        diff = np.nanmax(np.abs(full[FEATURES].to_numpy(float) - cut[FEATURES].to_numpy(float))) if same_set else np.nan
        print(f"{d}: 후보 종목 전체 {len(full)} / 절단 {len(cut)}, 같은 종목 {same_set}, 특징량 최대 차이 {diff:.2e}")

    print("\n== 수정주가 누수 규모 (평가 구간 후보일)")
    s = full_df[full_f["setup"] & (full_df["date"] >= "2021-01-01")][["code", "date"]]
    ev = raw[raw["adj_flag"].notna()][["code", "date"]].rename(columns={"date": "ev_date"})
    last_ev = ev.groupby("code")["ev_date"].max()
    later = s["date"] < s["code"].map(last_ev)
    print("후보일 중 이후에 수정주가 이벤트가 있는 종목의 행 비율", round(float(later.mean()), 3))


if __name__ == "__main__":
    main()
