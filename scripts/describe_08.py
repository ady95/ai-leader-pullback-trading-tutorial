"""08장 A1 데이터셋 통계. 최종 홀드아웃(2026) 표본의 라벨은 출력하지 않는다.

사용:
    uv run python -m pullback.dataset
    uv run python scripts/describe_08.py
"""
from __future__ import annotations

import sys
import tomllib

import pandas as pd

from pullback.paths import DATA

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)


def main() -> None:
    ds = pd.read_parquet(DATA / "dataset_a1.parquet")
    ds["year"] = pd.to_datetime(ds["date"]).dt.year
    spec = tomllib.loads(open("configs/a1.toml", encoding="utf-8").read())
    print("== 08-1 표본 수 (신호일 연도별)")
    per = ds.groupby("year").agg(samples=("code", "size"), stocks=("code", "nunique"), waves=("wave_id", "nunique"))
    print(per.T.to_string())
    per_wave = ds.groupby("wave_id").size()
    print("파동당 표본 수 분포", per_wave.value_counts().sort_index().head(8).to_dict(), "중앙", per_wave.median(), "최대", per_wave.max())
    print("파동 첫 표본만(B0 거래와 같은 시점) 비율", round(ds.groupby("wave_id").ngroups / len(ds), 3))

    open_ = ds[ds["year"] <= 2025]
    print("\n== 08-2 라벨 분포 (2021~2025만; 2026 홀드아웃 비공개)")
    for y in ("y", "y_up25", "y_short"):
        tab = open_.groupby("year")[y].value_counts(normalize=True).unstack().round(3)
        print(f"[{y}]\n{tab.to_string()}")
    days = (pd.to_datetime(open_["y_end"]) - pd.to_datetime(open_["entry_date"])).dt.days
    print("라벨 확정까지 달력일 분위", days.quantile([.25, .5, .75, .9]).to_dict())
    print("라벨 미확정(데이터 끝)", int(ds["y"].isna().sum()), "행 (모두 최근 표본)")

    print("\n== 08-3 퍼지: 폴드별 학습 표본과 제거 수")
    dates = pd.to_datetime(ds["date"])
    end = pd.to_datetime(ds["y_end"])
    for last_train, pred in spec["split"]["folds"]:
        start = pd.Timestamp(f"{pred}-01-01")
        train = ds[(dates >= spec["split"]["train_start"]) & (dates <= pd.Timestamp(f"{last_train}-12-31")) & ds["y"].notna()]
        test = ds[dates.dt.year == int(pred)]
        by_end = end.loc[train.index] >= start
        by_wave = train["wave_id"].isin(set(test["wave_id"]))
        kept = train[~(by_end | by_wave)]
        print(f"  학습 2021~{last_train} -> 예측 {pred}: 학습 후보 {len(train):,}, 라벨 겹침 {int(by_end.sum())},"
              f" 파동 공유 {int(by_wave.sum())}, 제거 합계 {int((by_end | by_wave).sum())}, 최종 학습 {len(kept):,}, 예측 {len(test):,}")


if __name__ == "__main__":
    main()
