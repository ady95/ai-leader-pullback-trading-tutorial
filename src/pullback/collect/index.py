"""05-2 시장 지수 일봉 (키움 ka20006 업종일봉조회).

업종코드 001 = 코스피 종합, 101 = 코스닥 종합. 응답 가격은 지수 x 100 이다.

사용:
    uv run python -m pullback.collect.index --start 2020-01-01 --end 2026-11-03
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from pullback.kiwoom import KiwoomClient
from pullback.paths import INDEX

sys.stdout.reconfigure(encoding="utf-8")
MARKETS = {"001": "KOSPI", "101": "KOSDAQ"}


def fetch(kw: KiwoomClient, inds_cd: str, start: str, end: str) -> pd.DataFrame:
    rows: list[dict] = []
    body = {"inds_cd": inds_cd, "base_dt": end.replace("-", "")}
    for chunk in kw.paginate("ka20006", "/api/dostk/chart", body, "inds_dt_pole_qry"):
        rows.extend(chunk)
        if chunk and chunk[-1]["dt"] < start.replace("-", ""):
            break
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "market": MARKETS[inds_cd],
        "date": pd.to_datetime(df["dt"]),
        "open": pd.to_numeric(df["open_pric"]) / 100,
        "high": pd.to_numeric(df["high_pric"]) / 100,
        "low": pd.to_numeric(df["low_pric"]) / 100,
        "close": pd.to_numeric(df["cur_prc"]) / 100,
        "amount": pd.to_numeric(df["trde_prica"]) * 1e6,   # 백만 원 단위
    })
    return out[(out["date"] >= start) & (out["date"] <= end)].drop_duplicates("date").sort_values("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-11-03")
    a = ap.parse_args()
    kw = KiwoomClient()
    df = pd.concat([fetch(kw, c, a.start, a.end) for c in MARKETS], ignore_index=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INDEX, index=False)
    print(df.groupby("market").agg(rows=("date", "size"), start=("date", "min"), end=("date", "max"),
                                   last=("close", "last")).to_string())
    print(f"호출 {kw.calls}회")


if __name__ == "__main__":
    main()
