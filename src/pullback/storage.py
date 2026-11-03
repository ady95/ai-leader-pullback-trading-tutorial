"""04-3 DuckDB 저장소.

종목별 Parquet 파일은 그대로 두고, DuckDB에는 그 파일을 읽는 뷰만 만든다.
원본 파일을 고치지 않으므로 언제든 뷰를 다시 만들 수 있다.

사용:
    uv run python -m pullback.storage
"""
from __future__ import annotations

import sys

import duckdb

from pullback.paths import DAILY_DIR, DB, UNIVERSE

sys.stdout.reconfigure(encoding="utf-8")


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB), read_only=read_only)


def build() -> None:
    con = connect()
    pattern = (DAILY_DIR / "*.parquet").as_posix()
    # amount 가 없는 행(폐지 종목 보조 소스)은 평균가격 x 거래량으로 거래대금을 추정한다.
    # 키움 데이터로 재 보면 이 추정치는 실제 거래대금의 ±10% 안에 99.8% 들어온다 (04-2).
    con.execute(f"""
        CREATE OR REPLACE VIEW daily AS
        SELECT *,
               coalesce(nullif(amount, 'NaN'::DOUBLE), (open + high + low + close) / 4 * volume) AS amount_filled,
               (amount IS NULL OR isnan(amount)) AS amount_is_est
        FROM (
            SELECT regexp_extract(filename, '([0-9A-Z]{{6}})\\.parquet$', 1) AS code, * EXCLUDE (filename)
            FROM read_parquet('{pattern}', filename = true, union_by_name = true)
        )
    """)
    con.execute(f"CREATE OR REPLACE VIEW universe AS SELECT * FROM read_parquet('{UNIVERSE.as_posix()}')")
    n = con.execute("SELECT count(*), count(DISTINCT code), min(date), max(date) FROM daily").fetchone()
    print(f"daily 뷰: {n[0]:,}행 · {n[1]:,}종목 · {n[2]:%Y-%m-%d} ~ {n[3]:%Y-%m-%d}")
    con.close()


if __name__ == "__main__":
    build()
