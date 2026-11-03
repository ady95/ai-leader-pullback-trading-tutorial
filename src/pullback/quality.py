"""04-3 일봉 품질 검사 리포트.

검사 항목
  Q1 중복 행           같은 종목·같은 날짜가 두 번 이상
  Q2 가격 관계 위반     저가 > 시가·종가, 고가 < 시가·종가, 0 이하 가격(거래일)
  Q3 거래 없는 날       거래량 0 (거래정지 등)
  Q4 결측 거래일        시장 거래일인데 그 종목 행이 없음 (상장 기간 안에서)
  Q5 설명 안 되는 급변  수정주가 이벤트 없이 전일 대비 ±30% 초과
  Q6 거래대금 결측      amount 가 비어 있음 (폐지 종목 보조 소스)

사용:
    uv run python -m pullback.quality --start 2021-01-01
"""
from __future__ import annotations

import argparse
import sys

from pullback.storage import connect

sys.stdout.reconfigure(encoding="utf-8")

CHECKS = {
    "Q1 중복 행": """
        SELECT count(*) FROM (SELECT code, date FROM d GROUP BY 1, 2 HAVING count(*) > 1)""",
    "Q2 가격 관계 위반": """
        SELECT count(*) FROM d WHERE volume > 0 AND (
            low > least(open, close) OR high < greatest(open, close) OR low <= 0)""",
    "Q3 거래량 0인 날": "SELECT count(*) FROM d WHERE volume = 0",
    "Q4 결측 거래일": """
        WITH cal AS (SELECT DISTINCT date FROM d WHERE source = 'kiwoom'),
             span AS (SELECT code, min(date) s, max(date) e FROM d GROUP BY 1)
        SELECT count(*) FROM span JOIN cal ON cal.date BETWEEN span.s AND span.e
        LEFT JOIN d ON d.code = span.code AND d.date = cal.date
        WHERE d.code IS NULL""",
    "Q5 설명 안 되는 ±30% 초과 변동": """
        SELECT count(*) FROM (
            SELECT close / lag(close) OVER (PARTITION BY code ORDER BY date) - 1 AS r, adj_flag, volume
            FROM d) WHERE abs(r) > 0.30 AND adj_flag IS NULL AND volume > 0""",
    "Q6 거래대금 결측": "SELECT count(*) FROM d WHERE amount IS NULL OR isnan(amount)",
}


def report(start: str) -> dict[str, int]:
    con = connect(read_only=True)
    con.execute(f"CREATE TEMP VIEW d AS SELECT * FROM daily WHERE date >= DATE '{start}'")
    total = con.execute("SELECT count(*), count(DISTINCT code) FROM d").fetchone()
    print(f"검사 대상: {total[0]:,}행 · {total[1]:,}종목 (date >= {start})")
    out = {}
    for name, sql in CHECKS.items():
        out[name] = con.execute(sql).fetchone()[0]
        print(f"  {name:<28} {out[name]:>10,}")
    by_src = con.execute("""
        SELECT source, count(DISTINCT code), sum(volume = 0)::INT, sum(amount IS NULL OR isnan(amount))::INT
        FROM d GROUP BY 1 ORDER BY 1""").fetchall()
    print("  출처별 (종목 수 / 거래량 0 / 거래대금 결측):")
    for s, n, z, m in by_src:
        print(f"    {s:<7} {n:>6,} {z:>8,} {m:>10,}")
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    report(ap.parse_args().start)


if __name__ == "__main__":
    main()
