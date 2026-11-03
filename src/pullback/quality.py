"""04-3 일봉 품질 검사 리포트.

검사 항목
  Q1 중복 행           같은 종목·같은 날짜가 두 번 이상
  Q2 가격 관계 위반     저가 > 시가·종가, 고가 < 시가·종가, 0 이하 가격(거래일)
  Q3 거래 없는 날       거래량 0 (거래정지 등)
  Q4 결측 거래일        시장 거래일인데 그 종목 행이 없음 (상장 기간 안에서)
  Q5 설명 안 되는 급변  수정주가 이벤트 없이 전일 대비 ±30.5% 초과
                       (가격제한폭 ±30% 에 수정주가 반올림 여유 0.5%p)
  Q6 거래대금 결측      amount 가 비어 있음 (폐지 종목 보조 소스)

전체 결과와 함께 출처(kiwoom, fdr)별 결과를 나란히 출력한다.

사용:
    uv run python -m pullback.quality --start 2021-01-01
"""
from __future__ import annotations

import argparse
import sys

from pullback.storage import connect

sys.stdout.reconfigure(encoding="utf-8")

# {t} 자리에 검사 대상 테이블(전체 또는 출처별)이 들어간다.
# 거래일 달력(cal)은 항상 키움 일봉 전체에서 만든다.
CHECKS = {
    "Q1 중복 행": """
        SELECT count(*) FROM (SELECT code, date FROM {t} GROUP BY 1, 2 HAVING count(*) > 1)""",
    "Q2 가격 관계 위반": """
        SELECT count(*) FROM {t} WHERE volume > 0 AND (
            low > least(open, close) OR high < greatest(open, close) OR low <= 0)""",
    "Q3 거래량 0인 날": "SELECT count(*) FROM {t} WHERE volume = 0",
    "Q4 결측 거래일": """
        WITH span AS (SELECT code, min(date) s, max(date) e FROM {t} GROUP BY 1)
        SELECT count(*) FROM span JOIN cal ON cal.date BETWEEN span.s AND span.e
        LEFT JOIN {t} x ON x.code = span.code AND x.date = cal.date
        WHERE x.code IS NULL""",
    "Q5 설명 안 되는 급변": """
        SELECT count(*) FROM (
            SELECT close / lag(close) OVER (PARTITION BY code ORDER BY date) - 1 AS r, adj_flag, volume
            FROM {t}) WHERE abs(r) > 0.305 AND adj_flag IS NULL AND volume > 0""",
    "Q6 거래대금 결측": "SELECT count(*) FROM {t} WHERE amount IS NULL OR isnan(amount)",
}


def report(start: str) -> dict[str, dict[str, int]]:
    con = connect(read_only=True)
    con.execute(f"CREATE TEMP TABLE d AS SELECT * FROM daily WHERE date >= DATE '{start}'")
    con.execute("CREATE TEMP TABLE cal AS SELECT DISTINCT date FROM d WHERE source = 'kiwoom'")
    total = con.execute("SELECT count(*), count(DISTINCT code) FROM d").fetchone()
    print(f"검사 대상: {total[0]:,}행 · {total[1]:,}종목 (date >= {start})")

    targets = {"전체": "d"}
    for (s,) in con.execute("SELECT DISTINCT source FROM d ORDER BY 1").fetchall():
        con.execute(f"CREATE TEMP TABLE d_{s} AS SELECT * FROM d WHERE source = '{s}'")
        n = con.execute(f"SELECT count(DISTINCT code) FROM d_{s}").fetchone()[0]
        targets[f"{s}({n:,})"] = f"d_{s}"

    print(f"  {'항목':<20}" + "".join(f"{k:>16}" for k in targets))
    out: dict[str, dict[str, int]] = {}
    for name, sql in CHECKS.items():
        out[name] = {k: con.execute(sql.format(t=t)).fetchone()[0] for k, t in targets.items()}
        print(f"  {name:<20}" + "".join(f"{v:>16,}" for v in out[name].values()))
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    report(ap.parse_args().start)


if __name__ == "__main__":
    main()
