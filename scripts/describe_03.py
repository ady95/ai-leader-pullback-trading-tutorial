"""03장 규칙 구조 통계 (진입 시점에 알 수 있는 값만, 수익률은 보지 않는다).

사용:
    uv run python scripts/describe_03.py
"""
from __future__ import annotations

import sys

import pandas as pd

from pullback.definitions import Params, build, flags, load_daily

sys.stdout.reconfigure(encoding="utf-8")
EVAL_START = "2021-01-01"


def main() -> None:
    p = Params()
    df = build(load_daily(), p)
    f = flags(df, p)
    g = df.groupby("code")
    df["rows_so_far"] = g.cumcount() + 1
    for c in ("kijun120", "env_low", "ma60"):
        df[f"{c}_same"] = g[c].diff().eq(0)
    m = (df["date"] >= EVAL_START) & f["setup"]
    s = df[m].copy()

    print("== 03-1 지표 예열")
    print("후보일", len(s), "기준선 없음(상장 120거래일 미만)", int(s["kijun120"].isna().sum()),
          "60일선 없음", int(s["ma60"].isna().sum()), "엔벨로프 없음", int(s["env_low"].isna().sum()))
    print("기준선 없는 후보일의 종목 수", s[s["kijun120"].isna()]["code"].nunique())

    print("\n== 03-1 전일과 같은 값 비율 (후보일)")
    print({c: round(float(s[f"{c}_same"].mean()), 3) for c in ("kijun120", "env_low", "ma60")})
    runs = s.assign(chg=~s["kijun120_same"]).groupby("code")["chg"].cumsum()
    run_len = s.groupby([s["code"], runs]).size()
    print("기준선 값이 유지된 연속 후보일 길이 분위", run_len.quantile([.5, .75, .9]).to_dict())

    print("\n== 03-2 1차 매수선 첫 접촉일 (파동별 첫 번째)")
    line = s[["kijun120", "env_low"]].where(s[["kijun120", "env_low"]].lt(s["close"], axis=0)).max(axis=1)
    low_n = g["low"].shift(-1).loc[s.index]
    halt_n = (g["volume"].shift(-1) == 0).loc[s.index]
    touch = line.notna() & (low_n <= line) & ~halt_n
    t = s[touch].assign(line=line[touch]).sort_values("date").groupby(["code", "peak_date"]).first()
    print("첫 접촉 파동", len(t))
    print("1차선 종류", {"기준선": int((t["line"] == t["kijun120"]).sum()), "엔벨로프": int((t["line"] == t["env_low"]).sum())})
    for pct in (0.03, 0.04, 0.05):
        p2 = t["line"] * (1 - pct)
        gap = t["ma60"] / p2 - 1
        print(f"  2차가 = 1차선 x (1-{pct:.0%}): 60일선이 2차가 ±3% 안 {float((gap.abs() <= 0.03).mean()):.3f},"
              f" 위 {float((gap > 0.03).mean()):.3f}, 아래 {float((gap < -0.03).mean()):.3f}")
    print("1차선 대비 엔벨로프 하단 위치 중앙값", round(float((t["env_low"] / t["line"] - 1).median()), 3),
          "60일선 위치 중앙값", round(float((t["ma60"] / t["line"] - 1).median()), 3))
    print("1차선의 고점 대비 위치 중앙값", round(float((t["line"] / t["peak_close"] - 1).median()), 3))
    print("첫 접촉 연도별", t.groupby(t.index.get_level_values("peak_date").year).size().to_dict())


if __name__ == "__main__":
    main()
