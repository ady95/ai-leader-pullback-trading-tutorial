"""02장 정의의 기술 통계 (수익률은 보지 않는다).

사전 등록 원칙에 따라 이 스크립트는 '조건을 만족하는 빈도'만 센다.
진입 뒤의 수익률·승률은 정의를 확정한 뒤 06~07장에서 처음 계산한다.

사용:
    uv run python scripts/describe_02.py
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from pullback.definitions import Params, build, flags, load_daily

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

EVAL_START = "2021-01-01"   # 2020년은 지표 예열 구간 (04-3)


def section(title: str) -> None:
    print(f"\n== {title}")


def main() -> None:
    p = Params()
    t = time.perf_counter()
    df = build(load_daily(), p)
    f = flags(df, p)
    print(f"build {time.perf_counter() - t:.0f}s, params {p.as_dict()}")
    m = df["date"] >= EVAL_START
    d, f = df[m].copy(), f[m]
    d["year"] = d["date"].dt.year

    section("02-1 상한가 일수 (연도별)")
    lu = d[d["limit_up"]]
    print(lu.groupby("year").agg(days=("code", "size"), stocks=("code", "nunique")).T.to_string())
    print("상한가 판정 중 종가=고가 비율", round((lu["close"] == lu["high"]).mean(), 3))

    section("02-1 단일일 거래대금 5,000억 이상 (연도별)")
    big = d[(d["amount"] >= p.amount_min) & ~d["amount_is_est"]]
    print(big.groupby("year").agg(days=("code", "size"), stocks=("code", "nunique")).T.to_string())
    top = big["code"].value_counts()
    print("상위 2종목 비중", round(top.head(2).sum() / len(big), 3), top.head(5).to_dict())

    section("02-1 그날 거래대금 상위 0.5% 경계값 (억 원, 연도별 중앙값)")
    thr = d[~d["halted"]].groupby("date")["amount"].quantile(p.amount_top_pct) / 1e8
    print(thr.groupby(thr.index.year).median().round(0).to_string())

    section("02-2 상승 파동 수 (code, 고점일 기준 고유 파동)")
    w = d[f["wave"]].copy()
    w["kki"], w["amt_day"], w["amt_cum"], w["amt_pct"] = f["kki"], f["amt_day"], f["amt_cum"], f["amt_pct"]
    waves = w.sort_values("date").groupby(["code", "peak_date"]).last()
    waves["peak_year"] = pd.to_datetime(waves.index.get_level_values("peak_date")).year
    print("고유 파동", len(waves), "종목", waves.index.get_level_values(0).nunique())
    print("상승률 분위", waves["wave_rise"].quantile([.25, .5, .75, .9]).round(2).to_dict())
    print("상승 기간(거래일) 분위", waves["rise_days"].quantile([.25, .5, .75]).to_dict())
    tab = pd.DataFrame({
        "전체": waves.groupby("peak_year").size(),
        "끼": waves[waves["kki"]].groupby("peak_year").size(),
        "끼+단일일5000억": waves[waves["kki"] & waves["amt_day"]].groupby("peak_year").size(),
        "끼+누적5000억": waves[waves["kki"] & waves["amt_cum"]].groupby("peak_year").size(),
        "끼+상위0.5%": waves[waves["kki"] & waves["amt_pct"]].groupby("peak_year").size(),
    }).fillna(0).astype(int)
    print(tab.T.to_string())

    section("02-2 B0 주도주 파동 중 조정 구간에 들어간 비율")
    lw = d[f["leader"]].groupby(["code", "peak_date"])
    reached = d[f["setup"]].groupby(["code", "peak_date"]).size()
    print("B0 주도주 파동", lw.ngroups, "조정 구간 진입", len(reached), round(len(reached) / max(lw.ngroups, 1), 3))
    first = d[f["setup"]].sort_values("date").groupby(["code", "peak_date"]).first()
    print("첫 조정일 고점 경과일 분위", first["days_since_peak"].quantile([.25, .5, .75]).to_dict())
    print("첫 조정일 낙폭 분위", first["drawdown"].quantile([.25, .5, .75]).round(3).to_dict())
    print("후보일(setup) 행 수", int(f["setup"].sum()), "종목", d[f["setup"]]["code"].nunique())

    section("02-3 후보일 종가 대비 지지선 위치 (지지선/종가 - 1, 중앙값)")
    s = d[f["setup"]].copy()
    lines = ["ma60", "ma120", "kijun120", "env_low"]
    for c in lines:
        s[f"gap_{c}"] = s[c] / s["close"] - 1
    print(s[[f"gap_{c}" for c in lines]].quantile([.25, .5, .75]).round(3).to_string())
    print("지지선이 종가 아래 있는 비율", {c: round((s[c] < s["close"]).mean(), 3) for c in lines})

    section("02-3 지지선 중첩 (종가 아래, 서로 3% 이내인 지지선 쌍)")
    below = {c: s[c] < s["close"] for c in lines}
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            ok = below[a] & below[b] & ((s[a] - s[b]).abs() / s["close"] <= p.overlap_pct)
            ok_atr = below[a] & below[b] & ((s[a] - s[b]).abs() <= s["atr"])
            print(f"  {a:>8} ~ {b:<8} 3%: {ok.mean():.3f}  1ATR: {ok_atr.mean():.3f}")
    n_overlap = sum(below[c].astype(int) for c in lines)
    s["n_below"] = n_overlap
    print("종가 아래 지지선 개수 분포", s["n_below"].value_counts(normalize=True).sort_index().round(3).to_dict())

    section("02-3 다음 날 장중 접촉 (지정가를 지지선에 걸었다면 닿았는가)")
    nxt = df.groupby("code")["low"].shift(-1).loc[s.index]
    for c in lines:
        has = s[c] < s["close"]
        touch = (nxt <= s[c]) & has
        print(f"  {c:<8} 지지선 아래 있음 {has.mean():.3f}  다음날 저가 접촉 {touch.sum() / max(has.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
