"""02장 정의의 기술 통계 (수익률은 보지 않는다).

사전 등록 원칙에 따라 이 스크립트는 '조건을 만족하는 빈도'와 '지지선의 위치'만 센다.
진입 뒤의 수익률·승률은 정의를 확정한 뒤 06~07장에서 처음 계산한다.

사용:
    uv run python scripts/describe_02.py
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from pullback.definitions import Params, build, flags, load_daily
from pullback.paths import UNIVERSE

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)

EVAL_START = "2021-01-01"   # 2020년은 지표 예열 구간 (04-3)
EXAMPLES = {"001210": "금호전기", "079650": "서산", "002990": "금호건설"}   # 원문 사례


def section(title: str) -> None:
    print(f"\n== {title}")


def n_waves(rows: pd.DataFrame) -> int:
    return rows.groupby(["code", "peak_date"]).ngroups


def main() -> None:
    p = Params()
    t = time.perf_counter()
    df = build(load_daily(), p)
    fa = flags(df, p)
    print(f"build {time.perf_counter() - t:.0f}s, params {p.as_dict()}")
    m = df["date"] >= EVAL_START
    d, f = df[m].copy(), fa[m]
    d["year"] = d["date"].dt.year

    section("02-1 상한가 일수 (연도별)")
    lu = d[d["limit_up"]]
    print(lu.groupby("year").agg(days=("code", "size"), stocks=("code", "nunique")).T.to_string())
    print("상한가 판정 중 종가=고가 비율", round((lu["close"] == lu["high"]).mean(), 3))

    section("02-1 하루 거래대금 5,000억 이상 (키움, 연도별)")
    big = d[(d["amount"] >= p.amount_min) & ~d["amount_is_est"]]
    print(big.groupby("year").agg(days=("code", "size"), stocks=("code", "nunique")).T.to_string())
    top = big["code"].value_counts()
    print("상위 2종목 비중", round(top.head(2).sum() / len(big), 3), top.head(3).to_dict())

    section("02-1 그날 거래대금 상위 0.5% 경계값 (억 원, 연도별 중앙값)")
    thr = d[~d["halted"]].groupby("date")["amount"].quantile(p.amount_top_pct) / 1e8
    print(thr.groupby(thr.index.year).median().round(0).to_string())

    section("02-1 원문 사례 종목의 거래대금 (2025년 이후)")
    ex = df[df["code"].isin(EXAMPLES) & (df["date"] >= "2025-01-01")]
    for code, name in EXAMPLES.items():
        x = ex[ex["code"] == code]
        fx = fa.loc[x.index]
        run = x["limit_up"].astype(int).groupby((~x["limit_up"]).cumsum()).cumsum().max()
        print(f"  {name}({code}) 하루 최대 {x['amount'].max() / 1e8:,.0f}억 ({x.loc[x['amount'].idxmax(), 'date']:%Y-%m-%d}),"
              f" 5,000억 이상 {int((x['amount'] >= p.amount_min).sum())}일, 최대 연속 상한가 {run},"
              f" 파동 누적 최대 {x['wave_amt_sum'].max() / 1e8:,.0f}억,"
              f" B0 주도주일 {int(fx['leader'].sum())}, 후보일 {int(fx['setup'].sum())},"
              f" 하루기준이면 주도주일 {int((fx['wave'] & fx['kki'] & fx['amt_day']).sum())}")

    section("02-2 상승 파동 수 (code, 고점일 기준 고유 파동)")
    w = d[f["wave"]].join(f[["kki", "kki_hist", "amt_day", "amt_cum", "amt_pct"]].add_prefix("f_"))
    waves = w.sort_values("date").groupby(["code", "peak_date"]).last()
    waves["peak_year"] = pd.to_datetime(waves.index.get_level_values("peak_date")).year
    print("고유 파동", len(waves), "종목", waves.index.get_level_values(0).nunique())
    print("상승률 분위", waves["wave_rise"].quantile([.25, .5, .75, .9]).round(2).to_dict())
    print("상승 기간(거래일) 분위", waves["rise_days"].quantile([.25, .5, .75]).to_dict())
    by = lambda mask: waves[mask].groupby("peak_year").size()
    tab = pd.DataFrame({
        "전체": by(waves["wave_rise"] > -1),
        "끼(파동 안 상한가)": by(waves["f_kki"]),
        "끼+누적5000억(B0)": by(waves["f_kki"] & waves["f_amt_cum"]),
        "끼+하루5000억": by(waves["f_kki"] & waves["f_amt_day"]),
        "끼+상위0.5%": by(waves["f_kki"] & waves["f_amt_pct"]),
        "끼120일+누적5000억": by(waves["f_kki_hist"] & waves["f_amt_cum"]),
    }).fillna(0).astype(int)
    print(tab.T.to_string())

    section("02-2 상승률 하한 민감도 (끼 + 누적 5,000억)")
    base = (d["days_since_peak"] >= 1) & f["kki"] & f["amt_cum"]
    for r in (0.3, 0.5, 1.0):
        x = d[base & (d["wave_rise"] >= r)]
        print(f"  rise>={r}: 파동 {n_waves(x)} 종목 {x['code'].nunique()}")

    section("02-2 B0 주도주 파동과 조정 구간")
    L, S = d[f["leader"]], d[f["setup"]]
    print("B0 주도주 파동", n_waves(L), "조정 구간 진입", n_waves(S), round(n_waves(S) / max(n_waves(L), 1), 3))
    first = S.sort_values("date").groupby(["code", "peak_date"]).first()
    print("첫 조정일 고점 경과일 분위", first["days_since_peak"].quantile([.25, .5, .75]).to_dict())
    print("첫 조정일 낙폭 분위", first["drawdown"].quantile([.25, .5, .75]).round(3).to_dict())
    print("후보일 행 수", len(S), "종목", S["code"].nunique(), "추정 거래대금 행", int(S["amount_is_est"].sum()))
    yl = L.groupby(["code", "peak_date"]).size().reset_index()
    ys = S.groupby(["code", "peak_date"]).size().reset_index()
    print(pd.DataFrame({"leader": yl.groupby(yl["peak_date"].dt.year).size(),
                        "setup": ys.groupby(ys["peak_date"].dt.year).size()}).T.to_string())
    u = pd.read_parquet(UNIVERSE)[["code", "market", "status"]]
    print("조정 파동 시장·상태", ys.merge(u, on="code").groupby(["market", "status"]).size().to_dict())

    section("02-3 후보일 종가 대비 지지선 위치 (지지선/종가 - 1)")
    s = S.copy()
    lines = ["ma60", "kijun120", "ma120", "env_low"]
    gaps = pd.DataFrame({c: s[c] / s["close"] - 1 for c in lines})
    print(gaps.quantile([.25, .5, .75]).round(3).to_string())
    print("종가 아래 비율", {c: round(float((s[c] < s["close"]).mean()), 3) for c in lines})
    print("ATR/종가 분위", (s["atr"] / s["close"]).quantile([.25, .5, .75]).round(3).to_dict())
    print("기준선 > 60일선 비율", round(float((s["kijun120"] > s["ma60"]).mean()), 3),
          "120일 이동평균 > 60일선 비율", round(float((s["ma120"] > s["ma60"]).mean()), 3))
    for per in (10, 20):
        mid = df.groupby("code")["close"].transform(lambda x: x.rolling(per, min_periods=per).mean()).loc[s.index]
        for pct in (0.10, 0.15, 0.20):
            low = mid * (1 - pct)
            print(f"  env({per},{pct:.0%}) 위치 중앙값 {float((low / s['close'] - 1).median()):.3f}"
                  f" 종가 아래 {float((low < s['close']).mean()):.3f}")

    section("02-3 지지선 중첩 (종가 아래, 서로 3% / 1 ATR 이내)")
    below = {c: s[c] < s["close"] for c in lines}
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            both = below[a] & below[b]
            dist = (s[a] - s[b]).abs()
            print(f"  {a:>8} ~ {b:<8} 3%: {float((both & (dist / s['close'] <= p.overlap_pct)).mean()):.3f}"
                  f"  1ATR: {float((both & (dist <= s['atr'])).mean()):.3f}")
    nb = sum(below[c].astype(int) for c in lines)
    print("종가 아래 지지선 개수 분포", nb.value_counts(normalize=True).sort_index().round(3).to_dict())

    section("02-3 다음 날 장중 접촉 (지지선 가격 지정가)")
    g = df.groupby("code")
    low_n = g["low"].shift(-1).loc[s.index]
    open_n = g["open"].shift(-1).loc[s.index]
    halt_n = (g["volume"].shift(-1) == 0).loc[s.index]
    cands = {"kijun120": ["kijun120"], "ma120": ["ma120"], "env_low": ["env_low"],
             "B0 1차(기준선/엔벨로프 중 높은 쪽)": ["kijun120", "env_low"],
             "대안 1차(120일 이동평균/엔벨로프)": ["ma120", "env_low"]}
    for name, cols in cands.items():
        line = s[cols].where(s[cols].lt(s["close"], axis=0)).max(axis=1)
        has = line.notna()
        touch = has & (low_n <= line) & ~halt_n
        rec = (s["low"] <= line) & (s["close"] > line)
        print(f"  {name}: 접촉 {int(touch.sum())}/{int(has.sum())} = {touch.sum() / max(has.sum(), 1):.3f}"
              f", 파동 {n_waves(s[touch])}, 시가가 이미 아래 {float((touch & (open_n <= line)).sum() / max(touch.sum(), 1)):.3f}"
              f", 회복확인일 {int(rec.sum())}")


if __name__ == "__main__":
    main()
