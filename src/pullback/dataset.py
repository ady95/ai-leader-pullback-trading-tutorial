"""08장 A1 학습 데이터셋: 종목 x 신호일 표본, Triple Barrier 라벨, 26개 특징량.

표본: B0 후보일(t) 중 1차 매수선이 있고, t+1일 저가가 그 선에 닿아 체결된 날.
라벨: t+1일 체결가 기준, t+2일부터 vertical_days 거래일 동안
      고가 >= 체결가 x (1 + upper)  -> 1 (목표 먼저)
      종가 <  체결가 x (1 - lower)  -> -1 (손절 먼저)   같은 날이면 손절 먼저
      둘 다 없음                     -> 0 (기간 만료)
label_end 는 라벨이 정해진 날(장벽에 닿은 날 또는 기간 끝)이며 08-3 퍼지에 쓴다.

사용:
    uv run python -m pullback.dataset
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from pullback.backtest import first_buy_price
from pullback.definitions import build, flags, load_daily
from pullback.features import FEATURES, add_feature_columns
from pullback.paths import DATA
from pullback.theme import MARKET_FEATURES, THEME_FEATURES, add_market_features, add_theme_features

sys.stdout.reconfigure(encoding="utf-8")
ALL_FEATURES = FEATURES + MARKET_FEATURES + THEME_FEATURES
BARRIERS = {"y": (0.20, 0.0688, 60), "y_up25": (0.25, 0.0688, 60), "y_short": (0.05, 0.04, 5)}


def triple_barrier(o, h, lo, c, entry_i, fill, up, dn, days):
    """한 표본의 라벨(1, -1, 0)과 라벨이 정해진 위치."""
    end = min(len(c) - 1, entry_i + days)
    for j in range(entry_i + 1, end + 1):
        hit_dn = c[j] < fill * (1 - dn)
        hit_up = h[j] >= fill * (1 + up)
        if hit_dn:
            return -1, j
        if hit_up:
            return 1, j
    complete = entry_i + days <= len(c) - 1
    return (0 if complete else np.nan), end


def make_dataset() -> pd.DataFrame:
    df = add_feature_columns(build(load_daily("2020-01-01")))
    f = flags(df)
    line = first_buy_price(df)
    g = df.groupby("code", sort=False)
    low_n, open_n, halt_n = g["low"].shift(-1), g["open"].shift(-1), g["volume"].shift(-1).eq(0)
    sig = f["setup"] & line.notna() & (low_n <= line) & ~halt_n
    mk = add_market_features(df)
    th = add_theme_features(df, sig)

    rows = []
    for code, x in df.groupby("code", sort=False):
        idx = x.index[sig.loc[x.index].to_numpy()]
        if len(idx) == 0:
            continue
        pos = {k: n for n, k in enumerate(x.index)}
        o, h, lo, c = (x[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        dates = x["date"].to_numpy()
        for k in idx:
            t = pos[k]
            fill = min(o[t + 1], line[k])
            rec = {"row": k, "code": code, "date": dates[t], "entry_date": dates[t + 1], "fill": fill,
                   "gap_fill": bool(o[t + 1] <= line[k]), "peak_date": x.at[k, "peak_date"]}
            for name, (up, dn, days) in BARRIERS.items():
                y, j = triple_barrier(o, h, lo, c, t + 1, fill, up, dn, days)
                rec[name] = y
                rec[f"{name}_end"] = dates[j]
            rows.append(rec)
    ds = pd.DataFrame(rows).set_index("row")
    feat = pd.DataFrame(index=ds.index)
    s = df.loc[ds.index]
    feat["drawdown"], feat["wave_rise"], feat["rise_days"] = s["drawdown"], s["wave_rise"], s["rise_days"]
    feat["days_since_peak"], feat["ret_1d"] = s["days_since_peak"], s["ret"]
    feat["ret_5d"] = s["close"] / s["close_5"] - 1
    feat["atr_pct"] = s["atr"] / s["close"]
    for name, col in [("gap_kijun", "kijun120"), ("gap_ma60", "ma60"), ("gap_env", "env_low"), ("gap_ma120", "ma120")]:
        feat[name] = s[col] / s["close"] - 1
    feat["log_wave_amt"] = np.log10(s["wave_amt_sum"] + 1)
    feat["amt_contraction"] = s["amt5"] / s["wave_amt_max"]
    feat["vol_ratio_5_20"] = s["vol5"] / s["vol20"]
    feat["amt_pct_today"] = s["amt_pct"]
    feat["log_amt20"] = np.log10(s["amt20"] + 1)
    feat["wave_lu_cnt"], feat["lu_run_max"] = s["wave_lu_cnt"], s["lu_run_max_wave"]
    for c in MARKET_FEATURES:
        feat[c] = mk.loc[ds.index, c]
    for c in THEME_FEATURES:
        feat[c] = th.loc[ds.index, c].astype(float)
    out = pd.concat([ds, feat[ALL_FEATURES]], axis=1).reset_index(drop=True)
    out["wave_id"] = out["code"] + "_" + pd.to_datetime(out["peak_date"]).dt.strftime("%Y%m%d")
    return out


def main() -> None:
    t0 = time.perf_counter()
    ds = make_dataset()
    ds.to_parquet(DATA / "dataset_a1.parquet", index=False)
    print(f"표본 {len(ds):,}행 · {ds['code'].nunique():,}종목 · 파동 {ds['wave_id'].nunique():,}개, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
