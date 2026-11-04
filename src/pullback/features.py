"""05-1 눌림목 후보일의 특징량.

모든 특징량은 t일 장 마감까지의 데이터로만 계산한다(05-3 절단 검증으로 확인).
가격의 절대 수준(원 단위 가격)은 수정주가 기준이 미래 이벤트에 따라 바뀌므로
특징량에 넣지 않고, 비율·순위·로그 거래대금만 쓴다.

사용:
    from pullback.definitions import build, flags, load_daily
    from pullback.features import add_feature_columns, feature_table
    df = add_feature_columns(build(load_daily()))
    X = feature_table(df, flags(df))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pullback.definitions import _range_max

PRICE = ["drawdown", "wave_rise", "rise_days", "days_since_peak", "ret_1d", "ret_5d", "atr_pct",
         "gap_kijun", "gap_ma60", "gap_env", "gap_ma120"]
FLOW = ["log_wave_amt", "amt_contraction", "vol_ratio_5_20", "amt_pct_today", "log_amt20",
        "wave_lu_cnt", "lu_run_max"]
FEATURES = PRICE + FLOW


def add_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """build() 결과에 특징량 계산용 열을 더한다 (종목별 과거 창만 사용)."""
    df = df.copy()
    g = df.groupby("code", sort=False)
    roll_mean = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=n).mean())
    df["amt5"] = roll_mean("amount", 5)
    df["amt20"] = roll_mean("amount", 20)
    df["vol5"] = roll_mean("volume", 5)
    df["vol20"] = roll_mean("volume", 20)
    df["close_5"] = g["close"].shift(5)
    # 연속 상한가 길이: 상한가가 끊기면 0으로 돌아간다
    lu = df["limit_up"].astype(int)
    df["lu_run"] = lu.groupby([df["code"], (lu == 0).cumsum()]).cumsum()
    parts = []
    for _, x in df.groupby("code", sort=False):
        idx = np.arange(len(x))
        tp = idx - x["days_since_peak"].to_numpy()
        tl = tp - x["rise_days"].to_numpy()
        parts.append(pd.Series(_range_max(x["lu_run"].to_numpy(float), tl, tp), index=x.index))
    df["lu_run_max_wave"] = pd.concat(parts)
    return df


def feature_table(df: pd.DataFrame, f: pd.DataFrame, mask: pd.Series | None = None) -> pd.DataFrame:
    """B0 후보일(기본) 행의 특징량 표."""
    m = f["setup"] if mask is None else mask
    s = df[m]
    eps = 1.0
    out = pd.DataFrame({
        "code": s["code"], "date": s["date"], "peak_date": s["peak_date"],
        # 가격 위치·조정 형태
        "drawdown": s["drawdown"],
        "wave_rise": s["wave_rise"],
        "rise_days": s["rise_days"],
        "days_since_peak": s["days_since_peak"],
        "ret_1d": s["ret"],
        "ret_5d": s["close"] / s["close_5"] - 1,
        "atr_pct": s["atr"] / s["close"],
        "gap_kijun": s["kijun120"] / s["close"] - 1,
        "gap_ma60": s["ma60"] / s["close"] - 1,
        "gap_env": s["env_low"] / s["close"] - 1,
        "gap_ma120": s["ma120"] / s["close"] - 1,
        # 거래대금·거래량·끼
        "log_wave_amt": np.log10(s["wave_amt_sum"] + eps),
        "amt_contraction": s["amt5"] / s["wave_amt_max"],
        "vol_ratio_5_20": s["vol5"] / s["vol20"],
        "amt_pct_today": s["amt_pct"],
        "log_amt20": np.log10(s["amt20"] + eps),
        "wave_lu_cnt": s["wave_lu_cnt"],
        "lu_run_max": s["lu_run_max_wave"],
        "amount_is_est": s["amount_is_est"],
    })
    return out.reset_index(drop=True)
