"""03-1 지표 계산 검증: 빠른 구현(build)과 단순 구현(support_lines)이 같은 값을 내는지 확인한다.

사용:
    uv run python scripts/verify_03_1.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pullback.definitions import build, load_daily

sys.stdout.reconfigure(encoding="utf-8")
CODES = ["005930", "002990", "079650", "001210", "0147B0"]


def support_lines(df: pd.DataFrame, env_period: int = 20, env_pct: float = 0.20) -> pd.DataFrame:
    out = df.copy()
    out["ma60"] = out["close"].rolling(60).mean()
    out["ma120"] = out["close"].rolling(120).mean()
    out["kijun120"] = (out["high"].rolling(120).max() + out["low"].rolling(120).min()) / 2
    out["env_low"] = out["close"].rolling(env_period).mean() * (1 - env_pct)
    return out


def main() -> None:
    raw = load_daily()
    fast = build(raw[raw["code"].isin(CODES)])
    worst = 0.0
    for code in CODES:
        a = fast[fast["code"] == code].reset_index(drop=True)
        b = support_lines(raw[raw["code"] == code].sort_values("date").reset_index(drop=True))
        for c in ("ma60", "ma120", "kijun120", "env_low"):
            diff = np.nanmax(np.abs(a[c].to_numpy() - b[c].to_numpy())) if a[c].notna().any() else 0.0
            same_nan = (a[c].isna() == b[c].isna()).all()
            worst = max(worst, diff)
            if diff > 1e-6 or not same_nan:
                print("불일치", code, c, diff, same_nan)
    print(f"{len(CODES)}종목 4개 지표 최대 차이 {worst:.2e}")

    x = raw[(raw["code"] == "002990") & (raw["date"] <= "2026-10-07")].sort_values("date").tail(120)
    hi, lo = x["high"].max(), x["low"].min()
    print(f"금호건설 2026-10-07 120일 최고 {hi:,.0f} ({x.loc[x['high'].idxmax(), 'date']:%Y-%m-%d})"
          f" 최저 {lo:,.0f} ({x.loc[x['low'].idxmin(), 'date']:%Y-%m-%d}) 기준선 {(hi + lo) / 2:,.1f}")


if __name__ == "__main__":
    main()
