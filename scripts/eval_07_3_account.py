"""07-3 주요 대안의 계좌 단위 비교 (1억, 1차 매수 100만 원 고정, 복리 없음).

사용:
    uv run python scripts/eval_07_3_account.py
"""
from __future__ import annotations

import sys
from dataclasses import replace

import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import equity_curve
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
PERIODS = {"dev": ("2021-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31")}


def main() -> None:
    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    base = Rules.from_spec(spec)
    variants = {"B0": base, "2차 매수 없음": replace(base, second_units=0.0), "2차 1.5배": replace(base, second_units=1.5),
                "손절 없음": replace(base, use_stop=False), "익절 고정 +25%": replace(base, tp_mode="fixed"),
                "시간 청산 20일": replace(base, max_hold_days=20), "기준선=120일 이동평균": replace(base, first_line="ma120_env")}
    for label, r in variants.items():
        t, e = run(df, f["setup"], r)
        cells = []
        for name, (a, b) in PERIODS.items():
            x = t[(t["entry_date"] >= a) & (t["entry_date"] <= b)]
            eq = equity_curve(e, x, df, r, unit_money=1e6, capital=1e8)
            money = (x["invested"] * 1e6 / x["entry_price"])
            cells.append(f"{name} 손익 {eq['equity'].iloc[-1] / 1e8 - 1:+.4f} MDD {eq['dd'].min():+.4f}"
                         f" 최대노출 {eq['exposure'].max() / 1e8:.2f}억 금액가중 {float((x['net_ret'] * money).sum() / money.sum()):+.4f}")
        print(f"{label:<16} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
