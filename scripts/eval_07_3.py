"""07-3 가설 H2·H3·H4와 명세서 [variants] 대안 비교 (개발·검증 구간).

대안은 한 번에 하나만 바꾼다. 모든 대안은 03-3 의 명세서에 미리 적어 둔 것이다.

사용:
    uv run python scripts/eval_07_3.py
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from pullback.backtest import Rules, first_buy_price, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import bootstrap_mean_ci, path_stats
from pullback.spec import load, to_params

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220)
PERIODS = {"dev": ("2021-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31")}


def stats(t: pd.DataFrame) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        x = t[(t["entry_date"] >= a) & (t["entry_date"] <= b)]
        lo, hi = bootstrap_mean_ci(x["net_ret"], x["entry_date"].dt.to_period("M"))
        out[name] = (len(x), x["net_ret"].mean(), lo, hi, (x["net_ret"] > 0).mean(), x["net_ret"].min())
    return out


def show(label: str, st: dict) -> None:
    cells = [f"{n:>4} {m:+.4f} [{lo:+.4f},{hi:+.4f}] 승률 {w:.2f} 최악 {wr:+.3f}" for n, m, lo, hi, w, wr in st.values()]
    print(f"  {label:<34} | " + " | ".join(cells))


def main() -> None:
    spec, digest = load("configs/b0.toml")
    p = to_params(spec)
    raw = load_daily(spec["periods"]["warmup_start"])
    df = build(raw, p)
    f = flags(df, p)
    base = Rules.from_spec(spec)
    print(f"spec {digest[:16]}   열: 거래 수 평균 [95% CI] 승률 최악   | dev | validation")

    print("\n== 기준")
    b0, ev0 = run(df, f["setup"], base)
    show("B0", stats(b0))

    print("\n== H2: 지지선 중첩 vs 60일선 단독")
    t60, _ = run(df, f["setup"], replace(base, first_line="ma60"))
    show("60일선 단독 진입", stats(t60))
    key = df.set_index(["code", "date"])
    sig = df.groupby("code")[["kijun120", "env_low", "ma60", "close"]].shift(1)
    sig["line"] = first_buy_price(df.assign(**{c: sig[c] for c in ["kijun120", "env_low", "close"]}))
    others = sig[["kijun120", "env_low", "ma60"]]
    near = (others.sub(sig["line"], axis=0).abs().div(sig["close"], axis=0) <= p.overlap_pct) & others.ne(sig["line"], axis=0)
    df["sig_overlap"] = near.any(axis=1)
    ov = df.set_index(["code", "date"])["sig_overlap"].loc[list(zip(b0["code"], b0["entry_date"]))].to_numpy()
    show("B0 중 중첩 신호 (3% 이내)", stats(b0[ov]))
    show("B0 중 비중첩 신호", stats(b0[~ov]))

    print("\n== H3: 추가 매수")
    for label, r in [("2차 매수 없음", replace(base, second_units=0.0)),
                     ("2차 1.5배", replace(base, second_units=1.5)),
                     ("2차 -3%", replace(base, second_drop=0.03)),
                     ("2차 -5%", replace(base, second_drop=0.05)),
                     ("2차 60일선 인근 조건 (원문 문구)", replace(base, second_require_ma60=True))]:
        show(label, stats(run(df, f["setup"], r)[0]))

    print("\n== H4: 25% 반등 선도달 (1차 체결가 기준, 60거래일, 손절선 = 1차 x 0.96 x 0.97 종가)")
    breakeven = 0.0688 / (0.25 + 0.0688)
    for name, (a, b) in PERIODS.items():
        x = b0[(b0["entry_date"] >= a) & (b0["entry_date"] <= b)]
        ps = path_stats(x, df)
        k = (ps["first"] == "up").sum()
        n = len(ps)
        rate = k / n
        se = np.sqrt(rate * (1 - rate) / n)
        print(f"  {name}: +25% 먼저 {k}/{n} = {rate:.3f} (±{1.96 * se:.3f}), 손절 먼저 {(ps['first'] == 'stop').mean():.3f},"
              f" 둘 다 없음 {(ps['first'] == 'none').mean():.3f}, 손익분기 확률 {breakeven:.3f}")
    show("익절 고정 +25% 전량", stats(run(df, f["setup"], replace(base, tp_mode="fixed", tp_fixed=0.25))[0]))

    print("\n== 손절·시간 청산")
    idx = pd.read_parquet("data/raw/index.parquet").sort_values(["market", "date"])
    idx["ret"] = idx.groupby("market")["close"].pct_change()
    crash = idx[idx["ret"] <= -0.08]
    two = idx.assign(prev=idx.groupby("market")["ret"].shift(1)).query("ret <= -0.08 and prev <= -0.08")
    print(f"  지수 일간 -8% 이하(서킷브레이커 1단계 기준) {len(crash)}일, 2일 연속 {len(two)}회:",
          crash[["market", "date", "ret"]].assign(ret=lambda z: z["ret"].round(3)).to_dict("records"))
    show("손절 없음 (= 원문 문구 그대로)", stats(run(df, f["setup"], replace(base, use_stop=False))[0]))
    for d in (20, 120):
        show(f"시간 청산 {d}거래일", stats(run(df, f["setup"], replace(base, max_hold_days=d))[0]))

    print("\n== 정의 대안 (02장)")
    show("120일 기준선 = 120일 이동평균", stats(run(df, f["setup"], replace(base, first_line="ma120_env"))[0]))
    leader_alt = {"거래대금 = 파동 안 하루 5,000억": f["wave"] & f["kki"] & f["amt_day"],
                  "거래대금 = 그날 상위 0.5%": f["wave"] & f["kki"] & f["amt_pct"],
                  "끼 = 최근 120거래일 상한가": f["wave"] & f["kki_hist"] & f["amt_cum"]}
    for label, lead in leader_alt.items():
        show(label, stats(run(df, lead & f["pullback"], base)[0]))
    for pct in (0.10, 0.15):
        p2 = replace(p, env_pct=pct)
        d2 = build(raw, p2)
        f2 = flags(d2, p2)
        show(f"엔벨로프 이격률 {pct:.0%}", stats(run(d2, f2["setup"], base)[0]))


if __name__ == "__main__":
    main()
