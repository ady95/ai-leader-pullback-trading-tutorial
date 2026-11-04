"""06장 B0 백테스트 실행.

기본 출력은 '실행 통계'(거래 수, 체결 방식, 보유 기간)뿐이다. 수익률은 07장에서
--show-returns 를 주어 처음 연다. 결과 파일에는 명세서 지문을 함께 남긴다.

사용:
    uv run python scripts/run_b0.py                     # 실행 통계
    uv run python scripts/run_b0.py --strict-touch      # 엄격 체결 가정
    uv run python scripts/run_b0.py --show-returns      # 07장
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time

import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.paths import DATA
from pullback.spec import check, load, to_params

sys.stdout.reconfigure(encoding="utf-8")


def period_of(d: pd.Timestamp, p: dict) -> str:
    if d < pd.Timestamp(p["dev_start"]):
        return "warmup"
    if d <= pd.Timestamp(p["dev_end"]):
        return "dev"
    if d <= pd.Timestamp(p["validation_end"]):
        return "validation"
    return "holdout"


def table_digest(t: pd.DataFrame) -> str:
    cols = ["code", "entry_date", "entry_price", "exit_date", "exit_reason", "units_max"]
    return hashlib.sha256(pd.util.hash_pandas_object(t[cols], index=False).values.tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/b0.toml")
    ap.add_argument("--strict-touch", action="store_true")
    ap.add_argument("--show-returns", action="store_true")
    ap.add_argument("--periods", default="dev", help="수익률을 열 구간 (쉼표 구분). holdout 은 14장에서만")
    a = ap.parse_args()

    spec, digest = load(a.spec)
    assert not check(spec), check(spec)
    t0 = time.perf_counter()
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    r = Rules.from_spec(spec, strict_touch=a.strict_touch)
    trades, events = run(df, f["setup"], r)
    trades["period"] = trades["entry_date"].map(lambda d: period_of(d, spec["periods"]))
    tag = f"{spec['meta']['id']}_v{spec['meta']['version']}" + ("_strict" if a.strict_touch else "")
    out = DATA / "results"
    out.mkdir(parents=True, exist_ok=True)
    trades.assign(spec_sha256=digest).to_parquet(out / f"{tag}_trades.parquet", index=False)
    events.to_parquet(out / f"{tag}_events.parquet", index=False)
    data_fp = hashlib.sha256(pd.util.hash_pandas_object(df[["code", "date", "close", "volume"]], index=False)
                             .values.tobytes()).hexdigest()[:16]
    print(f"{tag} spec {digest[:16]} 실행 {time.perf_counter() - t0:.0f}s, 거래표 지문 {table_digest(trades)}")
    print(f"데이터 {len(df):,}행 · {df['code'].nunique():,}종목 · ~{df['date'].max():%Y-%m-%d}, 데이터 지문 {data_fp}")

    print("\n== 실행 통계 (수익률 제외)")
    g = trades.groupby("period")
    stat = pd.DataFrame({
        "거래": g.size(),
        "종목": g["code"].nunique(),
        "갭체결": g["gap_fill"].mean().round(3),
        "2차체결": g["second_date"].apply(lambda s: s.notna().mean()).round(3),
        "보유일_중앙": g["hold_days"].median(),
    }).reindex(["warmup", "dev", "validation", "holdout"])
    print(stat.to_string())
    ev_n = events.groupby("kind").size().to_dict()
    print("체결 이벤트", ev_n)

    if a.show_returns:
        show = [s.strip() for s in a.periods.split(",")]
        if "holdout" in show:
            print("\n[경고] 최종 홀드아웃은 14장에서 한 번만 연다 (03-3).")
        sub = trades[trades["period"].isin(show)]
        print("\n== 수익률 (" + ", ".join(show) + ")")
        print(sub.groupby("period")["net_ret"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
