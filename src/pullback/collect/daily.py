"""04-2 일봉 수집.

- 상장 종목: 키움 ka10081 수정주가 일봉 (start 이전까지 연속 조회)
- 폐지 종목: FinanceDataReader 일봉 (거래대금 없음 → amount 는 결측)
- 종목별 data/raw/daily/<코드>.parquet. 이미 있는 파일은 건너뛴다 (중단 후 재실행 가능)

사용:
    uv run python -m pullback.collect.daily --start 2020-01-01 --end 2026-11-03
"""
from __future__ import annotations

import argparse
import sys
import time

import FinanceDataReader as fdr
import pandas as pd

from pullback.kiwoom import KiwoomClient
from pullback.paths import DAILY_DIR, UNIVERSE

sys.stdout.reconfigure(encoding="utf-8")

COLS = ["date", "open", "high", "low", "close", "volume", "amount", "adj_flag", "adj_rate", "source"]


def _num(s: pd.Series) -> pd.Series:
    # 키움 시세 문자열에는 등락 부호(+/-)가 붙을 수 있다 → 부호를 떼고 숫자로
    return pd.to_numeric(s.astype(str).str.lstrip("+-"), errors="coerce")


def kiwoom_daily(kw: KiwoomClient, code: str, start: str, end: str) -> pd.DataFrame:
    body = {"stk_cd": code, "base_dt": end.replace("-", ""), "upd_stkpc_tp": "1"}
    rows: list[dict] = []
    for chunk in kw.paginate("ka10081", "/api/dostk/chart", body, "stk_dt_pole_chart_qry"):
        chunk = [r for r in chunk if r.get("dt")]   # 폐지·미존재 종목은 빈 행 1개가 온다
        rows.extend(chunk)
        if not chunk or min(r["dt"] for r in chunk) < start.replace("-", ""):
            break
    if not rows:
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["dt"], format="%Y%m%d"),
        "open": _num(df["open_pric"]), "high": _num(df["high_pric"]),
        "low": _num(df["low_pric"]), "close": _num(df["cur_prc"]),
        "volume": _num(df["trde_qty"]),
        "amount": _num(df["trde_prica"]) * 1_000_000,   # 거래대금 단위: 백만 원 → 원
        "adj_flag": df["upd_stkpc_tp"].replace("", None),
        "adj_rate": _num(df["upd_rt"]),
        "source": "kiwoom",
    })
    out = out[out["date"] >= start].drop_duplicates("date")
    return out.sort_values("date").reset_index(drop=True)


def fdr_daily(code: str, start: str, end: str) -> pd.DataFrame:
    x = fdr.DataReader(code, start, end)
    if x.empty:
        return pd.DataFrame(columns=COLS)
    x = x.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high",
                                        "Low": "low", "Close": "close", "Volume": "volume"})
    x["amount"] = float("nan")
    x["adj_flag"], x["adj_rate"], x["source"] = None, float("nan"), "fdr"
    return x[COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-11-03")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N종목만 (시험용)")
    a = ap.parse_args()
    u = pd.read_parquet(UNIVERSE)
    if a.limit:
        u = u.head(a.limit)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    kw = KiwoomClient()
    t0, done, empty, failed = time.time(), 0, 0, []
    for i, r in enumerate(u.itertuples(), 1):
        path = DAILY_DIR / f"{r.code}.parquet"
        if path.exists():
            continue
        try:
            if r.status == "listed":
                df = kiwoom_daily(kw, r.code, a.start, a.end)
            else:
                df = fdr_daily(r.code, a.start, str(r.delisting_date.date()))
        except Exception as e:  # noqa: BLE001 - 종목 하나 실패로 전체 수집을 멈추지 않는다
            failed.append((r.code, str(e)[:80]))
            continue
        df.to_parquet(path, index=False)
        done += 1
        empty += df.empty
        if i % 100 == 0:
            print(f"{i}/{len(u)} 저장 {done} 빈 파일 {empty} 실패 {len(failed)} "
                  f"호출 {kw.calls} 재시도 {kw.retries} {time.time() - t0:.0f}s", flush=True)
    print(f"완료: 저장 {done} 빈 파일 {empty} 실패 {len(failed)} 호출 {kw.calls} 재시도 {kw.retries} "
          f"{time.time() - t0:.0f}s")
    for c, m in failed[:20]:
        print("  실패", c, m)


if __name__ == "__main__":
    main()
