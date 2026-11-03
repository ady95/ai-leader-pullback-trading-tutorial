"""04-1 실측: 키움 REST API 일봉·분봉 조회 범위, 연속 조회, 호출 제한.

사용:
    uv run python scripts/measure_04_1.py --out data/measure_04_1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

from pullback.kiwoom import KiwoomClient, KiwoomError

sys.stdout.reconfigure(encoding="utf-8")


def measure_daily(kw: KiwoomClient, code: str, base_dt: str) -> dict:
    t = time.time()
    pages = 0
    rows = []
    for chunk in kw.paginate("ka10081", "/api/dostk/chart",
                             {"stk_cd": code, "base_dt": base_dt, "upd_stkpc_tp": "1"},
                             "stk_dt_pole_chart_qry"):
        pages += 1
        rows.extend(chunk)
    dates = sorted(r["dt"] for r in rows)
    return {"code": code, "pages": pages, "rows": len(rows), "first": dates[0], "last": dates[-1],
            "dup_dates": len(dates) - len(set(dates)), "sec": round(time.time() - t, 1)}


def measure_minute(kw: KiwoomClient, code: str, max_pages: int) -> dict:
    t = time.time()
    pages = 0
    rows = []
    for chunk in kw.paginate("ka10080", "/api/dostk/chart",
                             {"stk_cd": code, "tic_scope": "1", "upd_stkpc_tp": "1"},
                             "stk_min_pole_chart_qry", max_pages=max_pages):
        pages += 1
        rows.extend(chunk)
    ts = sorted(r["cntr_tm"] for r in rows)
    days = sorted({x[:8] for x in ts})
    hhmm = Counter(x[8:12] for x in ts)
    outside = sum(v for k, v in hhmm.items() if not ("0900" <= k <= "1530"))
    return {"code": code, "pages": pages, "rows": len(rows), "first": ts[0], "last": ts[-1],
            "trading_days": len(days), "rows_outside_0900_1530": outside,
            "earliest_hhmm": min(hhmm), "latest_hhmm": max(hhmm),
            "signed_price_rows": sum(1 for r in rows if r["cur_prc"][:1] in "+-"),
            "hit_max_pages": pages >= max_pages, "sec": round(time.time() - t, 1)}


def measure_rate(kw: KiwoomClient, n: int) -> dict:
    """요청 간격 없이 n 번 연속 호출해 호출 제한 응답이 몇 번 나오는지 센다."""
    kw.min_interval = 0
    ok = limited = other = 0
    msgs = Counter()
    t = time.time()
    for _ in range(n):
        try:
            kw.request("ka10081", "/api/dostk/chart",
                       {"stk_cd": "005930", "base_dt": "20261103", "upd_stkpc_tp": "1"}, max_retry=0)
            ok += 1
        except KiwoomError as e:
            if "허용된 요청" in e.msg or str(e.code) in ("5", "429"):
                limited += 1
            else:
                other += 1
            msgs[f"{e.code}:{e.msg[:40]}"] += 1
    sec = time.time() - t
    return {"n": n, "ok": ok, "limited": limited, "other": other, "sec": round(sec, 2),
            "req_per_sec": round(n / sec, 1), "messages": dict(msgs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/measure_04_1.json")
    ap.add_argument("--minute-pages", type=int, default=300)
    ap.add_argument("--rate-n", type=int, default=40)
    a = ap.parse_args()
    kw = KiwoomClient()
    res = {"mode": kw.mode, "measured_at": time.strftime("%Y-%m-%d %H:%M")}
    res["daily"] = [measure_daily(kw, c, "20261103") for c in ("005930", "000660", "035720")]
    print(json.dumps(res["daily"], ensure_ascii=False, indent=1))
    res["minute"] = measure_minute(kw, "005930", a.minute_pages)
    print(json.dumps(res["minute"], ensure_ascii=False, indent=1))
    res["rate"] = measure_rate(kw, a.rate_n)
    print(json.dumps(res["rate"], ensure_ascii=False, indent=1))
    res["calls"], res["retries"] = kw.calls, kw.retries
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
