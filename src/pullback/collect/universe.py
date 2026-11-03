"""04-2 분석 대상 종목(유니버스) 만들기.

- 현재 상장 종목: 키움 ka10099 (코스피·코스닥 보통주, 스팩·ETF·ETN·리츠 제외)
- 상장폐지 종목: FinanceDataReader KRX-DELISTING (기본 2021-01-01 이후 폐지된 보통주)

사용:
    uv run python -m pullback.collect.universe --delisted-since 2021-01-01
"""
from __future__ import annotations

import argparse
import sys

import FinanceDataReader as fdr
import pandas as pd

from pullback.kiwoom import KiwoomClient
from pullback.paths import UNIVERSE

sys.stdout.reconfigure(encoding="utf-8")

MARKETS = {"0": "KOSPI", "10": "KOSDAQ"}


def is_common(code: str, name: str) -> bool:
    """보통주 판별: 종목코드 끝자리가 0 이고 스팩이 아닌 종목."""
    return code.endswith("0") and "스팩" not in name


def listed(kw: KiwoomClient) -> pd.DataFrame:
    frames = []
    for mrkt, market in MARKETS.items():
        df = pd.DataFrame(kw.stock_list(mrkt))
        df = df[df["marketName"].isin(["거래소", "코스닥"])]
        df = df[[is_common(c, n) for c, n in zip(df["code"], df["name"])]]
        frames.append(pd.DataFrame({
            "code": df["code"], "name": df["name"], "market": market,
            "listing_date": pd.to_datetime(df["regDay"], format="%Y%m%d", errors="coerce"),
            "delisting_date": pd.NaT, "status": "listed", "source": "kiwoom", "reason": None,
        }))
    return pd.concat(frames, ignore_index=True)


def delisted(since: str) -> pd.DataFrame:
    d = fdr.StockListing("KRX-DELISTING")
    d["DelistingDate"] = pd.to_datetime(d["DelistingDate"])
    d = d[(d["DelistingDate"] >= since) & (d["Market"].isin(["KOSPI", "KOSDAQ"]))
          & (d["SecuGroup"] == "주권")]
    d = d[[is_common(str(c), str(n)) for c, n in zip(d["Symbol"], d["Name"])]]
    return pd.DataFrame({
        "code": d["Symbol"], "name": d["Name"], "market": d["Market"],
        "listing_date": pd.to_datetime(d["ListingDate"]), "delisting_date": d["DelistingDate"],
        "status": "delisted", "source": "fdr", "reason": d["Reason"],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delisted-since", default="2021-01-01")
    a = ap.parse_args()
    kw = KiwoomClient()
    u = pd.concat([listed(kw), delisted(a.delisted_since)], ignore_index=True)
    # 폐지 후 같은 코드로 재상장한 경우 상장 쪽을 남긴다 ("delisted" < "listed" 정렬 후 마지막 행)
    u = u.sort_values("status").drop_duplicates("code", keep="last").reset_index(drop=True)
    UNIVERSE.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(UNIVERSE, index=False)
    print(u.groupby(["status", "market"]).size().to_string())
    print(f"저장: {UNIVERSE} ({len(u)}종목)")


if __name__ == "__main__":
    main()
