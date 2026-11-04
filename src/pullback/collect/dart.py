"""10-1 DART 공시 목록 수집 (OpenDART list.json).

- 고유번호 파일(corpCode.xml)로 종목코드 -> 고유번호를 찾는다. 상장폐지 종목도 들어 있다.
- 종목별로 기간 전체를 조회한다 (고유번호를 지정하면 기간 제한이 없다).
- 공개 시점은 접수일자(rcept_dt)뿐이다. 시각은 없으므로 백테스트에서는
  "t일 장 마감 뒤 다음 날 주문"에 t일까지 접수된 공시만 쓴다.

사용:
    uv run python -m pullback.collect.dart                 # 후보 종목 전체
    uv run python -m pullback.collect.dart --codes 024060  # 일부만
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests
from dotenv import load_dotenv

from pullback.paths import DART_DIR, DATA, DISCLOSURES

sys.stdout.reconfigure(encoding="utf-8")
BASE = "https://opendart.fss.or.kr/api"
COLS = ["stock_code", "corp_code", "corp_cls", "report_nm", "rcept_no", "rcept_dt", "flr_nm", "rm"]


def _key() -> str:
    load_dotenv()
    load_dotenv("../.env")
    return os.environ["DART_API_KEY"]


def corp_codes(key: str) -> dict[str, str]:
    path = DART_DIR / "corpCode.xml"
    if not path.exists():
        r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": key}, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        DART_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(z.read(z.namelist()[0]))
    out = {}
    for e in ET.parse(path).getroot().iter("list"):
        sc = (e.findtext("stock_code") or "").strip()
        if sc:
            out[sc] = e.findtext("corp_code")
    return out


def fetch(key: str, corp: str, bgn: str, end: str) -> list[dict]:
    rows, page = [], 1
    while True:
        for attempt in range(5):
            try:
                j = requests.get(f"{BASE}/list.json", timeout=30, params={
                    "crtfc_key": key, "corp_code": corp, "bgn_de": bgn, "end_de": end,
                    "page_count": 100, "page_no": page}).json()
                break
            except (requests.RequestException, ValueError):
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"{corp} page {page} 실패")
        if j["status"] == "013":          # 조회된 데이터 없음
            return rows
        if j["status"] != "000":
            raise RuntimeError(f"{corp}: {j['status']} {j.get('message')}")
        rows.extend(j["list"])
        if page >= j["total_page"]:
            return rows
        page += 1
        time.sleep(0.1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*")
    ap.add_argument("--bgn", default="20200101")
    ap.add_argument("--end", default="20261103")
    a = ap.parse_args()
    key = _key()
    cc = corp_codes(key)
    codes = a.codes or sorted(pd.read_parquet(DATA / "candidates_a1.parquet")["code"].unique())
    done = pd.read_parquet(DISCLOSURES) if DISCLOSURES.exists() else pd.DataFrame(columns=COLS)
    have = set(done["stock_code"])
    parts, t0 = [done], time.perf_counter()
    for n, code in enumerate(codes, 1):
        if code in have:
            continue
        rows = fetch(key, cc[code], a.bgn, a.end)
        df = pd.DataFrame(rows, columns=list(set(COLS) | {"corp_name"}))
        df["stock_code"] = code
        parts.append(df.loc[:, COLS])
        if n % 100 == 0:
            pd.concat(parts, ignore_index=True).to_parquet(DISCLOSURES, index=False)
            print(f"{n}/{len(codes)} 종목, {time.perf_counter() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True).drop_duplicates("rcept_no")
    out["report_nm"] = out["report_nm"].str.strip()
    out.to_parquet(DISCLOSURES, index=False)
    print(f"공시 {len(out):,}건, 종목 {out['stock_code'].nunique():,}개 -> {DISCLOSURES}")


if __name__ == "__main__":
    main()
