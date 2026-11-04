"""05-2 키움 테마 구성 스냅샷 (ka90001 테마그룹, ka90002 테마구성종목).

키움 테마 API는 '지금' 구성만 돌려준다. 과거 시점 구성은 얻을 수 없으므로
오늘부터 매일 스냅샷을 쌓아 두고, 백테스트에서는 비교·검증용으로만 쓴다.

사용 (장 마감 뒤 하루 한 번):
    uv run python -m pullback.collect.theme_snapshot
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from pullback.kiwoom import KiwoomClient
from pullback.paths import THEME_DIR

sys.stdout.reconfigure(encoding="utf-8")
PATH = "/api/dostk/thme"


def groups(kw: KiwoomClient) -> pd.DataFrame:
    body = {"qry_tp": "0", "stk_cd": "", "date_tp": "1", "thema_nm": "", "flu_pl_amt_tp": "1", "stex_tp": "1"}
    rows: list[dict] = []
    for chunk in kw.paginate("ka90001", PATH, body, "thema_grp"):
        rows.extend(chunk)
    return pd.DataFrame(rows)[["thema_grp_cd", "thema_nm", "stk_num"]].drop_duplicates("thema_grp_cd")


def members(kw: KiwoomClient, grp: str) -> list[dict]:
    j, _ = kw.request("ka90002", PATH, {"date_tp": "1", "thema_grp_cd": grp, "stex_tp": "1"})
    return j.get("thema_comp_stk") or []


def main() -> None:
    kw = KiwoomClient()
    g = groups(kw)
    rows = []
    for cd, nm in zip(g["thema_grp_cd"], g["thema_nm"]):
        for m in members(kw, cd):
            rows.append({"thema_grp_cd": cd, "thema_nm": nm, "code": m["stk_cd"], "name": m["stk_nm"]})
    snap = pd.DataFrame(rows)
    snap.insert(0, "snapshot_date", pd.Timestamp(date.today()))
    THEME_DIR.mkdir(parents=True, exist_ok=True)
    out = THEME_DIR / f"{date.today():%Y-%m-%d}.parquet"
    snap.to_parquet(out, index=False)
    per_stock = snap.groupby("code").size()
    print(f"테마 {g.shape[0]}개, 구성 {len(snap)}행, 종목 {snap['code'].nunique()}개,"
          f" 종목당 테마 중앙값 {per_stock.median():.0f}, 호출 {kw.calls}회 -> {out}")


if __name__ == "__main__":
    main()
