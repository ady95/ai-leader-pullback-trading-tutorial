"""15-2 장 마감 배치의 데이터·신호 모듈.

매일 밤 전 종목(약 2,700개)을 다시 받지 않는다. B0 후보가 되려면 상승 파동 안에 상한가가 있어야 하고
파동은 최근 60거래일 안에서 찾으므로, 다음 종목만 갱신하면 된다.
  감시 목록 = 전날까지 최근 60거래일 안에 상한가가 있었던 종목
           + 오늘 종가 +29.5% 이상 종목 (ka10027 전일대비등락률상위)
           + 보유 종목, 주문 중인 종목
종목마다 ka10081 한 페이지(최근 약 600거래일)를 다시 받아 통째로 바꾼다. 수정주가가 바뀌어도
과거 값까지 함께 다시 계산되므로, 어제 받은 값에 오늘 하루를 덧붙이는 방식보다 안전하다.
"""
from __future__ import annotations

import pandas as pd

from pullback.backtest import first_buy_price
from pullback.collect.daily import kiwoom_daily
from pullback.collect.universe import is_common
from pullback.definitions import Params, build, flags
from pullback.kiwoom import KiwoomClient
from pullback.paths import DATA

LIVE_DAILY = DATA / "live" / "daily"
WATCH_BARS = 60


def watchlist(df: pd.DataFrame, day: str, p: Params = Params(), bars: int = WATCH_BARS) -> set[str]:
    """day 까지(포함) 종목별 최근 bars 개 일봉 안에 상한가 종가가 있었던 종목 (day 에 일봉이 있는 종목만)."""
    x = df[df["date"] <= day].sort_values(["code", "date"])
    ret = x["close"] / x.groupby("code")["close"].shift(1) - 1
    lu = (ret >= p.limit_up_ret) & (x["volume"] > 0)
    recent = x.groupby("code").cumcount(ascending=False) < bars
    alive = x.groupby("code")["date"].transform("max") == x["date"].max()   # 폐지 종목은 마지막 일봉이 과거에 멈춰 있다
    return set(x.loc[lu & recent & alive, "code"])


def limit_up_today(kw: KiwoomClient, previous_day: bool = False) -> pd.DataFrame:
    """ka10017 상하한가요청. previous_day=True 면 '전일 상한' (updown_tp 6)."""
    body = {"mrkt_tp": "000", "updown_tp": "6" if previous_day else "1", "sort_tp": "1",
            "stk_cnd": "0", "trde_qty_tp": "0000", "crd_cnd": "0", "trde_gold_tp": "0", "stex_tp": "1"}
    rows = []
    for chunk in kw.paginate("ka10017", "/api/dostk/stkinfo", body, "updown_pric"):
        rows += chunk
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["code", "name", "flu_rt", "common"])
    out = pd.DataFrame({"code": df["stk_cd"].str.strip(), "name": df["stk_nm"],
                        "flu_rt": pd.to_numeric(df["flu_rt"].str.lstrip("+"), errors="coerce")})
    out["common"] = [is_common(c, n) for c, n in zip(out["code"], out["name"])]
    return out


def top_gainers(kw: KiwoomClient, min_ret: float) -> pd.DataFrame:
    """ka10027 전일대비등락률상위 (상승률 순). 등락률이 min_ret 이상인 종목만.
    ka10017 은 '상한가에 있는 종목'만 주므로, 종가 +29.5% 이상이지만 상한가가 아닌 날을 놓친다(15-2)."""
    body = {"mrkt_tp": "000", "sort_tp": "1", "trde_qty_cnd": "0000", "stk_cnd": "0", "crd_cnd": "0",
            "updown_incls": "1", "pric_cnd": "0", "trde_prica_cnd": "0", "stex_tp": "1"}
    rows = []
    for chunk in kw.paginate("ka10027", "/api/dostk/rkinfo", body, "pred_pre_flu_rt_upper"):
        rows += chunk
        if not chunk or float(chunk[-1]["flu_rt"].lstrip("+")) < min_ret * 100:
            break
    df = pd.DataFrame(rows)
    out = pd.DataFrame({"code": df["stk_cd"].str.strip(), "name": df["stk_nm"],
                        "flu_rt": pd.to_numeric(df["flu_rt"].str.lstrip("+"), errors="coerce")})
    out = out[out["flu_rt"] >= min_ret * 100].reset_index(drop=True)
    out["common"] = [is_common(c, n) for c, n in zip(out["code"], out["name"])]
    return out


def refresh(kw: KiwoomClient, codes: list[str], day: str, lookback_days: int = 400) -> dict[str, int]:
    """종목별 최근 일봉을 받아 data/live/daily/<code>.parquet 을 통째로 바꾼다. {code: 행 수}."""
    LIVE_DAILY.mkdir(parents=True, exist_ok=True)
    start = str((pd.Timestamp(day) - pd.Timedelta(days=lookback_days)).date())
    out = {}
    for code in codes:
        x = kiwoom_daily(kw, code, start, day)
        x.to_parquet(LIVE_DAILY / f"{code}.parquet", index=False)
        out[code] = len(x)
    return out


def load_live(codes: list[str] | None = None) -> pd.DataFrame:
    files = sorted(LIVE_DAILY.glob("*.parquet"))
    frames = []
    for f in files:
        if codes is not None and f.stem not in codes:
            continue
        x = pd.read_parquet(f)
        if len(x):
            frames.append(x.assign(code=f.stem))
    df = pd.concat(frames, ignore_index=True)
    df["amount_is_est"] = df["amount"].isna()
    return df


def signals(df: pd.DataFrame, day: str, p: Params = Params()) -> tuple[list[tuple[str, float, str]], pd.DataFrame]:
    """day 까지의 데이터만으로 B0 후보를 계산한다. ([(code, 1차 매수선, 파동 고점일)], day 의 지표 행)."""
    x = build(df[df["date"] <= day].copy(), p)
    f = flags(x, p)
    x["line"] = first_buy_price(x)
    today = x[x["date"] == day]
    su = today[f.loc[today.index, "setup"]]
    setups = [(c, float(ln), str(pd.Timestamp(pk).date())) for c, ln, pk in
              zip(su["code"], su["line"], su["peak_date"])]
    return setups, today
