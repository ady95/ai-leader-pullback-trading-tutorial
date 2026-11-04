"""02장 주도주·상승 파동·눌림목·지지선의 정량 정의.

원문 전략이 수치를 밝히지 않은 항목은 '잠정 기본값'으로 두고, 대안 값은
민감도 실험(07장 이후)에서 비교한다. 모든 계산은 t일 장 마감까지 알 수 있는
값만 쓴다(미래 데이터 금지).

사용:
    from pullback.definitions import Params, load_daily, build
    df = build(load_daily(), Params())
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pullback.storage import connect


@dataclass(frozen=True)
class Params:
    # 02-1 주도주
    limit_up_ret: float = 0.295        # 상한가로 보는 종가 등락률 (가격제한폭 30%, 반올림 여유)
    history_days: int = 120            # 대안 끼 정의(최근 N거래일 상한가)의 기간
    amount_min: float = 5e11           # 원문 하한 5,000억 원 (상승 파동 누적 기준)
    amount_top_pct: float = 0.995      # 대안: 그날 전 종목 거래대금 상위 0.5%
    # 02-2 상승 파동과 조정
    wave_window: int = 60              # 파동 고점을 찾는 기간(거래일)
    wave_rise_min: float = 0.50        # 저점 대비 고점 상승률 하한
    pullback_min: float = 0.10         # 고점 대비 최소 하락폭
    pullback_max: float = 0.40         # 고점 대비 최대 하락폭 (이보다 깊으면 추세 붕괴로 봄)
    pullback_days_min: int = 3         # 고점 이후 최소 경과 거래일
    # 02-3 지지선
    env_period: int = 20               # 엔벨로프 중심 이동평균 기간
    env_pct: float = 0.20              # 엔벨로프 하단 이격률
    overlap_pct: float = 0.03          # 지지선 중첩 허용 오차(종가 대비)
    atr_period: int = 14

    def as_dict(self) -> dict:
        return asdict(self)


def load_daily(start: str = "2020-01-01") -> pd.DataFrame:
    con = connect(read_only=True)
    df = con.execute(f"""
        SELECT code, date, open, high, low, close, volume, amount_filled AS amount,
               amount_is_est, adj_flag, source
        FROM daily WHERE date >= DATE '{start}' ORDER BY code, date
    """).fetchdf()
    con.close()
    return df


def _wave(close: np.ndarray, amount: np.ndarray, lu: np.ndarray, w: int):
    """t일까지의 최근 w거래일에서 고점(P)과 그 이전 저점(L)을 찾는다."""
    n = len(close)
    tp = np.empty(n, dtype=np.int64)
    tl = np.empty(n, dtype=np.int64)
    dq: list[int] = []          # 고점 후보 인덱스 (종가 내림차순 유지)
    head = 0
    for t in range(n):
        while len(dq) > head and close[dq[-1]] <= close[t]:
            dq.pop()
        dq.append(t)
        while dq[head] <= t - w:
            head += 1
        p = dq[head]
        s = max(0, t - w + 1)
        tp[t] = p
        tl[t] = s + int(np.argmin(close[s:p + 1]))
    csum_amt = np.concatenate([[0.0], np.cumsum(np.nan_to_num(amount))])
    csum_lu = np.concatenate([[0], np.cumsum(lu)])
    amt_sum = csum_amt[tp + 1] - csum_amt[tl]
    lu_cnt = csum_lu[tp + 1] - csum_lu[tl]
    return tp, tl, amt_sum, lu_cnt


def _range_max(a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """구간 [lo, hi] 최댓값 (sparse table, 구간 길이는 wave_window 이하)."""
    a = np.nan_to_num(a, nan=-np.inf)
    table = [a]
    k = 1
    while (1 << k) <= len(a):
        prev = table[-1]
        half = 1 << (k - 1)
        table.append(np.maximum(prev[:-half], prev[half:]))
        k += 1
    length = hi - lo + 1
    j = np.floor(np.log2(length)).astype(int)
    out = np.empty(len(lo))
    for jj in np.unique(j):
        m = j == jj
        t = table[jj]
        out[m] = np.maximum(t[lo[m]], t[hi[m] - (1 << jj) + 1])
    return out


def build(df: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", sort=False)

    # 기본 지표
    df["prev_close"] = g["close"].shift(1)
    df["ret"] = df["close"] / df["prev_close"] - 1
    df["halted"] = df["volume"] == 0
    df["limit_up"] = (df["ret"] >= p.limit_up_ret) & ~df["halted"]
    df["amt_pct"] = df.groupby("date")["amount"].rank(pct=True)

    roll = lambda col, n, fn: g[col].transform(lambda s: getattr(s.rolling(n, min_periods=n), fn)())
    df["ma60"] = roll("close", 60, "mean")
    df["ma120"] = roll("close", 120, "mean")
    df["kijun120"] = (roll("high", 120, "max") + roll("low", 120, "min")) / 2
    df["env_mid"] = roll("close", p.env_period, "mean")
    df["env_low"] = df["env_mid"] * (1 - p.env_pct)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["prev_close"]).abs(),
                    (df["low"] - df["prev_close"]).abs()], axis=1).max(axis=1)
    df["atr"] = tr.groupby(df["code"]).transform(lambda s: s.rolling(p.atr_period, min_periods=p.atr_period).mean())

    # 02-1 주도주 이력 (최근 history_days 거래일)
    h = p.history_days
    df["lu_cnt_hist"] = g["limit_up"].transform(lambda s: s.astype(int).rolling(h, min_periods=1).sum())

    # 02-2 상승 파동
    parts = []
    for _, x in df.groupby("code", sort=False):
        c = x["close"].to_numpy(float)
        tp, tl, amt_sum, lu_cnt = _wave(c, x["amount"].to_numpy(float), x["limit_up"].to_numpy(int), p.wave_window)
        parts.append(pd.DataFrame({
            "peak_close": c[tp], "trough_close": c[tl],
            "peak_date": x["date"].to_numpy()[tp], "trough_date": x["date"].to_numpy()[tl],
            "days_since_peak": np.arange(len(c)) - tp, "rise_days": tp - tl,
            "wave_amt_sum": amt_sum, "wave_lu_cnt": lu_cnt,
            "wave_amt_max": _range_max(x["amount"].to_numpy(float), tl, tp),
            "wave_amt_pct_max": _range_max(x["amt_pct"].to_numpy(float), tl, tp),
        }, index=x.index))
    df = df.join(pd.concat(parts))
    df["wave_rise"] = df["peak_close"] / df["trough_close"] - 1
    df["drawdown"] = df["close"] / df["peak_close"] - 1
    return df


def flags(df: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """정의별 참/거짓 열을 붙인다. 열 이름이 곧 2장 정의의 이름이다."""
    out = pd.DataFrame(index=df.index)
    out["kki"] = df["wave_lu_cnt"] >= 1                                  # 끼: 상승 파동 안 상한가 (원문)
    out["kki_hist"] = df["lu_cnt_hist"] >= 1                             # 대안: 최근 120거래일 상한가
    out["wave"] = (df["wave_rise"] >= p.wave_rise_min) & (df["days_since_peak"] >= 1)
    # 거래대금은 상승 파동 구간(저점~고점) 안에서 센다
    out["amt_cum"] = df["wave_amt_sum"] >= p.amount_min                  # B0: 파동 누적 5,000억 (원문 사례로 확인)
    out["amt_day"] = df["wave_amt_max"] >= p.amount_min                  # 대안: 파동 안 하루 5,000억
    out["amt_pct"] = df["wave_amt_pct_max"] >= p.amount_top_pct          # 대안: 그날 상위 0.5%
    out["pullback"] = (out["wave"] & (df["days_since_peak"] >= p.pullback_days_min)
                       & (df["drawdown"] <= -p.pullback_min) & (df["drawdown"] >= -p.pullback_max)
                       & ~df["halted"])
    out["leader"] = out["wave"] & out["kki"] & out["amt_cum"]            # B0 주도주 (재료 조건은 10장)
    out["setup"] = out["leader"] & out["pullback"]                       # B0 눌림목 후보일
    return out
