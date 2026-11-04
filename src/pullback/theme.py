"""05-2 시장 상대강도와 가격 동반 움직임 기반 테마 특징량.

테마는 외부 분류를 쓰지 않고 가격에서 직접 추정한다.
  - 급등일: 종가 등락률 +15% 이상 (거래정지일 제외)
  - t일의 동반 종목(peer): 최근 60거래일 안에서 후보 종목과 같은 날 급등한 날이 2일 이상인 종목
모든 값은 t일까지의 데이터로만 계산하므로 과거 시점 정합성이 유지된다.

사용:
    from pullback.theme import add_market_features, add_theme_features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pullback.paths import INDEX, UNIVERSE


@dataclass(frozen=True)
class ThemeParams:
    surge_ret: float = 0.15     # 급등일 기준
    window: int = 60            # 동반 급등을 세는 기간(거래일)
    min_common: int = 2         # 같은 날 급등한 날 수 하한


MARKET_FEATURES = ["mkt_ret20", "mkt_gap_ma60", "rs20", "rs60"]
THEME_FEATURES = ["theme_n_peers", "theme_rank_amt", "theme_peer_dd", "theme_peer_ret5"]


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """종목이 속한 시장 지수 대비 상대강도와 시장 상태."""
    idx = pd.read_parquet(INDEX).sort_values(["market", "date"])
    gi = idx.groupby("market")
    idx["mkt_ret20"] = gi["close"].pct_change(20)
    idx["mkt_ret60"] = gi["close"].pct_change(60)
    idx["mkt_gap_ma60"] = idx["close"] / gi["close"].transform(lambda s: s.rolling(60).mean()) - 1
    mk = pd.read_parquet(UNIVERSE)[["code", "market"]]
    out = df.merge(mk, on="code", how="left").merge(
        idx[["market", "date", "mkt_ret20", "mkt_ret60", "mkt_gap_ma60"]], on=["market", "date"], how="left")
    g = out.groupby("code", sort=False)["close"]
    out["rs20"] = g.pct_change(20) - out["mkt_ret20"]
    out["rs60"] = g.pct_change(60) - out["mkt_ret60"]
    out.index = df.index
    return out


def add_theme_features(df: pd.DataFrame, mask: pd.Series, p: ThemeParams = ThemeParams()) -> pd.DataFrame:
    """mask 행(보통 B0 후보일)에 동반 급등 테마 특징량을 붙인다. peer 목록도 함께 돌려준다."""
    surge = ((df["ret"] >= p.surge_ret) & ~df["halted"]).astype(np.int32)
    piv = lambda s: pd.DataFrame({"date": df["date"], "code": df["code"], "v": s}).pivot(index="date", columns="code", values="v")
    S = piv(surge).fillna(0).to_numpy(np.int32)
    amt = piv(df["wave_amt_sum"])
    dd = piv(df["drawdown"]).to_numpy(float)
    r5 = piv(df["close"] / df.groupby("code")["close"].shift(5) - 1).to_numpy(float)
    dates, codes = amt.index, amt.columns
    amt = amt.to_numpy(float)
    col = {c: i for i, c in enumerate(codes)}
    pos = {d: i for i, d in enumerate(dates)}

    feats = pd.DataFrame(index=df.index[mask], columns=THEME_FEATURES, dtype=float)
    peers_out = {}
    rows = df.loc[mask, ["date", "code"]]
    for d, grp in rows.groupby("date"):
        t = pos[d]
        w = S[max(0, t - p.window + 1): t + 1]
        ci = np.array([col[c] for c in grp["code"]])
        common = w[:, ci].T @ w                         # 후보 x 전 종목: 같은 날 급등한 날 수
        common[np.arange(len(ci)), ci] = 0              # 자기 자신 제외
        for k, (ridx, c) in enumerate(zip(grp.index, grp["code"])):
            peer = np.flatnonzero(common[k] >= p.min_common)
            peers_out[ridx] = [codes[j] for j in peer]
            n = len(peer)
            mine = amt[t, ci[k]]
            feats.loc[ridx] = [
                n,
                1 + int(np.sum(amt[t, peer] > mine)) if n else 1,
                float(np.nanmedian(dd[t, peer])) if n else np.nan,
                float(np.nanmedian(r5[t, peer])) if n else np.nan,
            ]
    feats["peers"] = pd.Series(peers_out)
    return feats
