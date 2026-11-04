"""07장 성과 지표.

거래 단위: 평균 순수익률, 승률, 손익비, Profit Factor, 최악 거래, 월 단위 군집 부트스트랩 신뢰구간
계좌 단위: 1차 매수 금액을 고정(unit_money)한 일별 평가 손익 곡선과 MDD
경로 지표: MAE·MFE, 목표가·손절가 선도달
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trade_stats(r: pd.Series) -> dict:
    r = r.dropna()
    win, loss = r[r > 0], r[r <= 0]
    return {
        "n": int(len(r)),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(win.mean()) if len(win) else np.nan,
        "avg_loss": float(loss.mean()) if len(loss) else np.nan,
        "payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan,
        "profit_factor": float(win.sum() / -loss.sum()) if len(loss) else np.nan,
        "worst": float(r.min()),
        "best": float(r.max()),
    }


def bootstrap_mean_ci(r: pd.Series, groups: pd.Series, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """같은 달에 진입한 거래를 한 묶음으로 다시 뽑는 군집 부트스트랩 (시장 공통 충격 반영)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"r": r.to_numpy(), "g": groups.to_numpy()}).dropna()
    keys = df["g"].unique()
    sums = df.groupby("g")["r"].sum().reindex(keys).to_numpy()
    cnts = df.groupby("g")["r"].size().reindex(keys).to_numpy()
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def equity_curve(events: pd.DataFrame, trades: pd.DataFrame, daily: pd.DataFrame, r,
                 unit_money: float = 5e6, capital: float = 1e8) -> pd.DataFrame:
    """거래마다 1차 매수 금액을 unit_money 로 맞춰 일별 평가 손익을 더한다 (복리 없음)."""
    closes = daily.set_index(["code", "date"])["close"]
    parts = []
    for t in trades.itertuples():
        g = events[(events["code"] == t.code) & (events["date"] >= t.entry_date) & (events["date"] <= t.exit_date)]
        g = g[g["kind"] != "defer"]
        k = unit_money / t.entry_price
        px = closes.loc[t.code].loc[t.entry_date:t.exit_date]
        shares = pd.Series(0.0, index=px.index)
        cash = pd.Series(0.0, index=px.index)
        invested = pd.Series(0.0, index=px.index)
        for e in g.itertuples():
            q = e.units * k
            if e.kind.startswith("buy"):
                shares.loc[e.date:] += q
                cash.loc[e.date:] -= e.price * q * (1 + r.commission + r.slippage)
                invested.loc[e.date:] += e.price * q
            else:
                fee = r.commission + r.slippage + r.tax(pd.Timestamp(e.date).year)
                shares.loc[e.date:] -= q
                cash.loc[e.date:] += e.price * q * (1 - fee)
                invested.loc[e.date:] -= e.price * q   # 근사: 매도 금액만큼 노출 감소
        pnl = cash + shares.clip(lower=0) * px
        parts.append(pd.DataFrame({"date": px.index, "pnl": pnl.to_numpy(),
                                   "exposure": (shares.clip(lower=0) * px).to_numpy(), "trade": t.Index}))
    p = pd.concat(parts)
    # 청산 뒤에는 실현 손익이 그대로 남는다
    last = p.groupby("trade").tail(1).set_index("trade")["pnl"]
    days = pd.Index(sorted(daily["date"].unique()))
    days = days[(days >= p["date"].min()) & (days <= p["date"].max())]
    open_pnl = p.pivot_table(index="date", columns="trade", values="pnl").reindex(days)
    exp = p.pivot_table(index="date", columns="trade", values="exposure").reindex(days).fillna(0).sum(axis=1)
    exit_day = p.groupby("trade")["date"].max()
    filled = open_pnl.copy()
    for tr in filled.columns:
        after = filled.index > exit_day[tr]
        filled.loc[after, tr] = last[tr]
    total = filled.fillna(0).sum(axis=1)
    eq = capital + total
    return pd.DataFrame({"equity": eq, "exposure": exp, "dd": eq / eq.cummax() - 1})


def path_stats(trades: pd.DataFrame, daily: pd.DataFrame, up: float = 0.25, stop: float = None,
               horizon: int = 60) -> pd.DataFrame:
    """1차 체결가 기준 MAE·MFE(보유 기간)와, 진입 뒤 horizon 거래일 안 '+up 선도달' vs '손절 종가 선도달'."""
    g = {c: x.reset_index(drop=True) for c, x in daily[["code", "date", "high", "low", "close"]].groupby("code")}
    rows = []
    for t in trades.itertuples():
        x = g[t.code]
        i0 = int(x.index[x["date"] == t.entry_date][0])
        i1 = int(x.index[x["date"] == t.exit_date][0])
        held = x.iloc[i0:i1 + 1]
        p1 = t.entry_price
        stop_line = p1 * (1 - 0.04) * (1 - 0.03) if stop is None else p1 * (1 - stop)
        fut = x.iloc[i0 + 1:i0 + 1 + horizon]          # 체결 다음 날부터
        hit_up = fut.index[fut["high"] >= p1 * (1 + up)]
        hit_dn = fut.index[fut["close"] < stop_line]
        a = hit_up[0] if len(hit_up) else np.inf
        b = hit_dn[0] if len(hit_dn) else np.inf
        first = "up" if a < b else ("stop" if b <= a and np.isfinite(b) else "none")
        rows.append({"mae": held["low"].min() / p1 - 1, "mfe": held["high"].max() / p1 - 1, "first": first})
    return pd.DataFrame(rows, index=trades.index)
