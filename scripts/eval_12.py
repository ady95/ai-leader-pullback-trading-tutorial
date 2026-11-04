"""12장 A3 위험관리 구성요소(P1 변동성 수량, C1 조건부 2차, E1 동시 보유 한도, E2 테마 한도, L1 월간 손실 한도).

사용:
    uv run python scripts/eval_12.py
    uv run python scripts/eval_12.py --skipped   # E1·E2·L1 이 건너뛴 거래의 연도별 성과
"""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace

import numpy as np
import pandas as pd

from pullback.backtest import Rules, run
from pullback.definitions import build, flags, load_daily
from pullback.metrics import equity_curve
from pullback.spec import load, to_params
from pullback.theme import add_theme_features

sys.stdout.reconfigure(encoding="utf-8")
A3 = tomllib.loads(open("configs/a3.toml", encoding="utf-8").read())
PER = {k: tuple(v) for k, v in A3["metrics"]["periods"].items()}
UNIT, CAPITAL = 1e6, 1e8
MAX_POS, THEME_MAX, MONTH_LOSS = 20, 2, 0.03


def money(t: pd.DataFrame) -> pd.Series:
    """거래의 총 매수 금액 (1차 + 2차), 1차 기준 금액 x w."""
    return t["invested"] * UNIT / t["entry_price"] * t.get("w", 1.0)


def rodc(t: pd.DataFrame) -> float:
    """투입 자본 연환산 수익률 = 손익 합 / sum(투입 금액 x 보유 거래일) x 250."""
    m = money(t)
    return float((t["net_ret"] * m).sum() / (m * t["hold_days"].clip(lower=1)).sum() * 250)


def account(t, e, df, r):
    x = t.assign(unit_money=UNIT * t.get("w", 1.0))
    eq = equity_curve(e, x, df, r, unit_money=UNIT, capital=CAPITAL)
    pnl = eq["equity"].iloc[-1] - CAPITAL
    mdd_won = float((eq["equity"] - eq["equity"].cummax()).min())
    exp = eq["exposure"].mean()
    return dict(n=len(t), mean=t["net_ret"].mean(), rodc=rodc(t), pnl=pnl / CAPITAL, mdd=eq["dd"].min(),
                exp=exp, mdd_exp=mdd_won / exp, worst=t["net_ret"].min(),
                worst_won=float((t["net_ret"] * money(t)).min()))


def in_per(t, name):
    a, b = PER[name]
    return t[t["entry_date"].between(a, b)]


def signal_index(trades: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """거래의 신호일(체결 전 거래일) 행 번호."""
    key = pd.MultiIndex.from_frame(df[["code", "date"]])
    prev = pd.Series(df.groupby("code").cumcount().to_numpy(), index=key)
    pos = prev.reindex(pd.MultiIndex.from_frame(trades[["code", "entry_date"]])).to_numpy()
    first = df.groupby("code").cumcount()
    idx_by = pd.Series(df.index.to_numpy(), index=pd.MultiIndex.from_arrays([df["code"], first]))
    return idx_by.reindex(pd.MultiIndex.from_arrays([trades["code"], pos - 1])).to_numpy()


def cap_positions(t: pd.DataFrame, max_pos: int) -> np.ndarray:
    """E1: 보유 종목 수가 max_pos 이면 새 진입을 건너뛴다. 청산일이 진입일보다 앞선 거래만 비운다."""
    keep = np.zeros(len(t), bool)
    held: list = []
    for k, (d, x) in enumerate(zip(t["entry_date"], t["exit_date"])):
        held = [e for e in held if e >= d]
        if len(held) < max_pos:
            keep[k] = True
            held.append(x)
    return keep


def cap_theme(t: pd.DataFrame, max_same: int) -> np.ndarray:
    """E2: 신호일 동반 급등 종목 중 보유 중인 종목이 max_same 개 이상이면 건너뛴다."""
    keep = np.zeros(len(t), bool)
    held: dict = {}
    for k, (code, d, x, peers) in enumerate(zip(t["code"], t["entry_date"], t["exit_date"], t["peers"])):
        held = {c: e for c, e in held.items() if e >= d}
        peers = peers if isinstance(peers, list) else []
        if sum(c in held for c in peers) < max_same:
            keep[k] = True
            held[code] = max(held.get(code, x), x)
    return keep


def month_loss_limit(t: pd.DataFrame, limit: float) -> np.ndarray:
    """L1: 그 달 이미 청산된 거래의 실현 손실 합이 자본의 limit 를 넘으면 그 달 새 진입을 건너뛴다."""
    keep = np.zeros(len(t), bool)
    pnl = (t["net_ret"] * money(t)).to_numpy()
    taken = []
    for k, (d, x) in enumerate(zip(t["entry_date"], t["exit_date"])):
        m = d.to_period("M")
        realized = sum(pnl[j] for j in taken if t["exit_date"].iloc[j] < d and t["exit_date"].iloc[j].to_period("M") == m)
        if realized > -limit * CAPITAL:
            keep[k] = True
            taken.append(k)
    return keep


def pct_rank(value, dist) -> float:
    return float((np.asarray(dist) < value).mean())


def skipped_report(df, setup, base) -> None:
    b0_t, _ = run(df, setup, base)
    b0_t = b0_t.sort_values(["entry_date", "code"]).reset_index(drop=True)
    sig = signal_index(b0_t, df).astype(int)
    mask = pd.Series(False, index=df.index)
    mask.loc[sig] = True
    b0_t["peers"] = add_theme_features(df, mask)["peers"].reindex(sig).to_numpy()
    # 진입일마다 보유 중인 B0 종목 수 (자기 자신 제외)
    ent, ext = b0_t["entry_date"].to_numpy(), b0_t["exit_date"].to_numpy()
    b0_t["held"] = [int(((ent < d) & (ext >= d)).sum()) for d in ent]
    for name, keep in [("E1", cap_positions(b0_t, MAX_POS)), ("E2", cap_theme(b0_t, THEME_MAX)),
                       ("L1", month_loss_limit(b0_t, MONTH_LOSS))]:
        x = b0_t.assign(skip=~keep, year=b0_t["entry_date"].dt.year)
        g = x.groupby(["year", "skip"])["net_ret"].agg(["size", "mean"]).unstack("skip")
        print(f"== {name} 건너뛴 거래 (연도 x [남김, 건너뜀])")
        print(g.round(4).to_string())
    x = b0_t.assign(year=b0_t["entry_date"].dt.year, busy=b0_t["held"] >= MAX_POS)
    print("== 진입 시 보유 종목 수 >= 20 인 B0 거래 (연도별 건수, 평균)")
    print(x.groupby(["year", "busy"])["net_ret"].agg(["size", "mean"]).unstack("busy").round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skipped", action="store_true")
    a = ap.parse_args()
    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    setup = f["setup"] & (df["date"] <= "2025-12-31")
    base = Rules.from_spec(spec)
    if a.skipped:
        return skipped_report(df, setup, base)
    rng = np.random.default_rng(A3["random_benchmark"]["seed"])
    draws = A3["random_benchmark"]["draws"]
    verdict = {}

    for cost_label, r in [("기본 비용", base), ("슬리피지 0.3%", replace(base, slippage=0.003))]:
        print(f"\n######## {cost_label}")
        b0_t, b0_e = run(df, setup, r)
        c1_t, c1_e = run(df, setup, replace(r, second_skip_gap=True))
        no2_t, _ = run(df, setup, replace(r, second_units=0.0))
        b0_t = b0_t.sort_values(["entry_date", "code"]).reset_index(drop=True)

        # P1 가중치 (신호일 ATR/종가), c 는 개발 구간 거래의 평균 w = 1
        sig = signal_index(b0_t, df)
        assert not np.isnan(sig).any(), "신호일을 찾지 못한 거래가 있음"
        sig = sig.astype(int)
        atr_pct = (df["atr"] / df["close"]).to_numpy()[sig]
        raw = 1.0 / atr_pct
        dev = b0_t["entry_date"].between(*PER["dev"]).to_numpy()
        c = 1.0 / np.nanmean(np.clip(raw[dev] / np.nanmean(raw[dev]), 0.33, 3.0))   # 자른 뒤 평균 1
        w = np.clip(raw / np.nanmean(raw[dev]) * c, 0.33, 3.0)
        b0_t["w_p1"] = np.nan_to_num(w, nan=1.0)

        # E2 동반 급등 종목 (신호일)
        mask = pd.Series(False, index=df.index)
        mask.loc[sig] = True
        th = add_theme_features(df, mask)
        b0_t["peers"] = th["peers"].reindex(sig).to_numpy()

        variants = {
            "B0": (b0_t, b0_e),
            "P1": (b0_t.assign(w=b0_t["w_p1"]), b0_e),
            "C1": (c1_t, c1_e),
            "E1": (b0_t[cap_positions(b0_t, MAX_POS)], b0_e),
            "E2": (b0_t[cap_theme(b0_t, THEME_MAX)], b0_e),
            "L1": (b0_t[month_loss_limit(b0_t, MONTH_LOSS)], b0_e),
        }
        res = {}
        for name, (t, e) in variants.items():
            for per in PER:
                x = in_per(t, per)
                res[(name, per)] = account(x, e, df, r)
        for per in PER:
            print(f"\n [{per}]")
            for name in variants:
                s = res[(name, per)]
                print(f"  {name:<3} 거래 {s['n']:>4}  평균 {s['mean']:+.4f}  투입연환산 {s['rodc']:+.4f}  계좌 {s['pnl']:+.4f}"
                      f"  MDD {s['mdd']:+.4f}  평균투입 {s['exp'] / 1e6:6.1f}백만  MDD/투입 {s['mdd_exp']:+.3f}"
                      f"  최악 {s['worst']:+.4f} ({s['worst_won'] / 1e4:,.0f}만)")

        if cost_label == "기본 비용":
            print("\n P1 가중치: c", round(c, 4), "분포", pd.Series(b0_t["w_p1"]).describe().round(3).to_dict(),
                  "하한 비율", round((b0_t["w_p1"] <= 0.33).mean(), 3), "상한 비율", round((b0_t["w_p1"] >= 3).mean(), 3))
            b2 = b0_e[b0_e["kind"] == "buy2"].merge(b0_t[["code", "entry_date", "entry_price", "exit_date"]], on="code")
            b2 = b2[(b2["date"] >= b2["entry_date"]) & (b2["date"] <= b2["exit_date"])]
            gap = b2["price"] < b2["entry_price"] * (1 - base.second_drop) - 1e-9
            print(f" C1 대상: 2차 체결 {len(b2)}건 중 갭 체결 {int(gap.sum())}건 ({gap.mean():.3f})")
            for name in ["E1", "E2", "L1"]:
                t = variants[name][0]
                for per in PER:
                    print(f" {name} [{per}] 건너뛴 거래 {len(in_per(b0_t, per)) - len(in_per(t, per))}")

        # 무작위 기준 (개발 구간 투입 자본 연환산 수익률)
        print("\n 무작위 기준 (개발 구간 투입 연환산, 2000회)")
        xd = in_per(b0_t, "dev").reset_index(drop=True)
        rand = {}
        wd = xd["w_p1"].to_numpy()
        rand["P1"] = [rodc(xd.assign(w=rng.permutation(wd))) for _ in range(draws)]
        # C1: 2차 체결 거래 중 같은 수를 골라 2차 없는 결과로 바꾼다
        key = ["code", "entry_date"]
        c1d, no2d = in_per(c1_t, "dev"), in_per(no2_t, "dev")
        m = xd.merge(c1d[key + ["net_ret", "invested", "hold_days"]], on=key, how="left", suffixes=("", "_c1"))
        m = m.merge(no2d[key + ["net_ret", "invested", "hold_days"]], on=key, how="left", suffixes=("", "_no2"))
        changed = (m["net_ret_c1"] - m["net_ret"]).abs().gt(1e-12) | (m["invested_c1"] - m["invested"]).abs().gt(1e-9)
        has2 = xd["second_date"].notna().to_numpy() & m["net_ret_no2"].notna().to_numpy()
        n_chg = int(changed.sum())
        print(f"  C1 로 결과가 바뀐 개발 구간 거래 {n_chg}건 (2차 체결 거래 {int(has2.sum())}건)")
        idx2 = np.flatnonzero(has2)
        out = []
        for _ in range(draws):
            s = rng.choice(idx2, n_chg, replace=False)
            y = xd.copy()
            for col in ["net_ret", "invested", "hold_days"]:
                y.loc[s, col] = m.loc[s, f"{col}_no2"].to_numpy()
            out.append(rodc(y))
        rand["C1"] = out
        for name in ["E1", "E2"]:
            k = len(in_per(variants[name][0], "dev"))
            rand[name] = [rodc(xd.iloc[np.sort(rng.choice(len(xd), k, replace=False))]) for _ in range(draws)]
        for name, dist in rand.items():
            v = res[(name, "dev")]["rodc"]
            print(f"  {name}: 실제 {v:+.4f}, B0 {res[('B0', 'dev')]['rodc']:+.4f}, 무작위 중앙 {np.median(dist):+.4f},"
                  f" 95백분위 {np.percentile(dist, 95):+.4f}, 백분위 {pct_rank(v, dist):.3f}")
            verdict[(cost_label, name, "p95")] = v > np.percentile(dist, 95)
        verdict[cost_label] = res

    print("\n== 채택 판정 (configs/a3.toml [adoption])")
    r0, r3 = verdict["기본 비용"], verdict["슬리피지 0.3%"]
    for name in ["P1", "C1", "E1", "E2"]:
        checks = {
            "개발 투입연환산 > B0": r0[(name, "dev")]["rodc"] > r0[("B0", "dev")]["rodc"],
            "개발 MDD/투입 >= B0": r0[(name, "dev")]["mdd_exp"] >= r0[("B0", "dev")]["mdd_exp"],
            "2025 투입연환산 >= B0": r0[(name, "validation")]["rodc"] >= r0[("B0", "validation")]["rodc"],
            "무작위 95백분위 초과": verdict[("기본 비용", name, "p95")],
            "슬리피지 0.3% 유지": r3[(name, "dev")]["rodc"] > r3[("B0", "dev")]["rodc"]
            and r3[(name, "validation")]["rodc"] >= r3[("B0", "validation")]["rodc"],
        }
        print(f"  {name}: " + ", ".join(f"{k} {'통과' if v else '실패'}" for k, v in checks.items())
              + f" -> {'채택' if all(checks.values()) else '채택하지 않음'}")


if __name__ == "__main__":
    main()
