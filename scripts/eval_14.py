"""14장 시장 국면 규칙(R1 급락 직후, R2 깊은 하락장), 스트레스 시험, Walk-forward 선택, 최종 홀드아웃.

사용:
    uv run python scripts/eval_14.py              # 14-1, 14-2 (2020~2025 만, 2026 성과는 계산하지 않음)
    uv run python scripts/eval_14.py --wf         # Walk-forward 선택만
    uv run python scripts/eval_14.py --holdout    # 14-3 최종 홀드아웃 (configs/final_14.toml, 한 번만)
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_12 import CAPITAL, UNIT, account, money, pct_rank, rodc  # noqa: E402

from pullback.backtest import Rules, run  # noqa: E402
from pullback.definitions import build, flags, load_daily  # noqa: E402
from pullback.metrics import bootstrap_mean_ci  # noqa: E402
from pullback.paths import DATA, UNIVERSE  # noqa: E402
from pullback.spec import load, to_params  # noqa: E402
from pullback.theme import INDEX  # noqa: E402

SPEC = tomllib.loads(open("configs/final_14.toml", encoding="utf-8").read())
PER = {"dev": ("2021-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31")}
HOLD = tuple(SPEC["holdout"]["window"])
RESULT = DATA / "results" / "regime_14.json"


def market_flags(df: pd.DataFrame) -> pd.DataFrame:
    """신호일 기준 지수 상태: 최근 5거래일 최저 일간 수익률, 20거래일 수익률, 그날 일간 수익률."""
    idx = pd.read_parquet(INDEX).sort_values(["market", "date"])
    g = idx.groupby("market")["close"]
    idx["mkt_ret1"] = g.pct_change()
    idx["mkt_min5"] = idx.groupby("market")["mkt_ret1"].transform(lambda s: s.rolling(5, min_periods=1).min())
    idx["mkt_ret20"] = g.pct_change(20)
    mk = pd.read_parquet(UNIVERSE)[["code", "market"]]
    out = df[["code", "date"]].merge(mk, on="code", how="left").merge(
        idx[["market", "date", "mkt_ret1", "mkt_min5", "mkt_ret20"]], on=["market", "date"], how="left")
    out.index = df.index
    return out


def in_win(t: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return t[t["entry_date"].between(a, b)]


def line(name: str, s: dict) -> str:
    return (f"  {name:<4} 거래 {s['n']:>4}  평균 {s['mean']:+.4f}  투입연환산 {s['rodc']:+.4f}  계좌 {s['pnl']:+.4f}"
            f"  MDD {s['mdd']:+.4f}  평균투입 {s['exp'] / 1e6:6.1f}백만  MDD/투입 {s['mdd_exp']:+.3f}"
            f"  최악 {s['worst']:+.4f} ({s['worst_won'] / 1e4:,.0f}만)")


def crash_report(t: pd.DataFrame, df: pd.DataFrame, mf: pd.DataFrame, a: str, b: str) -> None:
    """지수 일간 -5% 이하인 날 보유 중이던 거래(그날 시가 이전에 들고 있던 거래)."""
    days = mf.loc[(mf["mkt_ret1"] <= -0.05) & mf["date"].between(a, b), ["market", "date", "mkt_ret1"]].drop_duplicates()
    days = days.groupby("date")["mkt_ret1"].min()
    px = df.set_index(["code", "date"])[["open", "close", "prev_close"]]
    for d, r in days.items():
        held = t[(t["entry_date"] < d) & (t["exit_date"] >= d)]
        if held.empty:
            print(f"  {d.date()} 지수 최저 {r:+.3f}  보유 0")
            continue
        day = []
        for c, x in zip(held["code"], held["exit_date"]):
            row = px.loc[(c, d)] if (c, d) in px.index else None
            if row is None or not row["prev_close"] > 0:
                continue
            day.append((row["open"] if x == d else row["close"]) / row["prev_close"] - 1)
        print(f"  {d.date()} 지수 최저 {r:+.3f}  보유 {len(held)}건  그날 평균 {np.mean(day):+.4f}"
              f"  그날 최악 {np.min(day):+.4f}  최종 평균 {held['net_ret'].mean():+.4f}"
              f"  그날 청산 {int((held['exit_date'] == d).sum())}건")


def stress(df, setup, base, mf, a, b, label):
    print(f"\n== 14-2 스트레스 [{label}]")
    for slip in SPEC["stress"]["slippage_grid"]:
        r = replace(base, slippage=slip)
        t, e = run(df, setup, r)
        s = account(in_win(t, a, b), e, df, r)
        print(f"  슬리피지 {slip:.3f}: 거래 {s['n']}  평균 {s['mean']:+.4f}  투입연환산 {s['rodc']:+.4f}  계좌 {s['pnl']:+.4f}")
    t, e = run(df, setup, replace(base, strict_touch=True))
    s = account(in_win(t, a, b), e, df, base)
    print(f"  엄격 체결: 거래 {s['n']}  평균 {s['mean']:+.4f}  투입연환산 {s['rodc']:+.4f}  계좌 {s['pnl']:+.4f}")
    t, e = run(df, setup, base)
    x = in_win(t, a, b)
    q = x["net_ret"].quantile(1 - SPEC["stress"]["drop_best"])
    y = x[x["net_ret"] < q]
    print(f"  상위 5% 제외: {len(y)}건  평균 {y['net_ret'].mean():+.4f}  투입연환산 {rodc(y):+.4f}"
          f"  (제외한 {len(x) - len(y)}건 평균 {x.loc[x['net_ret'] >= q, 'net_ret'].mean():+.4f})")
    print("  급락일 (지수 일간 -5% 이하)")
    crash_report(t, df, mf, a, b)
    return t, e


def walk_forward(df, setup, base) -> None:
    """해마다 지난 해까지 누적 투입 자본 연환산 1위 변형을 골라 다음 해에 쓴다 (2022~2025)."""
    cands = {"B0": base, "second_none": replace(base, second_units=0.0), "second_1.5x": replace(base, second_units=1.5),
             "second_-3": replace(base, second_drop=0.03), "second_-5": replace(base, second_drop=0.05),
             "second_ma60": replace(base, second_require_ma60=True), "no_stop": replace(base, use_stop=False),
             "time_20": replace(base, max_hold_days=20), "time_120": replace(base, max_hold_days=120),
             "tp_fixed_25": replace(base, tp_mode="fixed", tp_fixed=0.25), "X2": replace(base, take_profit_1=0.10),
             "X3": replace(base, stop_mode="atr"), "X4": replace(base, trail_on=0.10), "T2": replace(base, noprog_day=10)}
    tr = {k: run(df, setup, v)[0] for k, v in cands.items()}
    for k in tr:
        tr[k] = tr[k].assign(year=tr[k]["entry_date"].dt.year)
    print("\n== 14-2 Walk-forward 선택 (지난 해까지 누적 투입 자본 연환산 1위를 다음 해에)")
    chain, chain_b0 = [], []
    for y in [2022, 2023, 2024, 2025]:
        past = {k: rodc(t[t["year"].between(2021, y - 1)]) for k, t in tr.items()}
        pick = max(past, key=past.get)
        rank_now = pd.Series({k: rodc(t[t["year"] == y]) for k, t in tr.items()}).rank(ascending=False)
        got, b0 = tr[pick][tr[pick]["year"] == y], tr["B0"][tr["B0"]["year"] == y]
        chain.append(got)
        chain_b0.append(b0)
        print(f"  {y}: 고른 변형 {pick} (과거 {past[pick]:+.4f})  그해 {rodc(got):+.4f} (순위 {int(rank_now[pick])}/{len(tr)})"
              f"  B0 {rodc(b0):+.4f} (순위 {int(rank_now['B0'])})  그해 1위 {rank_now.idxmin()}")
    c, cb = pd.concat(chain), pd.concat(chain_b0)
    print(f"  2022~2025 합계: 선택 {rodc(c):+.4f} (거래당 {c['net_ret'].mean():+.4f}, {len(c)}건, 최악 {c['net_ret'].min():+.4f},"
          f" 손익합 {(c['net_ret'] * money(c)).sum() / CAPITAL:+.4f}, 평균 보유 {c['hold_days'].mean():.1f}일)"
          f"  B0 {rodc(cb):+.4f} (거래당 {cb['net_ret'].mean():+.4f}, {len(cb)}건, 최악 {cb['net_ret'].min():+.4f},"
          f" 손익합 {(cb['net_ret'] * money(cb)).sum() / CAPITAL:+.4f}, 평균 보유 {cb['hold_days'].mean():.1f}일)")
    ns = tr["no_stop"]
    print(f"  no_stop 2022~2025: -30% 이하 거래 {int((ns['year'].between(2022, 2025) & (ns['net_ret'] <= -0.30)).sum())}건,"
          f" 연도별 최악 {ns[ns['year'].between(2022, 2025)].groupby('year')['net_ret'].min().round(4).to_dict()}")


def regime(df, setup, base, mf) -> list[str]:
    rng = np.random.default_rng(0)
    masks = {"R1": mf["mkt_min5"].gt(-0.03), "R2": mf["mkt_ret20"].gt(-0.10)}
    print("== 14-1 신호일 플래그 (2021~2025 후보일 중 주문 금지 비율)")
    cand = setup & df["date"].between("2021-01-01", "2025-12-31")
    for k, m in masks.items():
        print(f"  {k}: 후보일 {int(cand.sum())} 중 금지 {int((cand & ~m.fillna(True)).sum())}")
    verdict = {}
    for cost_label, r in [("기본 비용", base), ("슬리피지 0.3%", replace(base, slippage=0.003))]:
        runs = {"B0": run(df, setup, r)}
        for k, m in masks.items():
            runs[k] = run(df, setup & m.fillna(True), r)
        res = {(k, p): account(in_win(t, *PER[p]), e, df, r) for k, (t, e) in runs.items() for p in PER}
        print(f"\n######## {cost_label}")
        for p in PER:
            print(f" [{p}]")
            for k in runs:
                print(line(k, res[(k, p)]))
        verdict[cost_label] = res
        if cost_label == "기본 비용":
            b0d = in_win(runs["B0"][0], *PER["dev"]).reset_index(drop=True)
            key = ["code", "entry_date"]
            for k in masks:
                rd = in_win(runs[k][0], *PER["dev"])
                kept = b0d.merge(rd[key], on=key, how="left", indicator=True)["_merge"].eq("both").to_numpy()
                skipped = b0d[~kept]
                print(f"  {k} 개발 구간 건너뛴 B0 거래 {len(skipped)}건 평균 {skipped['net_ret'].mean():+.4f}"
                      f"  연도별 {skipped.groupby(skipped['entry_date'].dt.year)['net_ret'].agg(['size', 'mean']).round(4).to_dict('index')}")
                n = len(rd)
                dist = [rodc(b0d.iloc[np.sort(rng.choice(len(b0d), n, replace=False))]) for _ in range(2000)]
                v = res[(k, "dev")]["rodc"]
                print(f"  {k} 무작위: 실제 {v:+.4f}, 중앙 {np.median(dist):+.4f}, 95백분위 {np.percentile(dist, 95):+.4f},"
                      f" 백분위 {pct_rank(v, dist):.3f}")
                verdict[("p95", k)] = v > np.percentile(dist, 95)
                verdict[("skipped", k)] = len(skipped)
    print("\n== 14-1 채택 판정")
    r0, r3 = verdict["기본 비용"], verdict["슬리피지 0.3%"]
    adopted = []
    for k in masks:
        if verdict[("skipped", k)] < SPEC["regime"]["min_skipped"]:
            print(f"  {k}: 건너뛴 거래 {verdict[('skipped', k)]}건 < {SPEC['regime']['min_skipped']} -> 판정 보류")
            continue
        checks = {
            "개발 투입연환산 > B0": r0[(k, "dev")]["rodc"] > r0[("B0", "dev")]["rodc"],
            "개발 MDD/투입 >= B0": r0[(k, "dev")]["mdd_exp"] >= r0[("B0", "dev")]["mdd_exp"],
            "2025 투입연환산 >= B0": r0[(k, "validation")]["rodc"] >= r0[("B0", "validation")]["rodc"],
            "무작위 95백분위 초과": verdict[("p95", k)],
            "슬리피지 0.3% 유지": r3[(k, "dev")]["rodc"] > r3[("B0", "dev")]["rodc"]
            and r3[(k, "validation")]["rodc"] >= r3[("B0", "validation")]["rodc"],
        }
        ok = all(checks.values())
        print(f"  {k}: " + ", ".join(f"{c} {'통과' if v else '실패'}" for c, v in checks.items())
              + f" -> {'채택' if ok else '채택하지 않음'}")
        if ok:
            adopted.append(k)
    print("  채택:", adopted or "없음")
    return adopted


def holdout(df, f, base, mf) -> None:
    a, b = HOLD
    adopted = json.loads(RESULT.read_text(encoding="utf-8"))["adopted"]
    setup = f["setup"] & (df["date"] <= "2026-11-02")
    masks = {"B0": setup}
    for k in adopted:
        masks[f"B0+{k}"] = setup & {"R1": mf["mkt_min5"].gt(-0.03), "R2": mf["mkt_ret20"].gt(-0.10)}[k].fillna(True)
    print(f"== 14-3 최종 홀드아웃 {a} ~ {b}  대상 {list(masks)}")
    for name, m in masks.items():
        t, e = run(df, m, base)
        x = in_win(t, a, b)
        s = account(x, e, df, base)
        lo, hi = bootstrap_mean_ci(x["net_ret"], x["entry_date"].dt.to_period("M"))
        mw = float((x["net_ret"] * money(x)).sum() / money(x).sum())
        t3, _ = run(df, m, replace(base, slippage=0.003))
        r3 = rodc(in_win(t3, a, b))
        print(line(name, s))
        print(f"       95% 구간 [{lo:+.4f}, {hi:+.4f}]  승률 {(x['net_ret'] > 0).mean():.3f}  금액가중 {mw:+.4f}"
              f"  슬리피지 0.3% 투입연환산 {r3:+.4f}  미청산 {int((x['exit_reason'] == 'data_end').sum())}건")
        print(f"       청산 사유 {x['exit_reason'].value_counts().to_dict()}")
        print(f"       월별 {x.groupby(x['entry_date'].dt.month)['net_ret'].agg(['size', 'mean']).round(4).to_dict('index')}")
        if name == "B0":
            v = SPEC["holdout"]
            if s["n"] < v["min_trades"]:
                verdict = "판정 보류 (거래 수 부족)"
            elif s["mean"] > 0 and lo > 0 and r3 > 0:
                verdict = "우위 확인"
            elif s["mean"] > 0:
                verdict = "시기 의존"
            else:
                verdict = "H1 기각 확정"
            print(f"  >> B0 홀드아웃 판정: {verdict}")
            ent, ext = x["entry_date"].to_numpy(), x["exit_date"].to_numpy()
            allt = t
            held = [int(((allt["entry_date"].to_numpy() < d) & (allt["exit_date"].to_numpy() >= d)).sum()) for d in ent]
            busy = np.array(held) >= 20
            print(f"  [서술] 진입 시 보유 >= 20: {int(busy.sum())}건 평균 {x['net_ret'][busy].mean():+.4f}"
                  f" | 나머지 {int((~busy).sum())}건 평균 {x['net_ret'][~busy].mean():+.4f}")
            stress(df, m, base, mf, a, b, "홀드아웃")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--wf", action="store_true", help="Walk-forward 선택만")
    args = ap.parse_args()
    spec, _ = load("configs/b0.toml")
    p = to_params(spec)
    df = build(load_daily(spec["periods"]["warmup_start"]), p)
    f = flags(df, p)
    base = Rules.from_spec(spec)
    mf = market_flags(df)
    if args.holdout:
        return holdout(df, f, base, mf)
    setup = f["setup"] & (df["date"] <= "2025-12-31")
    if args.wf:
        return walk_forward(df, setup, base)
    adopted = regime(df, setup, base, mf)
    RESULT.write_text(json.dumps({"adopted": adopted}), encoding="utf-8")
    for label, (a, b) in PER.items():
        stress(df, setup, base, mf, a, b, label)
    walk_forward(df, setup, base)


if __name__ == "__main__":
    main()
