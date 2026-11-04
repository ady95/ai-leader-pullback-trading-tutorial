"""09-1·09-2 Walk-forward 학습, 기준 모델 비교, 확률 보정, SHAP.

예측 구간(2022~2025)의 예측값을 data/results/a1_scores.parquet 에 저장한다 (09-3 매매 평가용).
2026(최종 홀드아웃)은 예측하지 않는다.

사용:
    uv run python scripts/train_09.py
"""
from __future__ import annotations

import sys
import tomllib

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pullback.dataset import ALL_FEATURES
from pullback.model import fit, train_set
from pullback.paths import DATA

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)


def scores(y: np.ndarray, p: np.ndarray) -> str:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return f"AUC {roc_auc_score(y, p):.3f}  Brier {brier_score_loss(y, p):.4f}  logloss {log_loss(y, p):.4f}"


def main() -> None:
    spec = tomllib.loads(open("configs/a1.toml", encoding="utf-8").read())
    ds = pd.read_parquet(DATA / "dataset_a1.parquet")
    cand = pd.read_parquet(DATA / "candidates_a1.parquet")
    ds_year = pd.to_datetime(ds["date"]).dt.year
    cand_year = pd.to_datetime(cand["date"]).dt.year
    oos, cand_scores, last_fit = [], [], None
    print("== 09-1 폴드별 예측 성능 (예측 구간 표본, 라벨 = +20% 먼저)")
    for last_train, pred in spec["split"]["folds"]:
        train = train_set(ds, last_train, pred)
        test = ds[(ds_year == int(pred)) & ds["y"].notna()]
        y = (test["y"] == 1).astype(int).to_numpy()
        base = np.full(len(test), (train["y"] == 1).mean())
        m_lgb, m_log = fit(train, "lgbm"), fit(train, "logit")
        p_lgb, p_log = m_lgb.predict(test), m_log.predict(test)
        print(f"  [{pred}] 학습 {len(train):,} (양성 {train['y'].eq(1).mean():.3f}) 예측 {len(test):,} (양성 {y.mean():.3f})")
        print(f"     기준율       {scores(y, base)}")
        print(f"     로지스틱     {scores(y, p_log)}")
        print(f"     LightGBM     {scores(y, p_lgb)}  (보정 전 AUC {roc_auc_score(y, m_lgb.predict_raw(test)):.3f})")
        oos.append(test[["code", "date", "wave_id", "y"]].assign(p=p_lgb, p_logit=p_log, base=base, year=int(pred)))
        c = cand[cand_year == int(pred)]
        cand_scores.append(c[["code", "date", "touched"]].assign(p=m_lgb.predict(c), p_logit=m_log.predict(c), year=int(pred)))
        last_fit, last_train_df = m_lgb, train

    o = pd.concat(oos)
    pooled = o[o["year"] <= 2024]
    y = (pooled["y"] == 1).astype(int).to_numpy()
    print("\n  [2022~2024 합산]")
    for name, col in [("기준율", "base"), ("로지스틱", "p_logit"), ("LightGBM", "p")]:
        print(f"     {name:<10} {scores(y, pooled[col].to_numpy())}")
    # 파동 단위로 묶어도 같은 결론인지: 파동별 평균 예측 vs 파동별 양성 비율
    w = pooled.groupby("wave_id").agg(p=("p", "mean"), y=("y", lambda s: (s == 1).mean()))
    print(f"     파동 단위 순위 상관 (예측 평균 vs 양성 비율) {w['p'].corr(w['y'], method='spearman'):.3f}, 파동 {len(w)}")

    print("\n== 09-2 확률 보정 (2022~2024 합산, 예측 확률 구간별)")
    bins = [0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 1.0]
    tab = pooled.assign(bin=pd.cut(pooled["p"], bins), pos=(pooled["y"] == 1)).groupby("bin", observed=False).agg(
        n=("p", "size"), mean_p=("p", "mean"), actual=("pos", "mean"))
    print(tab.round(3).to_string())
    print("  임계값 0.30 이상 비율", round(float((pooled["p"] >= spec["decision"]["threshold"]).mean()), 3),
          "그 표본의 실제 양성 비율", round(float((pooled.loc[pooled["p"] >= 0.30, "y"] == 1).mean()), 3))
    v = o[o["year"] == 2025]
    print("  2025: 임계값 이상 비율", round(float((v["p"] >= 0.30).mean()), 3),
          "실제 양성", round(float((v.loc[v["p"] >= 0.30, "y"] == 1).mean()), 3), "전체 양성", round(float((v["y"] == 1).mean()), 3))

    print("\n== 09-2 SHAP (폴드 4 모델: 학습 2021~2024)")
    n_cal = int(len(last_train_df) * 0.2)
    X = last_train_df.iloc[:-n_cal][ALL_FEATURES]
    sv = shap.TreeExplainer(last_fit.model).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"mean_abs": np.abs(sv).mean(axis=0),
                        "direction": [pd.Series(X[c]).corr(pd.Series(sv[:, i]), method="spearman") for i, c in enumerate(ALL_FEATURES)]},
                       index=ALL_FEATURES).sort_values("mean_abs", ascending=False)
    print(imp.round(4).to_string())

    out = DATA / "results"
    out.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out / "a1_oos.parquet", index=False)
    pd.concat(cand_scores).to_parquet(out / "a1_scores.parquet", index=False)
    print("\n저장:", out / "a1_oos.parquet", out / "a1_scores.parquet")


if __name__ == "__main__":
    main()
