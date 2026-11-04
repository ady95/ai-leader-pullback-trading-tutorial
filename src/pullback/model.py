"""09장 A1 모델: LightGBM (고정 하이퍼파라미터) + 시간순 isotonic 보정, 비교용 기준 모델.

모든 설정은 configs/a1.toml 에 등록한 값이며 튜닝하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pullback.dataset import ALL_FEATURES

LGB_PARAMS = dict(num_leaves=15, learning_rate=0.03, n_estimators=300, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
CALIB_FRACTION = 0.2
TRAIN_START = "2021-01-01"


def train_set(ds: pd.DataFrame, last_train: str, pred: str) -> pd.DataFrame:
    """학습 구간 표본에서 라벨 겹침·파동 공유 표본을 지운다 (08-3)."""
    d = pd.to_datetime(ds["date"])
    train = ds[(d >= TRAIN_START) & (d <= f"{last_train}-12-31") & ds["y"].notna()]
    test = ds[d.dt.year == int(pred)]
    start = pd.Timestamp(f"{pred}-01-01")
    drop = (pd.to_datetime(train["y_end"]) >= start) | train["wave_id"].isin(set(test["wave_id"]))
    return train[~drop].sort_values("date")


@dataclass
class Fitted:
    model: object
    iso: IsotonicRegression
    features: list = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X[self.features or ALL_FEATURES])[:, 1]
        return self.iso.predict(raw)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.features or ALL_FEATURES])[:, 1]


def fit(train: pd.DataFrame, kind: str = "lgbm", features: list | None = None) -> Fitted:
    """시간순 앞 80% 로 학습, 뒤 20% 로 isotonic 보정."""
    features = features or ALL_FEATURES
    n_cal = int(len(train) * CALIB_FRACTION)
    fit_part, cal_part = train.iloc[:-n_cal], train.iloc[-n_cal:]
    y_fit = (fit_part["y"] == 1).astype(int)
    if kind == "lgbm":
        model = lgb.LGBMClassifier(**LGB_PARAMS)
    elif kind == "logit":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else:
        raise ValueError(kind)
    model.fit(fit_part[features], y_fit)
    raw = model.predict_proba(cal_part[features])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, (cal_part["y"] == 1).astype(int))
    return Fitted(model, iso, features)
