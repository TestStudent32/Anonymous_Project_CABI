"""Feature-based models and baselines.

Every model function takes (X_train, X_val, X_test, y_train, preset) and returns (pred_val, pred_test),
or None if it cannot be fit. Scoring is done by the caller (src/metrics.py).

Two hyperparameter presets were used in the paper:
  "system"  - the single system-wide series (large counts): squared-error boosting, 100 trees.
  "station" - 47 noisy, low-count station series: absolute-error boosting, 300 trees, deeper trees.
"""

from functools import reduce
from operator import add

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config import RANDOM_STATE

PRESETS = {
    "system": {
        "rf": dict(n_estimators=100, max_depth=12, n_jobs=1),
        "hgb": dict(max_iter=100, max_depth=6, learning_rate=0.1),
        "boost": dict(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=1),
        "xgb_objective": None, "lgbm_objective": None,   # library defaults (squared error)
        "two_stage_classifier": "logistic",
        "gam_max_features": 20,                          # GAM on the first 20 columns when > 25 features
    },
    "station": {
        "rf": dict(n_estimators=50, max_depth=10, n_jobs=-1),
        "hgb": dict(max_iter=300, max_depth=10, learning_rate=0.03, loss="absolute_error"),
        "boost": dict(n_estimators=300, max_depth=10, learning_rate=0.03, n_jobs=-1),
        "xgb_objective": "reg:absoluteerror", "lgbm_objective": "mae",
        "two_stage_classifier": "hgb",
        "gam_max_features": None,                        # GAM on all features
    },
}


def _scaled(X_train, X_val, X_test):
    """Standardize with statistics from the training split only."""
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_val), sc.transform(X_test)


def linear_regression(X_train, X_val, X_test, y_train, preset):
    tr, va, te = _scaled(X_train, X_val, X_test)
    m = LinearRegression().fit(tr, y_train)
    return m.predict(va), m.predict(te)


def polynomial_regression(X_train, X_val, X_test, y_train, preset):
    """Degree-2 pairwise interactions (no squared terms), then linear regression. Both feature sets
    have > 30 columns, where the paper's code used interaction_only=True to keep the design tractable."""
    poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=True)
    sc = StandardScaler()
    tr = sc.fit_transform(poly.fit_transform(X_train))
    va = sc.transform(poly.transform(X_val))
    te = sc.transform(poly.transform(X_test))
    m = LinearRegression().fit(tr, y_train)
    return m.predict(va), m.predict(te)


def random_forest(X_train, X_val, X_test, y_train, preset):
    tr, va, te = _scaled(X_train, X_val, X_test)
    m = RandomForestRegressor(random_state=RANDOM_STATE, **PRESETS[preset]["rf"]).fit(tr, y_train)
    return m.predict(va), m.predict(te)


def hist_gradient_boosting(X_train, X_val, X_test, y_train, preset):
    tr, va, te = _scaled(X_train, X_val, X_test)
    m = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **PRESETS[preset]["hgb"]).fit(tr, y_train)
    return m.predict(va), m.predict(te)


def soft_two_stage(X_train, X_val, X_test, y_train, preset):
    """Zero-inflation model: P(y > 0) from a classifier times E[y | y > 0] from a regressor
    trained on non-zero hours only."""
    tr, va, te = _scaled(X_train, X_val, X_test)
    y = np.asarray(y_train)
    if PRESETS[preset]["two_stage_classifier"] == "logistic":
        clf = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    else:
        clf = HistGradientBoostingClassifier(max_iter=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE)
    clf.fit(tr, (y > 0).astype(int))
    active = y > 0
    if active.sum() < 10:
        return None
    reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **PRESETS[preset]["hgb"]).fit(tr[active], y[active])
    return reg.predict(va) * clf.predict_proba(va)[:, 1], reg.predict(te) * clf.predict_proba(te)[:, 1]


def gam(X_train, X_val, X_test, y_train, preset):
    """Additive model with one spline (8 basis functions) per feature (pygam)."""
    from pygam import LinearGAM, s
    n = X_train.shape[1]
    cap = PRESETS[preset]["gam_max_features"]
    k = min(n, cap) if (cap and n > 25) else n
    tr, va, te = (np.asarray(X)[:, :k] for X in (X_train, X_val, X_test))
    m = LinearGAM(reduce(add, [s(i, n_splines=8) for i in range(k)])).fit(tr, np.asarray(y_train))
    return m.predict(va), m.predict(te)


def xgboost(X_train, X_val, X_test, y_train, preset):
    import xgboost as xgb
    p = PRESETS[preset]
    extra = {"objective": p["xgb_objective"]} if p["xgb_objective"] else {}
    m = xgb.XGBRegressor(random_state=RANDOM_STATE, **p["boost"], **extra).fit(X_train, y_train)
    return m.predict(X_val), m.predict(X_test)


def lightgbm(X_train, X_val, X_test, y_train, preset):
    import lightgbm as lgb
    p = PRESETS[preset]
    extra = {"objective": p["lgbm_objective"]} if p["lgbm_objective"] else {}
    m = lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, **p["boost"], **extra).fit(X_train, y_train)
    return m.predict(X_val), m.predict(X_test)


def poisson_regression(X_train, X_val, X_test, y_train, preset):
    tr, va, te = _scaled(X_train, X_val, X_test)
    m = PoissonRegressor(max_iter=300).fit(tr, np.maximum(np.asarray(y_train), 0))
    return m.predict(va), m.predict(te)


# Name -> function, in the order reported. The names match the result CSVs.
FEATURE_MODELS = {
    "Linear Regression": linear_regression,
    "Polynomial Regression": polynomial_regression,
    "Random Forest": random_forest,
    "HistGradientBoosting": hist_gradient_boosting,
    "Soft Two-Stage": soft_two_stage,
    "GAM": gam,
    "XGBoost": xgboost,
    "LightGBM": lightgbm,
    "Poisson": poisson_regression,
}

NAIVE_NAME = "Naive (same hour last week)"


def naive_same_hour_last_week(X_val, X_test, y_train, target):
    """Seasonal-naive baseline: demand at the same hour one week earlier (lag_168h)."""
    col = f"lag_168h_{target.replace('_trips', '')}"
    fallback = float(np.mean(y_train))
    return X_val[col].fillna(fallback).values, X_test[col].fillna(fallback).values


def arima(y_train, y_val, y_test):
    """Univariate ARIMA(2,0,2), system level only. Fit on train -> forecast the validation year;
    refit on train+validation -> forecast the whole test year (a weak baseline by construction:
    no seasonal term, so long multi-step forecasts revert to the mean)."""
    from statsmodels.tsa.arima.model import ARIMA
    pred_val = ARIMA(y_train, order=(2, 0, 2)).fit().forecast(steps=len(y_val)).values
    full = pd.concat([y_train, y_val], ignore_index=True)
    pred_test = ARIMA(full, order=(2, 0, 2)).fit().forecast(steps=len(y_test)).values
    return pred_val, pred_test
