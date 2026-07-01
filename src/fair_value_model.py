"""
fair_value_model.py
=====================

Two fair-value estimators, sharing a common interface:

1. `BayesianUpdateModel` — a transparent, auditable model.
   Prior = yesterday's clean market probability.
   Evidence = standardized macro surprises (CPI surprise, jobs surprise,
   yield moves, futures-implied shift).
   Posterior = prior odds updated by a log-odds (logit) shift estimated
   from a small linear regression of macro evidence on historical
   log-odds changes. This is a simple, well-known way to do Bayesian-style
   updating on probabilities: work in logit space (where updates are
   additive), then convert back to probability space.

2. `GradientBoostFairValueModel` — a scikit-learn GradientBoostingRegressor
   trained directly on the feature table to predict `target_prob`
   (i.e., tomorrow's / near-term clean market probability). Useful as a
   flexible benchmark against the Bayesian model.

Both expose `.fit(train_df)` and `.predict(df) -> pd.Series` (fair_prob),
so `backtester.py` can treat them interchangeably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLUMNS

EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class BayesianUpdateModel:
    """Logit-space Bayesian-style updating.

    fair_prob = sigmoid( logit(prior_prob) + beta . evidence )

    `beta` is fit via OLS regressing the realized logit *change*
    (logit(target_prob) - logit(prior_prob)) on standardized macro
    evidence features. This keeps the model interpretable: each
    coefficient tells you how many logits of probability shift a
    1-std-dev macro surprise is worth.
    """

    EVIDENCE_COLS = ["momentum_1d", "momentum_5d", "fed_funds_futures_implied",
                      "cpi_surprise", "jobs_surprise", "y2_yield", "y10_yield", "vix"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.reg = LinearRegression()
        self.fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "BayesianUpdateModel":
        df = train_df.dropna(subset=self.EVIDENCE_COLS + ["prior_prob", "target_prob"]).copy()
        X = self.scaler.fit_transform(df[self.EVIDENCE_COLS])
        y = _logit(df["target_prob"].values) - _logit(df["prior_prob"].values)
        self.reg.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call .fit() before .predict().")
        work = df.copy()
        for c in self.EVIDENCE_COLS:
            work[c] = work[c].fillna(work[c].median())
        X = self.scaler.transform(work[self.EVIDENCE_COLS])
        logit_shift = self.reg.predict(X)
        prior = work["prior_prob"].fillna(work["prior_prob"].median()).values
        fair_prob = _sigmoid(_logit(prior) + logit_shift)
        return pd.Series(fair_prob, index=df.index, name="fair_prob")

    def coefficients(self) -> pd.Series:
        return pd.Series(self.reg.coef_, index=self.EVIDENCE_COLS).sort_values(key=abs, ascending=False)


class GradientBoostFairValueModel:
    """Flexible non-linear benchmark model."""

    def __init__(self, **gbr_kwargs):
        defaults = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        defaults.update(gbr_kwargs)
        self.model = GradientBoostingRegressor(**defaults)
        self.fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "GradientBoostFairValueModel":
        df = train_df.dropna(subset=FEATURE_COLUMNS + ["target_prob"]).copy()
        self.model.fit(df[FEATURE_COLUMNS], df["target_prob"])
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if not self.fitted_:
            raise RuntimeError("Call .fit() before .predict().")
        work = df.copy()
        for c in FEATURE_COLUMNS:
            work[c] = work[c].fillna(work[c].median())
        preds = self.model.predict(work[FEATURE_COLUMNS])
        preds = np.clip(preds, 0.001, 0.999)
        return pd.Series(preds, index=df.index, name="fair_prob")

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)


def time_based_train_test_split(df: pd.DataFrame, test_frac: float = 0.25):
    """Split by timestamp (not randomly) — never train on the future."""
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].quantile(1 - test_frac)
    train = df[df["timestamp"] <= cutoff].copy()
    test = df[df["timestamp"] > cutoff].copy()
    return train, test


if __name__ == "__main__":
    from data_loader import load_data
    from probability_cleaning import build_clean_probability_table
    from feature_engineering import build_feature_table

    markets, prices, macro = load_data(use_live=False)
    clean = build_clean_probability_table(prices)
    features = build_feature_table(clean, markets, macro)

    train, test = time_based_train_test_split(features)

    bayes = BayesianUpdateModel().fit(train)
    test = test.copy()
    test["fair_prob_bayes"] = bayes.predict(test)
    print("Bayesian model coefficients (logit-space, per 1 std-dev evidence):")
    print(bayes.coefficients())

    gbr = GradientBoostFairValueModel().fit(train)
    test["fair_prob_gbr"] = gbr.predict(test)
    print("\nGBR feature importance:")
    print(gbr.feature_importance())

    print("\nSample predictions vs market:")
    print(test[["market_id", "outcome_name", "timestamp", "clean_prob",
                "fair_prob_bayes", "fair_prob_gbr"]].head(10))
