"""
feature_engineering.py
=======================

Builds the feature matrix used by the fair-value model:

* time_to_resolution (days)
* macro signals joined on date: fed_funds_futures_implied, y2_yield,
  y10_yield, cpi_surprise, jobs_surprise, vix
* momentum features: 1-day and 5-day change in clean_prob
* lagged market clean_prob (prior day's probability -> today's prior)

Output is one row per (market_id, outcome_name, timestamp), ready to
feed into `fair_value_model.py`.
"""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "prior_prob",
    "days_to_resolution",
    "momentum_1d",
    "momentum_5d",
    "fed_funds_futures_implied",
    "y2_yield",
    "y10_yield",
    "cpi_surprise",
    "jobs_surprise",
    "vix",
    "trust_weight",
]


def add_time_to_resolution(df: pd.DataFrame, markets_df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[c for c in ("expiration_date", "days_to_resolution") if c in df.columns])
    out = out.merge(markets_df[["market_id", "expiration_date"]].drop_duplicates(),
                     on="market_id", how="left")
    out["expiration_date"] = pd.to_datetime(out["expiration_date"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["days_to_resolution"] = (out["expiration_date"] - out["timestamp"]).dt.days.clip(lower=0)
    return out


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["market_id", "outcome_name", "timestamp"]).copy()
    grp = df.groupby(["market_id", "outcome_name"])["clean_prob"]
    df["prior_prob"] = grp.shift(1)
    df["momentum_1d"] = df["clean_prob"] - df["prior_prob"]
    df["momentum_5d"] = df["clean_prob"] - grp.shift(5)
    return df


def join_macro_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    macro = macro_df.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    out = df.merge(
        macro[["date", "market_id", "fed_funds_futures_implied", "y2_yield",
               "y10_yield", "cpi_surprise", "jobs_surprise", "vix"]],
        on=["date", "market_id"], how="left",
    )
    return out.drop(columns=["date"])


def build_feature_table(clean_price_df: pd.DataFrame, markets_df: pd.DataFrame,
                         macro_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature pipeline. `clean_price_df` should already have
    `clean_prob` and `trust_weight` from probability_cleaning.py.
    """
    df = add_time_to_resolution(clean_price_df, markets_df)
    df = add_momentum_features(df)
    df = join_macro_features(df, macro_df)

    # target: the realized clean_prob (what we're trying to fair-value)
    df["target_prob"] = df["clean_prob"]

    # drop rows without enough history for momentum features
    df = df.dropna(subset=["prior_prob"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    from data_loader import load_data
    from probability_cleaning import build_clean_probability_table

    markets, prices, macro = load_data(use_live=False)
    clean = build_clean_probability_table(prices)
    features = build_feature_table(clean, markets, macro)
    print(features[["market_id", "outcome_name", "timestamp"] + FEATURE_COLUMNS + ["target_prob"]].head(10))
    print(f"\nFeature table shape: {features.shape}")
