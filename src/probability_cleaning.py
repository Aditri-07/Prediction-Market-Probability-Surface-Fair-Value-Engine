"""
probability_cleaning.py
========================

Converts raw bid/ask prices into clean, de-vigged, normalized
implied-probability curves.

Key ideas
---------
* Mid-price = (best bid + best ask) / 2  -> naive implied probability
* Multi-outcome markets (e.g. "0/1/2/3+ cuts") don't sum exactly to 1.0
  because of the bid/ask spread and market maker vig. We normalize
  per timestamp so probabilities across outcomes always sum to 1.
* We also compute a liquidity-aware "trust" weight per outcome, since
  wide-spread / low-volume outcomes carry noisier prices.
"""

from __future__ import annotations

import pandas as pd


def add_mid_price(price_df: pd.DataFrame) -> pd.DataFrame:
    df = price_df.copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]
    return df


def normalize_probabilities(price_df: pd.DataFrame,
                             group_cols: tuple = ("market_id", "timestamp")) -> pd.DataFrame:
    """Remove overround/vig by rescaling outcome mids to sum to 1 within
    each (market_id, timestamp) group.

    Adds a `clean_prob` column.
    """
    df = add_mid_price(price_df)
    group_sum = df.groupby(list(group_cols))["mid"].transform("sum")
    df["overround"] = group_sum - 1.0
    df["clean_prob"] = df["mid"] / group_sum
    return df


def liquidity_weight(price_df: pd.DataFrame, spread_floor: float = 0.005) -> pd.DataFrame:
    """Adds a `trust_weight` column in (0, 1]: higher volume and tighter
    spreads earn more trust. Useful for weighting the fair-value model
    and for filtering low-conviction signals in the backtester.
    """
    df = price_df.copy()
    if "spread" not in df.columns:
        df = add_mid_price(df)

    vol_norm = (df["volume"] - df["volume"].min()) / (df["volume"].max() - df["volume"].min() + 1e-9)
    spread_penalty = 1.0 / (df["spread"].clip(lower=spread_floor) * 100)
    spread_penalty_norm = (spread_penalty - spread_penalty.min()) / (
        spread_penalty.max() - spread_penalty.min() + 1e-9
    )
    df["trust_weight"] = (0.5 * vol_norm + 0.5 * spread_penalty_norm).clip(0.01, 1.0)
    return df


def build_clean_probability_table(price_df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: mid price -> normalize -> liquidity weight."""
    df = normalize_probabilities(price_df)
    df = liquidity_weight(df)
    return df.sort_values(["market_id", "timestamp", "outcome_name"]).reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import load_data

    _, prices, _ = load_data(use_live=False)
    clean = build_clean_probability_table(prices)
    print(clean.head(10))
    print("\nOverround stats:")
    print(clean.groupby("market_id")["overround"].mean().describe())
