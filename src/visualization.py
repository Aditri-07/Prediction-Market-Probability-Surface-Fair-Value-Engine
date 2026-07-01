"""
visualization.py
==================

Plotting helpers used by the notebooks and the Streamlit dashboard.
All functions take a matplotlib Axes (or create one) and return it,
so they compose easily in subplot grids.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_probability_paths(clean_df: pd.DataFrame, market_id: str, ax=None):
    """Line chart of clean_prob over time, one line per outcome, for a
    single market_id — the core "probability surface slice" chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    sub = clean_df[clean_df["market_id"] == market_id].sort_values("timestamp")
    for outcome, g in sub.groupby("outcome_name"):
        ax.plot(g["timestamp"], g["clean_prob"], marker="o", markersize=2, label=outcome)
    ax.set_title(f"Implied probability path — {market_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Clean probability")
    ax.set_ylim(0, 1)
    ax.legend(title="Outcome")
    ax.grid(alpha=0.3)
    return ax


def plot_probability_surface_3d(clean_df: pd.DataFrame, outcome_name: str, ax=None):
    """3D surface: x = days_to_resolution, y = market (event date), z = probability,
    for a single outcome bucket across all markets. Requires `days_to_resolution`
    column (from feature_engineering.add_time_to_resolution) or expiration_date
    to compute it locally.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = clean_df[clean_df["outcome_name"] == outcome_name].copy()
    if "days_to_resolution" not in sub.columns:
        raise ValueError("clean_df must include days_to_resolution (see feature_engineering.py)")

    markets = sorted(sub["market_id"].unique())
    market_idx = {m: i for i, m in enumerate(markets)}
    sub["market_idx"] = sub["market_id"].map(market_idx)

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(sub["days_to_resolution"], sub["market_idx"], sub["clean_prob"],
               c=sub["clean_prob"], cmap="viridis", s=8)
    ax.set_xlabel("Days to resolution")
    ax.set_ylabel("Market index (chronological)")
    ax.set_zlabel("Probability")
    ax.set_title(f'Probability surface — outcome "{outcome_name}"')
    ax.invert_xaxis()
    return ax


def plot_mispricing_table(scored_df: pd.DataFrame, top_n: int = 15, ax=None):
    """Horizontal bar chart of the largest |edge| = fair_prob - clean_prob
    observations, most recent snapshot only.
    """
    latest = scored_df.sort_values("timestamp").groupby(
        ["market_id", "outcome_name"]).tail(1).copy()
    latest["edge"] = latest["fair_prob"] - latest["clean_prob"]
    latest["label"] = latest["market_id"].str.replace("FED-", "") + " | " + latest["outcome_name"]
    top = latest.reindex(latest["edge"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.sort_values("edge")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    colors = np.where(top["edge"] > 0, "#2a9d8f", "#e76f51")
    ax.barh(top["label"], top["edge"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Edge (fair value − market price)")
    ax.set_title("Largest current mispricings")
    ax.grid(alpha=0.3, axis="x")
    return ax


def plot_backtest_pnl(trades: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    if trades.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return ax
    t = trades.sort_values("timestamp").copy()
    t["cum_pnl"] = t["pnl"].cumsum()
    ax.plot(t["timestamp"], t["cum_pnl"], color="#264653")
    ax.fill_between(t["timestamp"], t["cum_pnl"], alpha=0.15, color="#264653")
    ax.set_title("Cumulative backtest P&L (probability points, after fees)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.grid(alpha=0.3)
    return ax


def plot_calibration(scored_df: pd.DataFrame, n_bins: int = 10, ax=None):
    """Reliability diagram: bucket fair_prob predictions and compare to
    realized frequency (next_prob) — a visual complement to Brier/log loss.
    """
    df = scored_df.dropna(subset=["fair_prob"]).copy()
    if "next_prob" not in df.columns:
        df["next_prob"] = df.groupby(["market_id", "outcome_name"])["clean_prob"].shift(-1)
    df = df.dropna(subset=["next_prob"])

    df["bin"] = pd.cut(df["fair_prob"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    calib = df.groupby("bin", observed=True).agg(
        predicted=("fair_prob", "mean"), realized=("next_prob", "mean"), n=("fair_prob", "size")
    ).dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(calib["predicted"], calib["realized"], s=calib["n"] * 3, alpha=0.7, color="#e76f51")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean realized probability")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
