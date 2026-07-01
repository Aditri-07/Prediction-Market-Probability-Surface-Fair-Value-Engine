"""
dashboard/streamlit_app.py
============================

Interactive dashboard for the Prediction Market Probability Surface &
Mispricing Engine. Run with:

    streamlit run dashboard/streamlit_app.py

Uses the same synthetic-data fallback as the notebooks, so it runs with
zero setup. Toggle "Use live Polymarket data" in the sidebar once you
have network access and want to point it at real markets.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_loader import load_data
from probability_cleaning import build_clean_probability_table
from feature_engineering import build_feature_table
from fair_value_model import BayesianUpdateModel, GradientBoostFairValueModel, time_based_train_test_split
from backtester import Backtester, summarize_by_market
import visualization as viz


st.set_page_config(page_title="Prediction Market Mispricing Engine", layout="wide")

st.title("Prediction Market Probability Surface & Mispricing Engine")
st.caption(
    "Converts event-market prices into clean probabilities, models fair value, "
    "and flags dislocations — demoed here on a Fed-rate-decision market category."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
use_live = st.sidebar.checkbox("Use live Polymarket data", value=False,
                                help="Requires network access. Falls back to synthetic data if the fetch fails.")

st.sidebar.header("Model")
model_choice = st.sidebar.radio("Fair-value model", ["Bayesian (logit-update)", "Gradient boosted regressor"])

st.sidebar.header("Backtest parameters")
entry_threshold = st.sidebar.slider("Minimum edge to trade", 0.0, 0.05, 0.01, 0.001, format="%.3f")
min_trust = st.sidebar.slider("Minimum liquidity/trust weight", 0.0, 1.0, 0.10, 0.05)
fee_bps = st.sidebar.slider("Round-trip fee (bps)", 0, 100, 20, 5)


@st.cache_data(show_spinner="Loading market data...")
def _load(use_live: bool):
    return load_data(use_live=use_live)


@st.cache_data(show_spinner="Building features...")
def _build_features(price_df: pd.DataFrame, markets_df: pd.DataFrame, macro_df: pd.DataFrame):
    clean = build_clean_probability_table(price_df)
    features = build_feature_table(clean, markets_df, macro_df)
    return clean, features


markets_df, price_df, macro_df = _load(use_live)

if price_df.empty:
    st.warning("No price history available (live fetch returned metadata only). "
               "Uncheck 'Use live Polymarket data' to see the full synthetic demo.")
    st.stop()

clean_df, features_df = _build_features(price_df, markets_df, macro_df)
train_df, test_df = time_based_train_test_split(features_df)

if model_choice.startswith("Bayesian"):
    model = BayesianUpdateModel().fit(train_df)
else:
    model = GradientBoostFairValueModel().fit(train_df)

test_df = test_df.copy()
test_df["fair_prob"] = model.predict(test_df)

bt = Backtester(entry_threshold=entry_threshold, min_trust_weight=min_trust, fee_bps=fee_bps)
trades_df, metrics = bt.run(test_df)

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", metrics.get("n_trades", 0))
c2.metric("Total P&L", f"{metrics.get('total_pnl', 0):.4f}" if metrics.get("n_trades") else "—")
c3.metric("Hit rate", f"{metrics.get('hit_rate', 0):.1%}" if metrics.get("n_trades") else "—")
c4.metric("Max drawdown", f"{metrics.get('max_drawdown', 0):.4f}" if metrics.get("n_trades") else "—")
c5.metric("Brier score", f"{metrics.get('brier_score', float('nan')):.5f}")

st.divider()

# ---------------------------------------------------------------------------
# Market selector + probability path
# ---------------------------------------------------------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Market")
    market_ids = sorted(clean_df["market_id"].unique())
    selected_market = st.selectbox("Select a market", market_ids, index=len(market_ids) - 1)

    event_title = markets_df.loc[markets_df["market_id"] == selected_market, "event_title"].iloc[0]
    st.write(f"**{event_title}**")

    latest = (
        clean_df[clean_df["market_id"] == selected_market]
        .sort_values("timestamp")
        .groupby("outcome_name")
        .tail(1)[["outcome_name", "clean_prob"]]
        .rename(columns={"clean_prob": "probability"})
        .set_index("outcome_name")
    )
    st.write("Latest implied probabilities:")
    st.dataframe((latest * 100).round(1).astype(str) + "%", use_container_width=True)

with right:
    st.subheader("Probability path into resolution")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    viz.plot_probability_paths(clean_df, selected_market, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# Mispricing + calibration
# ---------------------------------------------------------------------------
m1, m2 = st.columns(2)

with m1:
    st.subheader("Largest current mispricings")
    fig, ax = plt.subplots(figsize=(7, 6))
    viz.plot_mispricing_table(test_df, top_n=12, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with m2:
    st.subheader("Model calibration")
    fig, ax = plt.subplots(figsize=(6, 6))
    viz.plot_calibration(test_df, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
st.subheader("Backtest: cumulative P&L")
fig, ax = plt.subplots(figsize=(10, 4.5))
viz.plot_backtest_pnl(trades_df, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.subheader("By-market breakdown")
st.dataframe(summarize_by_market(trades_df), use_container_width=True)

with st.expander("Trade log"):
    if trades_df.empty:
        st.write("No trades met the current edge/liquidity threshold.")
    else:
        st.dataframe(
            trades_df[["market_id", "outcome_name", "timestamp", "clean_prob", "fair_prob",
                       "edge", "signal", "pnl"]].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

st.caption(
    "Demo data is synthetic when live access is unavailable — see `src/data_loader.py` "
    "for the real Polymarket API client. This is a methodology demonstration, not a "
    "live trading signal."
)
