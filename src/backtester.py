"""
backtester.py
==============

Turns fair-value predictions into a simple edge-filtered trading
strategy and evaluates it.

Strategy
--------
edge = fair_prob - market_prob (using ask when buying YES, bid when
selling/avoiding, to be conservative about crossing the spread)

* BUY the outcome (pay ask) if edge > entry_threshold AND
  liquidity/trust_weight above a minimum AND days_to_resolution > 0
* Otherwise stay flat.

Positions are marked to the *next available* clean_prob (approximating
the eventual resolution / exit price) to compute P&L. This is a
simplified backtest meant to demonstrate the methodology, not a
production execution simulator (no partial fills, no market impact).

Metrics reported
-----------------
* Total simulated P&L (in probability points, i.e. cents per $1 notional)
* Hit rate (% of trades that were profitable)
* Max drawdown (on cumulative P&L)
* Sharpe-like ratio (mean / std of per-trade P&L, annualized-ish by sqrt(N))
* Brier score and log loss of the fair-value model vs. realized outcomes
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Backtester:
    def __init__(self, entry_threshold: float = 0.01, min_trust_weight: float = 0.1,
                 fee_bps: float = 20):
        """
        entry_threshold : minimum |edge| required to trade (in probability points, e.g. 0.05 = 5%)
        min_trust_weight : minimum liquidity/trust score (0-1) required to trade
        fee_bps : round-trip transaction cost in basis points of notional (100bps = 1%)
        """
        self.entry_threshold = entry_threshold
        self.min_trust_weight = min_trust_weight
        self.fee = fee_bps / 10_000

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """`df` must contain: clean_prob, fair_prob, trust_weight, days_to_resolution."""
        out = df.copy()
        out["edge"] = out["fair_prob"] - out["clean_prob"]
        out["signal"] = 0
        long_mask = (
            (out["edge"] > self.entry_threshold)
            & (out["trust_weight"] >= self.min_trust_weight)
            & (out["days_to_resolution"] > 0)
        )
        short_mask = (
            (out["edge"] < -self.entry_threshold)
            & (out["trust_weight"] >= self.min_trust_weight)
            & (out["days_to_resolution"] > 0)
        )
        out.loc[long_mask, "signal"] = 1
        out.loc[short_mask, "signal"] = -1
        return out

    def compute_trade_pnl(self, df: pd.DataFrame) -> pd.DataFrame:
        """For each signaled trade, mark P&L to the *next day's* clean_prob
        for that (market_id, outcome_name), minus fees. This approximates
        "buy today, exit/resolve one step later."
        """
        df = df.sort_values(["market_id", "outcome_name", "timestamp"]).copy()
        df["next_prob"] = df.groupby(["market_id", "outcome_name"])["clean_prob"].shift(-1)
        df = df.dropna(subset=["next_prob"])

        df["raw_pnl"] = df["signal"] * (df["next_prob"] - df["clean_prob"])
        df["pnl"] = df["raw_pnl"] - (df["signal"].abs() * self.fee)
        return df

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        signaled = self.generate_signals(df)
        priced = self.compute_trade_pnl(signaled)
        trades = priced[priced["signal"] != 0].copy()
        metrics = self._compute_metrics(trades, signaled)
        return trades, metrics

    @staticmethod
    def _compute_metrics(trades: pd.DataFrame, full_df: pd.DataFrame) -> dict:
        if trades.empty:
            return {"n_trades": 0, "message": "No trades met the edge/liquidity threshold."}

        trades = trades.sort_values("timestamp")
        cum_pnl = trades["pnl"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_drawdown = drawdown.min()

        hit_rate = (trades["pnl"] > 0).mean()
        mean_pnl = trades["pnl"].mean()
        std_pnl = trades["pnl"].std(ddof=1) if len(trades) > 1 else np.nan
        sharpe_like = (mean_pnl / std_pnl) * np.sqrt(len(trades)) if std_pnl and std_pnl > 0 else np.nan

        brier, logloss = _forecast_scores(full_df)

        return {
            "n_trades": int(len(trades)),
            "total_pnl": float(cum_pnl.iloc[-1]),
            "mean_pnl_per_trade": float(mean_pnl),
            "hit_rate": float(hit_rate),
            "max_drawdown": float(max_drawdown),
            "sharpe_like": float(sharpe_like) if not np.isnan(sharpe_like) else None,
            "brier_score": brier,
            "log_loss": logloss,
        }


def _forecast_scores(df: pd.DataFrame, eps: float = 1e-4) -> tuple[float, float]:
    """Brier score and log loss of `fair_prob` vs. realized `next_prob`
    (treated as the pseudo-realized outcome probability), evaluated over
    all scored rows regardless of whether a trade was taken.
    """
    work = df.dropna(subset=["fair_prob", "clean_prob"]).copy()
    if "next_prob" not in work.columns:
        work["next_prob"] = work.groupby(["market_id", "outcome_name"])["clean_prob"].shift(-1)
    work = work.dropna(subset=["next_prob"])
    if work.empty:
        return float("nan"), float("nan")

    y = work["next_prob"].clip(eps, 1 - eps)
    p = work["fair_prob"].clip(eps, 1 - eps)

    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, logloss


def summarize_by_market(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (
        trades.groupby("market_id")
        .agg(n_trades=("pnl", "size"), total_pnl=("pnl", "sum"), hit_rate=("pnl", lambda s: (s > 0).mean()))
        .sort_values("total_pnl", ascending=False)
    )


if __name__ == "__main__":
    from data_loader import load_data
    from probability_cleaning import build_clean_probability_table
    from feature_engineering import build_feature_table
    from fair_value_model import BayesianUpdateModel, time_based_train_test_split

    markets, prices, macro = load_data(use_live=False)
    clean = build_clean_probability_table(prices)
    features = build_feature_table(clean, markets, macro)
    train, test = time_based_train_test_split(features)

    model = BayesianUpdateModel().fit(train)
    test = test.copy()
    test["fair_prob"] = model.predict(test)

    bt = Backtester(entry_threshold=0.01, min_trust_weight=0.1, fee_bps=20)
    trades, metrics = bt.run(test)

    print("Backtest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nBy-market summary:")
    print(summarize_by_market(trades))
