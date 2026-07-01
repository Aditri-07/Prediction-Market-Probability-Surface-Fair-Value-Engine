"""
data_loader.py

Two data paths, used deliberately so the provenance of every number in this
project is always traceable:

1. LIVE MODE  (`source="polymarket"`)
   Pulls real markets/prices from Polymarket's public REST APIs:
     - Gamma API   (market/event metadata):  https://gamma-api.polymarket.com
     - CLOB API    (order books, price history): https://clob.polymarket.com
   No API key is required for public market data as of 2026-Q1. Requires
   outbound network access to *.polymarket.com, which may be blocked in
   sandboxed/CI environments.

2. SYNTHETIC MODE (`source="synthetic"`, the default in this repo's demo run)
   Generates a mechanically realistic but entirely fabricated set of Fed
   rate-decision markets (prices, order books, resolution outcomes) using a
   seeded random process calibrated to look like real Fed-cut markets
   (correlated outcome buckets, bid/ask spreads, vig, drift toward
   resolution). This mode exists ONLY so the rest of the pipeline
   (probability cleaning, surface construction, fair-value model,
   backtest) can be run and unit-tested without network access.

   IMPORTANT: Synthetic output is never to be described as "real Polymarket
   data" anywhere in this repo, in a resume bullet, or in an interview.
   Every number produced in synthetic mode is a demonstration of the
   pipeline's mechanics, not a market-verified claim.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


@dataclass
class MarketSnapshot:
    market_id: str
    event_title: str
    outcome: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    volume_24h: float
    liquidity: float
    source: str = "synthetic"


# --------------------------------------------------------------------------
# LIVE MODE
# --------------------------------------------------------------------------

def fetch_polymarket_events(query: str = "fed", limit: int = 50) -> pd.DataFrame:
    """
    Pull live event metadata from Polymarket's Gamma API, filtered by a
    keyword (e.g. "fed", "fomc", "cpi", "inflation").

    Returns columns: event_id, title, category, end_date, outcomes (list)
    """
    params = {"limit": limit, "active": "true", "closed": "false"}
    resp = requests.get(f"{GAMMA_API}/events", params=params, timeout=15)
    resp.raise_for_status()
    events = resp.json()

    rows = []
    for ev in events:
        title = ev.get("title", "")
        if query.lower() not in title.lower():
            continue
        rows.append(
            {
                "event_id": ev.get("id"),
                "title": title,
                "category": ev.get("category"),
                "end_date": ev.get("endDate"),
                "markets": ev.get("markets", []),
            }
        )
    return pd.DataFrame(rows)


def fetch_polymarket_price_history(clob_token_id: str, interval: str = "1h") -> pd.DataFrame:
    """
    Pull historical price series for a single CLOB token (one side of one
    outcome) from the Polymarket CLOB API.
    """
    params = {"market": clob_token_id, "interval": interval}
    resp = requests.get(f"{CLOB_API}/prices-history", params=params, timeout=15)
    resp.raise_for_status()
    hist = resp.json().get("history", [])
    df = pd.DataFrame(hist)
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df


def fetch_polymarket_orderbook(clob_token_id: str) -> dict:
    """Pull the current order book for a single CLOB token."""
    resp = requests.get(f"{CLOB_API}/book", params={"token_id": clob_token_id}, timeout=15)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------
# SYNTHETIC MODE
# --------------------------------------------------------------------------

FED_OUTCOMES = ["0 cuts", "1 cut", "2 cuts", "3+ cuts"]


def _dirichlet_path(rng: np.random.Generator, n_steps: int, n_outcomes: int,
                     start_alpha: np.ndarray, drift_strength: float = 0.06) -> np.ndarray:
    """
    Simulate a probability path for a multi-outcome event market that
    gradually concentrates toward a terminal outcome as resolution nears
    (mimicking real Fed-decision market behavior: uncertainty collapses
    into the realized outcome as the FOMC date approaches).
    """
    terminal = rng.integers(0, n_outcomes)
    alpha = start_alpha.copy()
    path = np.zeros((n_steps, n_outcomes))
    for t in range(n_steps):
        frac = t / max(n_steps - 1, 1)
        alpha_t = alpha.copy()
        alpha_t[terminal] += drift_strength * frac * n_steps
        probs = rng.dirichlet(alpha_t)
        path[t] = probs
    return path, terminal


def generate_synthetic_fed_markets(
    n_meetings: int = 8,
    days_before_meeting: int = 30,
    snapshots_per_day: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic panel of Fed rate-decision prediction markets:
    n_meetings FOMC-style events, each with 4 outcome buckets
    (0 / 1 / 2 / 3+ cuts), sampled at `snapshots_per_day` intervals for
    `days_before_meeting` days leading into resolution.

    Bid/ask spreads and per-outcome liquidity are simulated with
    realistic magnitudes drawn from public Polymarket macro-market ranges
    (spreads ~0.5-3 cents on liquid outcomes, wider on tail buckets).
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for m in range(n_meetings):
        meeting_date = base_date + timedelta(days=45 * m)
        n_steps = days_before_meeting * snapshots_per_day
        start_alpha = np.array([2.0, 3.0, 2.0, 1.0]) * rng.uniform(0.7, 1.4)
        path, terminal_idx = _dirichlet_path(rng, n_steps, len(FED_OUTCOMES), start_alpha)

        event_id = f"FOMC-{meeting_date.strftime('%Y%m%d')}"
        for t in range(n_steps):
            ts = meeting_date - timedelta(days=days_before_meeting) + timedelta(
                hours=24 * t / snapshots_per_day
            )
            true_probs = path[t]
            for i, outcome in enumerate(FED_OUTCOMES):
                p = true_probs[i]
                # liquidity higher for higher-probability outcomes
                liquidity = max(5_000, 250_000 * p * rng.uniform(0.6, 1.4))
                # spread widens for thin/tail outcomes
                spread = np.clip(0.008 + 0.05 * (1 - p), 0.004, 0.08)
                noise = rng.normal(0, 0.01)
                mid = np.clip(p + noise, 0.01, 0.99)
                best_bid = np.clip(mid - spread / 2, 0.01, 0.99)
                best_ask = np.clip(mid + spread / 2, 0.01, 0.99)
                rows.append(
                    {
                        "event_id": event_id,
                        "meeting_date": meeting_date,
                        "outcome": outcome,
                        "outcome_idx": i,
                        "timestamp": ts,
                        "best_bid": round(best_bid, 4),
                        "best_ask": round(best_ask, 4),
                        "volume_24h": round(liquidity * rng.uniform(0.05, 0.25), 2),
                        "liquidity": round(liquidity, 2),
                        "true_latent_prob": round(p, 4),  # only known b/c synthetic
                        "resolved_outcome": FED_OUTCOMES[terminal_idx],
                        "source": "synthetic",
                    }
                )
    df = pd.DataFrame(rows)
    df["days_to_resolution"] = (df["meeting_date"] - df["timestamp"]).dt.total_seconds() / 86400
    return df


def generate_synthetic_macro_features(df_markets: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """
    Attach synthetic macro signals (2Y yield daily change, Fed Funds futures
    implied move, CPI surprise on release days, nonfarm payrolls surprise)
    to each timestamp in the market panel. These mimic the *shape* of real
    macro data (small daily yield moves, occasional surprise spikes on
    release dates) but are not sourced from FRED/BLS.
    """
    rng = np.random.default_rng(seed)
    ts = pd.to_datetime(df_markets["timestamp"]).sort_values().unique()
    feat = pd.DataFrame({"timestamp": ts})
    feat["yield_2y_chg_bps"] = rng.normal(0, 3.5, len(feat)).round(2)
    feat["fed_funds_futures_chg_bps"] = rng.normal(0, 2.0, len(feat)).round(2)

    # CPI/NFP "surprise" only on ~1 in 20 timestamps (mimics monthly releases)
    is_release = rng.random(len(feat)) < 0.05
    feat["cpi_surprise_bps"] = np.where(is_release, rng.normal(0, 8, len(feat)), 0).round(2)
    feat["nfp_surprise_k"] = np.where(is_release, rng.normal(0, 40, len(feat)), 0).round(1)
    return feat


def save_raw(df: pd.DataFrame, name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def load_or_generate(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """
    Single entry point used by the rest of the pipeline.
    source="synthetic" -> generate_synthetic_fed_markets()
    source="polymarket" -> fetch_polymarket_events() + price history (live)
    """
    if source == "synthetic":
        return generate_synthetic_fed_markets(**kwargs)
    elif source == "polymarket":
        events = fetch_polymarket_events(query=kwargs.get("query", "fed"))
        if events.empty:
            raise RuntimeError("No live Polymarket events matched the query.")
        # NOTE: full live price-history assembly per event is intentionally
        # left as a documented extension point -- see README "Going live".
        return events
    else:
        raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    df = generate_synthetic_fed_markets()
    path = save_raw(df, "synthetic_fed_markets")
    macro = generate_synthetic_macro_features(df)
    macro_path = save_raw(macro, "synthetic_macro_features")
    print(f"Saved {len(df)} rows -> {path}")
    print(f"Saved {len(macro)} rows -> {macro_path}")
