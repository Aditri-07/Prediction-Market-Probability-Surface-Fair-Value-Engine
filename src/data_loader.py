"""
data_loader.py
==============

Two ways to get data into this pipeline:

1. `PolymarketClient` — talks to Polymarket's public REST APIs
   (Gamma API for market metadata, CLOB API for price history).
   Use this when you have internet access and want live/historical data.

2. `SyntheticFedMarketGenerator` — generates realistic, structurally
   correct fake data for a Fed-rate-decision-style event market
   (multiple outcomes, bid/ask spreads, drifting probabilities into
   an FOMC date, and correlated macro signals). Use this for
   development, demos, or CI environments without network access.

Both return data in the same shape so the rest of the pipeline
(`probability_cleaning`, `feature_engineering`, `fair_value_model`,
`backtester`) doesn't care which source it came from.

Canonical schemas
------------------
markets_df columns:
    market_id, event_title, outcome_name, expiration_date, category

price_history_df columns:
    market_id, outcome_name, timestamp, bid, ask, mid, volume

macro_df columns:
    date, fed_funds_futures_implied, y2_yield, y10_yield,
    cpi_surprise, jobs_surprise, vix
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# 1. Live Polymarket client
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketClient:
    """Thin wrapper around Polymarket's public Gamma + CLOB APIs.

    Notes
    -----
    * No API key is required for read-only market/price data.
    * Polymarket's schema changes occasionally — if a field is missing,
      check https://docs.polymarket.com for the current response shape
      and adjust the parsing below.
    * Be a good citizen: this client sleeps briefly between paginated
      requests to avoid hammering the API.
    """

    def __init__(self, timeout: int = 15, sleep_between_calls: float = 0.25):
        self.timeout = timeout
        self.sleep_between_calls = sleep_between_calls
        self.session = requests.Session()

    def search_markets(self, query: str, limit: int = 50, closed: bool = False) -> pd.DataFrame:
        """Search markets by keyword, e.g. 'Fed', 'FOMC', 'rate cut'."""
        params = {"limit": limit, "closed": str(closed).lower(), "search": query}
        resp = self.session.get(f"{GAMMA_API}/markets", params=params, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json()
        return self._markets_to_df(raw)

    def get_market_by_id(self, market_id: str) -> Optional[dict]:
        resp = self.session.get(f"{GAMMA_API}/markets/{market_id}", timeout=self.timeout)
        if resp.status_code != 200:
            return None
        return resp.json()

    def get_price_history(self, clob_token_id: str, interval: str = "1d",
                           fidelity: int = 60) -> pd.DataFrame:
        """Fetch historical price series for a single outcome token.

        `clob_token_id` comes from the market's `clobTokenIds` field
        (one per outcome). `fidelity` is resolution in minutes.
        """
        params = {"market": clob_token_id, "interval": interval, "fidelity": fidelity}
        resp = self.session.get(f"{CLOB_API}/prices-history", params=params, timeout=self.timeout)
        resp.raise_for_status()
        raw = resp.json().get("history", [])
        time.sleep(self.sleep_between_calls)
        if not raw:
            return pd.DataFrame(columns=["timestamp", "price"])
        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s")
        df = df.rename(columns={"p": "price"})[["timestamp", "price"]]
        return df

    @staticmethod
    def _markets_to_df(raw: list[dict]) -> pd.DataFrame:
        rows = []
        for m in raw:
            outcomes = m.get("outcomes", ["Yes", "No"])
            if isinstance(outcomes, str):
                import json as _json
                outcomes = _json.loads(outcomes)
            for outcome in outcomes:
                rows.append(
                    {
                        "market_id": m.get("id"),
                        "event_title": m.get("question"),
                        "outcome_name": outcome,
                        "expiration_date": m.get("endDate"),
                        "category": m.get("category", "uncategorized"),
                        "clob_token_ids": m.get("clobTokenIds"),
                    }
                )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Synthetic Fed-market generator (offline fallback / demo data)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConfig:
    n_fomc_meetings: int = 8
    days_before_meeting: int = 45
    outcomes: tuple = ("0 cuts", "1 cut", "2 cuts", "3+ cuts")
    seed: int = 42


class SyntheticFedMarketGenerator:
    """Generates a structurally realistic Fed-rate-decision market dataset.

    For each simulated FOMC meeting we generate a daily probability path
    for each outcome bucket that:
      * starts near a random prior
      * randomly walks and mean-reverts toward a "true" terminal outcome
      * always sums to ~100% (with a small vig baked into bid/ask)
      * has bid/ask spreads that widen with lower liquidity / further
        time-to-resolution

    We also generate a correlated macro feature set (rate-futures-implied
    probability, yields, CPI/jobs surprises, VIX) that nudges the price
    path, so the fair-value model in this project has real signal to find.
    """

    def __init__(self, config: SyntheticConfig = SyntheticConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        markets_rows = []
        price_rows = []
        macro_rows = []

        today = dt.date.today()
        # Space FOMC-style meetings roughly 6 weeks apart, working backward
        meeting_dates = [
            today - dt.timedelta(days=6 * 7 * (self.cfg.n_fomc_meetings - i))
            for i in range(self.cfg.n_fomc_meetings)
        ]

        for m_idx, meeting_date in enumerate(meeting_dates):
            market_id = f"FED-{meeting_date.isoformat()}"
            event_title = f"How many rate cuts at the {meeting_date.strftime('%B %Y')} FOMC meeting?"

            # "true" terminal outcome this cycle resolves to
            true_outcome_idx = self.rng.choice(len(self.cfg.outcomes), p=[0.25, 0.40, 0.25, 0.10])

            for outcome_idx, outcome in enumerate(self.cfg.outcomes):
                markets_rows.append(
                    {
                        "market_id": market_id,
                        "event_title": event_title,
                        "outcome_name": outcome,
                        "expiration_date": meeting_date.isoformat(),
                        "category": "fed_rates",
                    }
                )

            # Simulate a daily probability path into the meeting for all outcomes jointly
            n_days = self.cfg.days_before_meeting
            dates = [meeting_date - dt.timedelta(days=n_days - d) for d in range(n_days + 1)]

            # start from a randomized prior, drift toward the true outcome
            probs = self.rng.dirichlet(alpha=np.ones(len(self.cfg.outcomes)) * 3)
            target = np.zeros(len(self.cfg.outcomes))
            target[true_outcome_idx] = 0.72
            remaining = (1 - 0.72) / (len(self.cfg.outcomes) - 1)
            for i in range(len(self.cfg.outcomes)):
                if i != true_outcome_idx:
                    target[i] = remaining

            macro_state = {
                "fed_funds_futures_implied": probs[true_outcome_idx],
                "y2_yield": 4.3 + self.rng.normal(0, 0.05),
                "y10_yield": 4.1 + self.rng.normal(0, 0.05),
                "vix": 15 + self.rng.normal(0, 1.5),
            }

            for d_idx, date in enumerate(dates):
                t_to_resolution = n_days - d_idx
                # pull probs toward target as we approach resolution
                pull = 0.06 * (1 - t_to_resolution / n_days) + 0.01
                noise = self.rng.normal(0, 0.015, size=len(self.cfg.outcomes))
                probs = probs + pull * (target - probs) + noise
                probs = np.clip(probs, 0.01, None)
                probs = probs / probs.sum()

                # macro signals drift with a touch of mean reversion + noise
                macro_state["fed_funds_futures_implied"] = np.clip(
                    macro_state["fed_funds_futures_implied"]
                    + 0.05 * (probs[true_outcome_idx] - macro_state["fed_funds_futures_implied"])
                    + self.rng.normal(0, 0.01),
                    0.01, 0.99,
                )
                macro_state["y2_yield"] += self.rng.normal(0, 0.02)
                macro_state["y10_yield"] += self.rng.normal(0, 0.015)
                macro_state["vix"] = max(8, macro_state["vix"] + self.rng.normal(0, 0.6))
                cpi_surprise = self.rng.normal(0, 0.08) if date.day in (10, 11, 12) else 0.0
                jobs_surprise = self.rng.normal(0, 0.10) if date.day in (5, 6, 7) else 0.0

                macro_rows.append(
                    {
                        "date": date,
                        "market_id": market_id,
                        "fed_funds_futures_implied": macro_state["fed_funds_futures_implied"],
                        "y2_yield": macro_state["y2_yield"],
                        "y10_yield": macro_state["y10_yield"],
                        "cpi_surprise": cpi_surprise,
                        "jobs_surprise": jobs_surprise,
                        "vix": macro_state["vix"],
                    }
                )

                # liquidity / spread widens further from resolution
                base_spread = 0.01 + 0.04 * (t_to_resolution / n_days)
                volume_base = self.rng.integers(500, 5000)

                for outcome_idx, outcome in enumerate(self.cfg.outcomes):
                    mid = probs[outcome_idx]
                    spread = base_spread * self.rng.uniform(0.7, 1.3)
                    bid = np.clip(mid - spread / 2, 0.01, 0.99)
                    ask = np.clip(mid + spread / 2, 0.01, 0.99)
                    price_rows.append(
                        {
                            "market_id": market_id,
                            "outcome_name": outcome,
                            "timestamp": pd.Timestamp(date),
                            "bid": round(float(bid), 4),
                            "ask": round(float(ask), 4),
                            "volume": int(volume_base * self.rng.uniform(0.5, 1.5)),
                        }
                    )

        markets_df = pd.DataFrame(markets_rows).drop_duplicates()
        price_df = pd.DataFrame(price_rows)
        price_df["mid"] = (price_df["bid"] + price_df["ask"]) / 2
        macro_df = pd.DataFrame(macro_rows).drop_duplicates(subset=["date", "market_id"])

        return markets_df, price_df, macro_df


def load_data(use_live: bool = False, search_query: str = "Fed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single entry point used by the notebooks / dashboard.

    Parameters
    ----------
    use_live : if True, attempt to pull real data from Polymarket.
        Falls back to synthetic data if the request fails (e.g. no
        network access, as in a sandboxed environment).
    """
    if use_live:
        try:
            client = PolymarketClient()
            markets_df = client.search_markets(search_query)
            if markets_df.empty:
                raise ValueError("No live markets returned")
            # NOTE: live price history requires per-outcome clobTokenIds;
            # see PolymarketClient.get_price_history for the follow-up call.
            print("Fetched live markets from Polymarket. "
                  "Call client.get_price_history(token_id) per outcome for price series.")
            return markets_df, pd.DataFrame(), pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            print(f"Live fetch failed ({e}); falling back to synthetic data.")

    gen = SyntheticFedMarketGenerator()
    return gen.generate()


if __name__ == "__main__":
    markets, prices, macro = load_data(use_live=False)
    print(markets.head())
    print(prices.head())
    print(macro.head())
