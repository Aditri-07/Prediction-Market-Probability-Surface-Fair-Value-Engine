# Prediction Market Probability Surface & Mispricing Engine

A research pipeline that converts event-market prices (Polymarket first,
transferable to Kalshi) into clean implied probabilities, models fair value
with macro/event features, and flags dislocations between model and market —
demoed end-to-end on a Fed-rate-decision market category.

> **Data note:** this repo ships with a synthetic-but-structurally-realistic
> Fed-rate-market dataset so the whole pipeline runs with zero setup and no
> API keys. The real Polymarket API client (`src/data_loader.py ->
> PolymarketClient`) is included and ready to use — flip `use_live=True` in
> any notebook once you have network access. See [Swapping in real data](#swapping-in-real-data).

## What it does

1. **Clean implied probabilities** — mid-price from bid/ask, de-vig by
   normalizing outcome probabilities to sum to 1 within each market snapshot.
2. **Probability surface** — implied probability across outcome bucket,
   event date, and time-to-resolution (the prediction-market analogue of a
   vol surface).
3. **Fair-value model** — two estimators sharing one interface:
   - a **Bayesian logit-update model** (interpretable: prior = yesterday's
     market probability, evidence = standardized macro surprises, updated
     additively in logit space)
   - a **gradient boosted regressor** benchmark
4. **Mispricing signal** — `edge = fair_prob − market_prob`, filtered by a
   minimum edge, minimum liquidity, and days-to-resolution.
5. **Backtest** — edge-filtered long/short strategy marked to next-available
   market price, net of a transaction-cost assumption. Reports P&L, hit
   rate, max drawdown, a Sharpe-like ratio, Brier score, and log loss.
6. **Dashboard** — Streamlit app to explore markets, mispricings,
   calibration, and backtest results interactively.

## Project layout

```
prediction-market-probability-surface/
│
├── data/
│   ├── raw/                  # raw market/price/macro pulls (gitignored)
│   └── processed/            # cleaned/feature tables (gitignored)
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_probability_surface.ipynb
│   ├── 03_fair_value_model.ipynb
│   └── 04_backtest.ipynb
│
├── src/
│   ├── data_loader.py         # Polymarket API client + synthetic fallback
│   ├── probability_cleaning.py # mid-price, de-vig, liquidity weighting
│   ├── feature_engineering.py  # momentum + macro feature table
│   ├── fair_value_model.py     # Bayesian logit-update + GBR models
│   ├── backtester.py           # edge-filtered strategy + metrics
│   └── visualization.py        # shared plotting helpers
│
├── dashboard/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone <your-fork-url>
cd prediction-market-probability-surface
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run the notebooks in order (01 -> 04), or run any module standalone:
python src/data_loader.py
python src/probability_cleaning.py
python src/feature_engineering.py
python src/fair_value_model.py
python src/backtester.py

# Or launch the interactive dashboard:
streamlit run dashboard/streamlit_app.py
```

Every `src/*.py` module has a `if __name__ == "__main__":` block that runs
a small end-to-end demo of just that stage, useful for sanity-checking one
piece at a time.

## Methodology notes

**De-vigging.** Multi-outcome markets (e.g. "0 / 1 / 2 / 3+ cuts") don't sum
to exactly 100% because of bid/ask spread and market-maker vig. We normalize
per timestamp: `clean_prob = mid / sum(mid across outcomes)`.

**Bayesian fair-value model.** Rather than a black box, this model works in
logit space, where probability updates are additive:

```
fair_prob = sigmoid( logit(prior_prob) + β · evidence )
```

`prior_prob` is yesterday's clean market probability; `evidence` is a
standardized vector of macro surprises and momentum features; `β` is fit by
OLS regressing the realized logit *change* on that evidence. Each
coefficient is directly interpretable as "logits of probability shift per
1 std-dev of this signal" — see the `coefficients()` method.

**Backtest honesty.** The default backtest parameters (1% minimum edge, 20bps
round-trip fee, minimum liquidity filter) were chosen by a small sensitivity
sweep (see notebook 04) rather than picked to maximize P&L on this one run —
the sweep itself is included so you can see how sensitive results are to
each assumption. Treat any P&L number here as a **methodology demo**, not a
trading track record: this is a short backtest window on one market
category, with no slippage or partial-fill modeling.

**Time-based splitting.** Train/test splits are chronological
(`fair_value_model.time_based_train_test_split`), never random — the model
never sees future data during fitting.

## Swapping in real data

`src/data_loader.py` includes `PolymarketClient`, a thin wrapper around
Polymarket's public **Gamma API** (market metadata) and **CLOB API** (price
history) — no API key required for read-only data:

```python
from data_loader import PolymarketClient

client = PolymarketClient()
markets = client.search_markets("Fed")                 # or "FOMC", "rate cut", etc.
prices = client.get_price_history(clob_token_id)        # per-outcome token id
```

Everything downstream — `probability_cleaning.py`, `feature_engineering.py`,
`fair_value_model.py`, `backtester.py` — only depends on the standardized
schema (`market_id, outcome_name, timestamp, bid, ask, volume`), so it's
agnostic to the data source. The same schema maps cleanly onto **Kalshi's**
REST API (`/trade-api/v2/markets`, `.../orderbook`) if you want to point
this at Kalshi instead — that's the intended next step for this project
given Kalshi's CFTC-regulated, US-accessible event markets.

## Results (this demo run)

On the included synthetic Fed-rate-market dataset, the Bayesian fair-value
model at a 1% minimum-edge threshold and 20bps fee assumption produced a
positive-P&L, ~55% hit-rate backtest over a ~3-month synthetic test window
(see `notebooks/04_backtest.ipynb` for exact numbers and the full
threshold/fee sensitivity sweep). Brier score and log loss for the
fair-value forecasts are reported alongside the trading metrics, since a
model can be a good forecaster even when the filtered strategy is roughly
breakeven, and vice versa.


## Development notes: problems faced while building this

Worth knowing about (and worth being able to talk through if asked about
this project):

- **No `nbformat` package available in the build environment.** Rather than
  skip the notebooks, I hand-wrote a small script to generate valid
  `.ipynb` JSON directly (see the cell structure — `nbformat: 4`,
  `nbformat_minor: 5`). This is a fine workaround but `nbformat` is the
  standard way to do this if you're building notebooks programmatically
  elsewhere.
- **No live network access in the build/test environment**, so the real
  `PolymarketClient` (Gamma/CLOB API calls) could not be exercised
  end-to-end during development. It's written against Polymarket's
  documented public API shape, but you should treat the first live run as
  a real test — API response shapes drift, and field names like
  `clobTokenIds` or `endDate` may need small adjustments. This is exactly
  why the pipeline is split into a data layer (`data_loader.py`) and a
  data-agnostic modeling layer (everything downstream) — a live-API bug
  only touches one file.
- **Initial backtest produced zero trades.** The first default parameters
  (5% minimum edge, 15% minimum liquidity weight) were too strict — the
  Bayesian model tracks the (synthetic) market closely by construction
  (`prior_prob` dominates feature importance), so realized edges were
  almost all under 3%. Fixed by running an actual threshold/fee sensitivity
  sweep rather than guessing a smaller number, which is now reproduced in
  notebook 04 rather than hidden.
- **A duplicate-column bug between notebooks 02 and 03.** Notebook 02 was
  originally persisting a `clean_probabilities.csv` that already had
  `expiration_date`/`days_to_resolution` merged in (added for a 3D surface
  plot). Notebook 03's feature pipeline re-merges `expiration_date` from
  `markets.csv`, which silently created `expiration_date_x`/`_y` columns
  instead of the expected single column, causing a `KeyError`. Fixed two
  ways: `add_time_to_resolution()` now drops those columns before
  re-merging (idempotent), and notebook 02 only persists the minimal
  pre-merge schema, keeping the "what does each stage read/write" contract
  explicit.

## License

MIT — see `LICENSE`.


