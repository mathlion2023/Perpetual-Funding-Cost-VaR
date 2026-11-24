"""Project-wide configuration for the Funding-Cost VaR pipeline.

Scope, per `Funding_Cost_VaR_Data_Methodology_Upd.docx`: Deribit BTC perpetual
only (see Section 1's exchange-scope note). Do not add other venues here
without re-running the Section 2 validation checks against their own funding
conventions.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "OneDrive_1_9-23-2026"

FUNDING_DIR = DATA_ROOT / "Data - Funding" / "drbt_funding" / "btcperpetual"
TRADES_DIR = DATA_ROOT / "Data - Trades" / "drbt" / "btcperpetual"

# Auxiliary cross-venue liquidity control ONLY (Spread_binc, per Tina's
# 2026-09-26 email replacing the planned Depth/depth-imbalance control with a
# second, independent quoted-spread series). This does NOT extend the scope
# note above: only bid/ask are ever read from this feed (see
# data_loading.load_binance_quotes_range); funding_rate/mark_price/basis/OI
# and every funding-cost outcome remain Deribit-only.
BINANCE_FUNDING_DIR = DATA_ROOT / "Data - Funding" / "binc_funding" / "btcusd_perp"

# Second full venue, added 2026-11-06 (Tina) as a parallel comparison
# alongside the primary Deribit backtest -- NOT part of the original scope
# note above, which still governs every Deribit-only figure in the proposal.
# Binance BTCUSD_PERP is COIN-margined/inverse (see panel.py's
# build_binance_spread_hourly docstring), the same contract type as Deribit's
# BTC-PERPETUAL, but funding settles discretely three times daily (00:00,
# 08:00, 16:00 UTC) rather than continuously -- confirmed empirically
# (2026-11-06): the raw `funding_rate` field changes only at those three
# hours-of-day and is flat in between, vs. Deribit's near-continuous series
# (data_validation.validate_funding_conventions' "fraction changing within
# the hour" is ~0.000 for Binance vs ~0.92 for Deribit). `predicted_funding_rate`
# is entirely null for Binance in this data, so the Deribit-style
# intra-hour-mean-then-/8 accrual (panel.build_ticker_hourly) does not apply;
# see panel.build_binance_ticker_hourly for the discrete-settlement analogue.
BINANCE_TRADES_DIR = DATA_ROOT / "Data - Trades" / "binc" / "btcusd_perp"
BINANCE_SETTLEMENT_HOURS_UTC = (0, 8, 16)

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"

# --- Horizons (Section 4) -----------------------------------------------
# Primary horizons, in hours, conditional on Deribit funding being an
# effectively hourly-continuous series (validated in data_validation.py).
HORIZONS_HOURS = (72, 168)

# --- VaR confidence level ------------------------------------------------
VAR_ALPHA = 0.95

# --- Leverage levels for collateral-drain reporting (Section 5) ---------
LEVERAGE_LEVELS = (2, 5, 10)
COLLATERAL_DRAIN_THRESHOLDS = (0.05, 0.10, 0.20)

# --- Rolling-window lengths (hours) used for the persistence /
# high-funding indicator and for realised-volatility states (Section 3.4,
# Section 6) ---------------------------------------------------------------
RV_WINDOWS_HOURS = {"primary": 24, "robustness_short": 6, "robustness_long": 72}
VOLUME_SHOCK_LOOKBACK_HOURS = 24
HIGH_FUNDING_LOOKBACK_HOURS = 24
HIGH_FUNDING_QUANTILE = 0.90

# --- Estimation windows ---------------------------------------------------
# Rolling re-estimation window length, in hours, for out-of-sample
# forecasting (Section 8/10). Kept configurable; default gives ~1 year.
ROLLING_WINDOW_HOURS = 24 * 365

# --- Minimum in-sample size before the pipeline will produce its first
# out-of-sample forecast. ---------------------------------------------------
MIN_TRAIN_HOURS = 24 * 180  # ~6 months

RANDOM_SEED = 20260923
