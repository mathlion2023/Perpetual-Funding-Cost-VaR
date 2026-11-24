"""Section 6: the forecast information set X_t.

X_t = [f_eff_t, f_trail8h_t, Basis_t, RV_24h_t, dOI_t, Spread_t, Spread_binc_t,
       OFI_t, VolumeShock_t, HighFunding_t]

Spread_binc_t (Binance) was added 2026-09-26 alongside Spread_t (Deribit,
already in the frozen set) per Tina's email replacing the planned Depth
control -- the two are kept as separate regressors rather than merged into
one, since they are independent venues with different tick coverage (see
panel.build_binance_spread_hourly).

**Correction (2026-10-20):** the second funding-rate feature was `f_pred`
(built from `predicted_funding_rate`) until the field-definition fix in
panel.py::build_ticker_hourly. `predicted_funding_rate` is now what `f_eff`
itself is built from (confirmed empirically to be the live current rate,
not `funding_rate`'s mistaken original role), so `f_pred` would now be a
numerically-redundant duplicate of `f_eff_t`. Replaced with `f_trail8h_t`
(built from the OLD `f_eff` formula, i.e. `funding_rate`'s own intra-hour
mean -- `funding_rate` is itself a trailing 8h accrual, so this is a
genuinely distinct, slower-moving funding signal, not a duplicate), which
preserves X_t's original two-timescale design intent (a current/live
funding feature alongside a trailing/smoothed one).

Every component here must be observable by the end of hour t -- this module
only reads columns already built in panel.py that satisfy that by
construction (no forward_sum/forward_value calls here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

PREDICTOR_COLUMNS = [
    "f_eff",
    "f_trail8h",
    "basis",
    "rv_primary",
    "d_oi",
    "spread",
    "spread_binc",
    "ofi",
    "volume_shock",
    "high_funding",
]


def add_high_funding_indicator(
    panel: pd.DataFrame,
    lookback: int = config.HIGH_FUNDING_LOOKBACK_HOURS,
    quantile: float = config.HIGH_FUNDING_QUANTILE,
    method: str = "expanding",
    fixed_initial_hours: int | None = None,
) -> pd.DataFrame:
    """Section 6: persistent high-funding regime indicator.

    method="expanding": the q-th percentile threshold at t is estimated from
        f_bar_24h observed up to and including t (no future information used
        anywhere in the threshold or in the indicator).
    method="fixed_initial": the threshold is estimated once from the first
        `fixed_initial_hours` of f_bar_24h and held fixed thereafter.
    """
    out = panel.copy()
    f_bar = out["f_eff"].rolling(lookback, min_periods=lookback).mean()
    out["f_bar_24h"] = f_bar

    if method == "expanding":
        min_obs = max(lookback * 30, 24 * 90)  # require a reasonably sized sample before the indicator activates
        threshold = f_bar.expanding(min_periods=min_obs).quantile(quantile)
    elif method == "fixed_initial":
        n_init = fixed_initial_hours or config.MIN_TRAIN_HOURS
        init_threshold = f_bar.iloc[:n_init].quantile(quantile)
        threshold = pd.Series(init_threshold, index=out.index)
        threshold.iloc[:n_init] = np.nan  # indicator undefined before the initial window is observed
    else:
        raise ValueError(f"Unknown method: {method}")

    out["high_funding_threshold"] = threshold
    out["high_funding"] = (f_bar > threshold).astype(float)
    out.loc[threshold.isna() | f_bar.isna(), "high_funding"] = np.nan
    return out


def build_forecast_set(panel: pd.DataFrame, **high_funding_kwargs) -> pd.DataFrame:
    """Attach HighFunding and return the panel with all X_t columns present."""
    out = add_high_funding_indicator(panel, **high_funding_kwargs)
    missing = [c for c in PREDICTOR_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Missing predictor columns after construction: {missing}")
    return out


def get_X(panel: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or PREDICTOR_COLUMNS
    return panel[cols]
