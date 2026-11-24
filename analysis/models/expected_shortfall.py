"""Section 11 item 5: Expected Shortfall, built on top of the existing
direct-quantile VaR model rather than as a new estimation family.

No ES estimator existed anywhere in the pipeline before this. Rather than
fit a fifth conditional-distribution model, ES_alpha is approximated the
standard non-parametric way as the average of conditional VaR forecasts
over a fine grid of quantiles spanning [alpha, 1):

    ES_alpha(X_t) ~= mean_{u in grid} VaR_u(X_t),  grid = alpha, ..., 0.995

This reuses caviar.direct_quantile_var (Section 8.2) at each grid quantile,
so it inherits that model's feature set and out-of-sample discipline
(fits only on train_labeled) for free, and requires no change to the
run_rolling_backtest harness in models/base.py -- this is just another
model_fn with the same signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import caviar

DEFAULT_TAIL_GRID_POINTS = 6  # alpha, then 5 more points up to 0.995


def direct_quantile_es(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    feature_cols: list[str] | None = None,
    min_obs: int = 60,
    tail_grid_points: int = DEFAULT_TAIL_GRID_POINTS,
    clip_nonnegative: bool = True,
) -> float:
    """ES_alpha via averaging direct_quantile_var over quantiles in [alpha, 0.995].

    `clip_nonnegative` (bug fix, 2026-09-29): forwarded to direct_quantile_var
    -- see that function's docstring. Both current callers (price_es_backtest,
    net_es_backtest in omission_gap.py) forecast signed targets (L_price_H{h},
    L_net_H{h}) and must pass False; ES for a future non-negative target would
    keep the default True.
    """
    grid = np.linspace(alpha, 0.995, tail_grid_points)
    forecasts = []
    for u in grid:
        v = caviar.direct_quantile_var(
            train_full,
            train_labeled,
            origin_row,
            target_col=target_col,
            alpha=float(u),
            horizon_hours=horizon_hours,
            feature_cols=feature_cols,
            min_obs=min_obs,
            clip_nonnegative=clip_nonnegative,
        )
        if np.isnan(v):
            return np.nan
        forecasts.append(v)
    return float(np.mean(forecasts))
