"""Section 8.1: historical simulation benchmark.

Transparent non-parametric benchmark: the funding-cost VaR forecast is the
empirical alpha-quantile of past realised H-hour cumulative funding costs.
Uses train_labeled only, since it is directly supervised on target_col.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_simulation_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    min_obs: int = 30,
) -> float:
    s = train_labeled[target_col].dropna()
    if len(s) < min_obs:
        return np.nan
    return float(s.quantile(alpha))
