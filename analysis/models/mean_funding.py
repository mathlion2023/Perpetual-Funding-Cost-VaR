"""Revised design Section 6.4, P1: mean-funding reserve.

Transparent benchmark -- the funding reserve forecast is the rolling
expanding-window mean of past realised H-hour cumulative adverse funding
loss (target_col is C_long_H{h} or C_short_H{h}, both non-negative by
construction, so no floor is needed on the raw mean; the max(.,0) in
build_reserve_policies is kept only for symmetry with the other policies
and to guard a small/degenerate sample producing a negative fitted mean
under future reuse with a different target).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_funding_reserve(
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
    return max(float(s.mean()), 0.0)
