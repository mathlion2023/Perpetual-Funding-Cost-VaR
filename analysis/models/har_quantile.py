"""Section 4.6.1 (added 2026-09-28): HAR-style direct quantile model.

The static direct-quantile model (caviar.direct_quantile_var) augmented with
heterogeneous-autoregressive (HAR) averages of the funding rate and of
realised volatility over trailing daily, weekly, and monthly windows
(Corsi, 2009). Its role in the proposal is to be the long-memory control
that rough_vol's fractional-kernel feature has to beat: a power-law kernel
is partly just a long-memory weighting scheme, so if the kernel feature does
not improve on plain HAR averages, rough_vol's gain is attributed to long
memory rather than roughness.

All HAR averages are *trailing* windows ending at hour t (inclusive), built
from contemporaneous series only (f_eff, rv_1h), so they are observable by
the end of hour t and carry no target_col look-ahead. They are computed on
train_full (every row dated <= the origin) and then joined onto
train_labeled by timestamp; the model itself is still supervised on
train_labeled only (see base.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .caviar import DEFAULT_FEATURES, direct_quantile_var

# 24h is already represented by rv_primary (a 24h realised-vol aggregate) for
# the RV leg; the funding leg needs all three windows.
HAR_WINDOWS_HOURS = {"d": 24, "w": 168, "m": 720}


# Averages are taken over the hours actually observed in each trailing window,
# requiring at least this share of the window. Requiring every hour
# (min_periods=w) discarded any origin with a single missing f_eff hour in the
# previous 30 days -- 122 of direct_quantile's 471 H=72 forecasts on the real
# panel, whose f_eff has 8,743 missing hours (mostly the 2021 schema gap).
# At 25% coverage the 30-day average still rests on >= 7.5 days of data;
# origins below that (e.g. 2021-03-01, with 719 of 720 trailing hours
# missing) have no honest HAR average and are left as NaN.
MIN_WINDOW_COVERAGE = 0.25


def add_har_features(df: pd.DataFrame, windows: dict[str, int] = HAR_WINDOWS_HOURS) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    cols = []
    for tag, w in windows.items():
        c = f"f_eff_har_{tag}"
        out[c] = out["f_eff"].rolling(w, min_periods=int(w * MIN_WINDOW_COVERAGE)).mean()
        cols.append(c)
    for tag, w in windows.items():
        if tag == "d":
            continue  # rv_primary already carries the 24h RV average
        c = f"rv_har_{tag}"
        out[c] = out["rv_1h"].rolling(w, min_periods=int(w * MIN_WINDOW_COVERAGE)).mean()
        cols.append(c)
    return out, cols


def har_quantile_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    min_obs: int = 60,
) -> float:
    aug_full, har_cols = add_har_features(train_full[list(dict.fromkeys(DEFAULT_FEATURES + ["rv_1h", target_col]))])
    aug_labeled = aug_full.loc[train_labeled.index]
    aug_origin = aug_full.iloc[-1]  # train_full ends at the origin row (base.run_rolling_backtest)
    if aug_origin.name != origin_row.name:
        return np.nan
    return direct_quantile_var(
        aug_full,
        aug_labeled,
        aug_origin,
        target_col=target_col,
        alpha=alpha,
        horizon_hours=horizon_hours,
        feature_cols=DEFAULT_FEATURES + har_cols,
        min_obs=min_obs,
    )
