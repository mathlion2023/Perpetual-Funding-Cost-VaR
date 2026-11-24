"""Shared rolling out-of-sample forecasting harness for Section 8/10.

Every model in this package exposes a function with signature

    model_fn(train_full, train_labeled, origin_row, *, target_col, alpha,
             horizon_hours, **kwargs) -> float

- train_full    : all panel rows dated <= the forecast origin. Safe for any
                  model that only uses *contemporaneous* series (e.g. f_eff_t,
                  X_t) -- there is no look-ahead in these by construction.
- train_labeled : the subset of train_full whose target_col value is fully
                  realised as of the forecast origin (i.e. excludes the last
                  `horizon_hours` rows, whose target_col peeks `horizon_hours`
                  hours past the origin). Models that are directly supervised
                  on target_col (historical simulation, quantile
                  regression/CAViaR) MUST fit only on train_labeled, never on
                  train_full, or the backtest leaks future information.
- origin_row    : the single row at the forecast origin, i.e. X_t as observed
                  at the end of hour t -- used to generate *today's* forecast,
                  never used as a labeled training example.

This split exists because target_col (e.g. "C_long_H72") is a forward-summed
outcome: its value at row t already encodes information through t+H. Rows
within H hours of the origin cannot be used as (X, y) training pairs without
leaking the future into the backtest.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def assert_hourly_grid(index: pd.Index, what: str = "panel") -> None:
    """Every positional operation in this pipeline (forward_sum, shift,
    rolling, label_cutoff = pos - H, origin spacing) equates one row with one
    elapsed hour. That only holds on a complete, strictly increasing hourly
    grid with missing hours kept as NaN rows -- never dropped. Fail loudly
    otherwise (lag-audit, 2026-11-06)."""
    if len(index) < 2:
        return
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"{what} index must be a DatetimeIndex, got {type(index).__name__}")
    # Compare as timedeltas, not raw integers: pandas 3 stores datetimes at
    # microsecond resolution, so asi8 units differ by pandas version.
    steps = index[1:] - index[:-1]
    one_hour = pd.Timedelta("1h")
    if not (steps == one_hour).all():
        bad = int((steps != one_hour).sum())
        raise ValueError(f"{what} index is not a complete hourly grid ({bad} steps != 1h); "
                         "reindex onto pd.date_range(..., freq='h') and keep missing hours as NaN.")


def last_valid_is_origin(s: pd.Series) -> bool:
    """True if `s`'s last row (the origin hour) is observed. Simulation models
    start their paths from the origin hour's own state; if that hour is missing
    the "current" value would silently be an older hour, so they return NaN."""
    return len(s) > 0 and bool(np.isfinite(s.iloc[-1]))


def non_overlapping_origins(index: pd.DatetimeIndex, horizon_hours: int, min_train_hours: int) -> pd.DatetimeIndex:
    """Section 10: forecast origins spaced `horizon_hours` apart, starting once
    `min_train_hours` of history is available, so successive evaluation
    windows do not overlap.
    """
    n = len(index)
    positions = list(range(min_train_hours, n - horizon_hours, horizon_hours))
    return index[positions]


def run_rolling_backtest(
    panel: pd.DataFrame,
    model_fn: Callable[..., float],
    target_col: str,
    horizon_hours: int,
    alpha: float = 0.95,
    min_train_hours: int = 24 * 180,
    window_hours: int | None = None,
    origins: pd.DatetimeIndex | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """Rolling/expanding out-of-sample VaR backtest for one model, one horizon.

    window_hours=None => expanding window (all history up to the origin).
    window_hours=N    => rolling window of the trailing N hours.
    """
    idx = panel.index
    assert_hourly_grid(idx)
    if origins is None:
        origins = non_overlapping_origins(idx, horizon_hours, min_train_hours)

    pos_of = {ts: i for i, ts in enumerate(idx)}
    rows = []
    for origin in origins:
        pos = pos_of[origin]
        label_cutoff = pos - horizon_hours  # last position whose target_col is fully realised by `origin`
        train_start = 0 if window_hours is None else max(0, pos - window_hours + 1)

        train_full = panel.iloc[train_start : pos + 1]
        train_labeled = panel.iloc[train_start : max(train_start, label_cutoff + 1)]
        origin_row = panel.iloc[pos]

        try:
            forecast = model_fn(
                train_full,
                train_labeled,
                origin_row,
                target_col=target_col,
                alpha=alpha,
                horizon_hours=horizon_hours,
                **model_kwargs,
            )
            err = None
        except Exception as exc:  # noqa: BLE001 -- surfaced in output, not swallowed
            forecast = np.nan
            err = str(exc)
        rows.append({"origin": origin, "var_forecast": forecast, "error": err})

    out = pd.DataFrame(rows).set_index("origin")
    out["realized"] = panel.loc[out.index, target_col]
    out["exception"] = (out["realized"] > out["var_forecast"]).astype("float")
    return out
