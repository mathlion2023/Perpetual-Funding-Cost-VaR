"""Section 4.6.5 (added 2026-09-28): stationary block bootstrap benchmark.

Historical simulation resamples *terminal* H-hour outcomes, which throws away
within-horizon timing. The stationary bootstrap (Politis & Romano, 1994)
instead builds each simulated H-hour path from contiguous blocks of the
observed hourly funding series, with geometrically distributed block lengths
(mean `mean_block_hours`), wrapping circularly at the end of the sample. This
keeps short-run serial dependence without any parametric model.

Stage 3 role: a distribution-free robustness benchmark for C_long_H{h}.
Section 4.12 role (cash-timing extension, not wired in here): the primary
naive benchmark, where blocks would be drawn jointly from funding and price.

Resamples f_eff from train_full only (contemporaneous series, no target_col
look-ahead). Paths start at a random point in history, not at the origin's
state, so this is deliberately an *unconditional* benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stationary_bootstrap_paths(x: np.ndarray, n_paths: int, horizon: int, mean_block: float,
                               rng: np.random.Generator) -> np.ndarray:
    """`x` may be on the complete hourly grid with NaN for unobserved hours
    (lag-audit fix, 2026-11-06): blocks start only at observed hours and a
    block is cut (a new block starts) wherever its next hour is unobserved,
    so no block splices together hours that are not consecutive in time. The
    circular wrap is kept only when both ends of the sample are observed.
    With no NaNs this is exactly the original Politis-Romano scheme."""
    n = len(x)
    valid = np.isfinite(x)
    valid_pos = np.flatnonzero(valid)
    p_new = 1.0 / mean_block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = valid_pos[rng.integers(0, len(valid_pos), size=n_paths)]
    for j in range(1, horizon):
        nxt = (idx[:, j - 1] + 1) % n
        restart = (rng.random(n_paths) < p_new) | ~valid[nxt]
        idx[:, j] = np.where(restart, valid_pos[rng.integers(0, len(valid_pos), size=n_paths)], nxt)
    return x[idx]


def stationary_bootstrap_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    mean_block_hours: float = 24.0,
    n_sims: int = 5000,
    seed: int = 0,
    min_obs: int = 24 * 90,
) -> float:
    x = train_full["f_eff"].to_numpy(dtype=float)  # hourly grid, NaN kept so blocks respect gaps
    if np.isfinite(x).sum() < min_obs:
        return np.nan
    rng = np.random.default_rng(seed)
    paths = stationary_bootstrap_paths(x, n_sims, horizon_hours, mean_block_hours, rng)
    return float(np.quantile(np.clip(paths, 0.0, None).sum(axis=1), alpha))
