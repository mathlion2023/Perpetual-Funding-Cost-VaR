"""Shared ARX(1) mean equation for the two-step volatility models added
2026-09-28 (gas.py, ms_garch.py).

    f_t = a + rho*f_{t-1} + theta'Z_{t-1} + u_t

estimated by OLS on the scaled effective funding rate, with Z standardised
on the training window. f_{t-1} and Z_{t-1} are the values one elapsed hour
earlier (complete hourly grid), and the residual series is returned on that
grid with NaN for unobserved hours -- the same mean equation as arx_garch.py, and the
same two-step (mean first, volatility on residuals) structure as
stochastic_vol.py. Forward simulation holds Z fixed at the origin's own
(standardised) value, the documented simplification shared with
arx_garch.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_EXOG = ["basis", "rv_primary", "d_oi", "spread", "ofi", "volume_shock"]


@dataclass
class ARXFit:
    const: float
    rho: float
    theta: np.ndarray
    resid: np.ndarray       # scaled residuals u_t on the complete hourly grid, NaN where unobserved
    resid_index: pd.DatetimeIndex  # the hour of each entry of `resid`; resid[-1] is the origin hour
    last_y: float           # scaled f_eff at the origin
    x_origin_std: np.ndarray
    scale: float

    def mean_shift(self) -> float:
        """Constant part of the one-step mean at the origin: a + theta'Z_origin."""
        return self.const + float(self.theta @ self.x_origin_std)


def fit_arx_mean(
    train_full: pd.DataFrame,
    origin_row: pd.Series,
    exog_cols: list[str] | None = None,
    scale: float = 1e5,
    min_obs: int = 24 * 90,
) -> ARXFit | None:
    exog_cols = exog_cols or DEFAULT_EXOG
    # Lags are taken on train_full's complete hourly grid *before* dropping
    # incomplete rows (lag-audit fix, 2026-11-06). Previously dropna() ran
    # first, so after a missing hour f_lag1/Z_lag1 were the last *observed*
    # hour, however long ago.
    grid = pd.concat([train_full["f_eff"], train_full["f_eff"].shift(1).rename("f_lag1"),
                      train_full[exog_cols].shift(1)], axis=1)
    data = grid.dropna()
    if len(data) < min_obs:
        return None

    x_raw = data[exog_cols]
    x_mu, x_sd = x_raw.mean(), x_raw.std().replace(0, 1.0)
    X = np.column_stack([np.ones(len(data)), data["f_lag1"].to_numpy() * scale, ((x_raw - x_mu) / x_sd).to_numpy()])
    y = data["f_eff"].to_numpy() * scale
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rho = float(beta[1])
    if not (-0.999 < rho < 0.999):
        return None  # non-stationary AR root: same guard as arx_garch.py / rough_vol.py

    x_origin_raw = origin_row[exog_cols].to_numpy(dtype=float)
    if np.isnan(x_origin_raw).any() or not np.isfinite(origin_row["f_eff"]):
        return None
    # Residuals placed back on the hourly grid from the first fitted hour to the
    # origin, so the variance filters (gas.py, ms_garch.py) step through missing
    # hours as elapsed time instead of splicing the observed hours together.
    resid_grid = pd.Series(np.nan, index=train_full.index[train_full.index.get_loc(data.index[0]):])
    resid_grid.loc[data.index] = y - X @ beta
    return ARXFit(
        const=float(beta[0]),
        rho=rho,
        theta=beta[2:].astype(float),
        resid=resid_grid.to_numpy(dtype=float),
        resid_index=resid_grid.index,
        last_y=float(origin_row["f_eff"]) * scale,
        x_origin_std=(x_origin_raw - x_mu.to_numpy()) / x_sd.to_numpy(),
        scale=scale,
    )


def simulate_funding_cost(fit: ARXFit, shocks: np.ndarray) -> np.ndarray:
    """Roll the ARX mean forward with given (n_sims, H) scaled shocks and
    return each path's cumulative long-side funding cost sum(max(f, 0)) in
    unscaled units -- the C_long_H{h} target built in funding_cost.py."""
    n_sims, horizon = shocks.shape
    y_prev = np.full(n_sims, fit.last_y)
    shift = fit.mean_shift()
    cost = np.zeros(n_sims)
    for j in range(horizon):
        y_t = shift + fit.rho * y_prev + shocks[:, j]
        cost += np.clip(y_t, 0.0, None)
        y_prev = y_t
    return cost / fit.scale
