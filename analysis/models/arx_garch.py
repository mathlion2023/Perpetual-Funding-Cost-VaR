"""Section 8.3: ARX-GARCH-t model for the effective funding rate.

Fits directly on f_eff_t (a contemporaneous series -- no target_col leakage
concern), so this model uses train_full rather than train_labeled. Forecasts
are produced by simulating M future f_eff paths from the fitted model and
aggregating each into a cumulative long-side funding cost; the alpha-quantile
of the M simulated costs is the funding-cost VaR forecast.

    y_t = c + rho*y_{t-1} + theta'Z_{t-1} + e_t,   e_t = sqrt(h_t) * z_t,  z_t ~ t_nu (unit variance)
    h_t = omega + a*e_{t-1}^2 + b*h_{t-1}

Lag-audit rewrite (2026-11-06): the model was previously fitted with
`arch.arch_model(..., mean="ARX", lags=1)` on the rows left after dropna().
That made both the AR lag y_{t-1} and Z_{t-1} the previous *observed* row,
and the GARCH recursion treat a gap of g hours as one step. `arch` cannot
take missing values, so the same joint (mean + variance) Student-t MLE is
now computed here on the complete hourly grid: lags are one elapsed hour,
and an unobserved hour contributes no likelihood while the variance
advances by its conditional expectation, h_{t+1} = omega + (a+b)*h_t. On a
gap-free series this is the same likelihood `arch` maximises (checked in
tests/test_time_alignment.py).

Known simplification (documented, not silently assumed): the exogenous
regressors Z_{t-1} are unknown beyond the forecast origin, so they are held
fixed at their origin-hour (standardised) value for the whole simulated
horizon. This matches the treatment used for CAViaR/quantile-regression
one-step recursion elsewhere in this package, and is a standard practical
approximation for short exogenous-driven horizons -- but it means the model
is not capturing any expected *drift* in basis/RV/OFI/etc. over the horizon.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

DEFAULT_EXOG = ["basis", "rv_primary", "d_oi", "spread", "ofi", "volume_shock"]


@njit(cache=True)
def _garch_t_filter(e: np.ndarray, omega: float, a: float, b: float, nu: float, h0: float):
    """Negative log-likelihood of standardized-t GARCH(1,1) innovations `e`
    (NaN = unobserved hour), plus the last innovation and the variance h_T at
    the final row. Returns (nll, e_T, h_T, observed_T)."""
    n = e.shape[0]
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    h = h0
    e_prev = math.nan
    nll = 0.0
    for t in range(n):
        if t > 0:
            if math.isnan(e_prev):
                h = omega + (a + b) * h
            else:
                h = omega + a * e_prev * e_prev + b * h
        if not math.isnan(e[t]):
            nll -= const - 0.5 * math.log(h) - 0.5 * (nu + 1) * math.log(1 + e[t] * e[t] / ((nu - 2) * h))
        e_prev = e[t]
    return nll, e[n - 1], h, not math.isnan(e[n - 1])


def _fit_arx_garch_t(y: np.ndarray, X: np.ndarray):
    """Joint MLE of the ARX mean and GARCH(1,1)-t variance. `y` (n,) and the
    regressor matrix X (n, 1 + 1 + k: const, y_{t-1}, Z_{t-1}) are on the
    hourly grid; rows with any NaN are unobserved hours."""
    obs = np.isfinite(y) & np.isfinite(X).all(axis=1)
    beta0, *_ = np.linalg.lstsq(X[obs], y[obs], rcond=None)
    resid0 = y[obs] - X[obs] @ beta0
    var0 = float(np.var(resid0))
    h0 = var0  # backcast start, as in arch's default
    Xz = np.where(obs[:, None], X, 0.0)
    k_mean = X.shape[1]

    def unpack(p):
        beta = p[:k_mean]
        omega = var0 * math.exp(p[k_mean])
        a, b, nu = p[k_mean + 1], p[k_mean + 2], p[k_mean + 3]
        return beta, omega, a, b, nu

    def nll(p):
        beta, omega, a, b, nu = unpack(p)
        if a + b >= 0.9999:
            return 1e20
        e = np.where(obs, y - Xz @ beta, np.nan)
        val, *_ = _garch_t_filter(e, omega, a, b, nu, h0)
        return val if np.isfinite(val) else 1e20

    p0 = np.concatenate([beta0, [math.log(0.05), 0.05, 0.90, 8.0]])
    bounds = [(None, None)] * k_mean + [(math.log(1e-8), math.log(10.0)), (0.0, 0.9999), (0.0, 0.9999), (2.05, 200.0)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    beta, omega, a, b, nu = unpack(res.x)
    e = np.where(obs, y - Xz @ beta, np.nan)
    val, e_T, h_T, e_T_obs = _garch_t_filter(e, omega, a, b, nu, h0)
    return {"beta": beta, "omega": omega, "a": a, "b": b, "nu": nu, "nll": float(val),
            "e_T": float(e_T), "h_T": float(h_T), "e_T_observed": bool(e_T_obs), "n_obs": int(obs.sum())}


def arx_garch_t_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    exog_cols: list[str] | None = None,
    n_sims: int = 2000,
    scale: float = 1e5,
    seed: int = 0,
    min_obs: int = 24 * 90,
) -> float:
    exog_cols = exog_cols or DEFAULT_EXOG

    # Lags on the complete hourly grid (one elapsed hour), then standardise Z
    # on the observed fitting rows only.
    f = train_full["f_eff"]
    z_lag1 = train_full[exog_cols].shift(1)
    y_grid = (f * scale).to_numpy(dtype=float)
    y_lag1 = (f.shift(1) * scale).to_numpy(dtype=float)
    z_raw = z_lag1.to_numpy(dtype=float)
    obs = np.isfinite(y_grid) & np.isfinite(y_lag1) & np.isfinite(z_raw).all(axis=1)
    if obs.sum() < min_obs:
        return np.nan
    x_mu, x_sd = z_raw[obs].mean(axis=0), z_raw[obs].std(axis=0, ddof=1)
    x_sd[x_sd == 0] = 1.0
    first = int(np.flatnonzero(obs)[0])  # start the filter at the first fitted hour
    X = np.column_stack([np.ones(len(y_grid)), y_lag1, (z_raw - x_mu) / x_sd])[first:]
    y = y_grid[first:]

    if not np.isfinite(y[-1]):
        # Paths start from f at the origin hour; never from an older hour.
        return np.nan

    try:
        fit = _fit_arx_garch_t(y, X)
    except Exception:
        return np.nan

    beta = fit["beta"]
    const, rho, theta = float(beta[0]), float(beta[1]), beta[2:]
    omega, a_arch, b_arch, nu = fit["omega"], fit["a"], fit["b"], max(fit["nu"], 2.1)

    # The unconstrained ARX(1) mean equation can occasionally fit a
    # non-stationary AR root (observed |rho|>1 at full-history scale on real
    # data) -- forward-simulating that explodes geometrically over the
    # horizon into a meaningless VaR forecast. Flag rather than mask: return
    # NaN for this origin instead of silently reporting a runaway number.
    if not (-0.999 < rho < 0.999):
        return np.nan
    if not np.isfinite(a_arch + b_arch) or (a_arch + b_arch) >= 1.0 or not np.isfinite(fit["nll"]):
        return np.nan

    # Variance for the first simulated hour, T+1, from the origin hour's state.
    if fit["e_T_observed"]:
        h_next = omega + a_arch * fit["e_T"] ** 2 + b_arch * fit["h_T"]
    else:
        h_next = omega + (a_arch + b_arch) * fit["h_T"]

    # Origin's own exog (standardised the same way as training), held fixed
    # across the simulated horizon -- see module docstring.
    x_origin_raw = origin_row[exog_cols].to_numpy(dtype=float)
    if np.isnan(x_origin_raw).any():
        return np.nan
    x_origin_std = (x_origin_raw - x_mu) / x_sd

    rng = np.random.default_rng(seed)
    t_scale = np.sqrt((nu - 2) / nu)
    drift = const + float(theta @ x_origin_std)

    sims = np.empty((n_sims, horizon_hours))
    for m in range(n_sims):
        y_prev, h_t = float(y[-1]), h_next
        for j in range(horizon_hours):
            resid_t = np.sqrt(max(h_t, 1e-12)) * rng.standard_t(nu) * t_scale
            y_t = drift + rho * y_prev + resid_t
            sims[m, j] = y_t
            y_prev = y_t
            h_t = omega + a_arch * resid_t**2 + b_arch * h_t

    sims_unscaled = sims / scale
    c_paths = np.clip(sims_unscaled, 0.0, None).sum(axis=1)
    return float(np.quantile(c_paths, alpha))
