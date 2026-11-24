"""Section 4.6.4 (added 2026-09-28): score-driven GAS-t model.

    f_t = a + rho*f_{t-1} + theta'Z_{t-1} + u_t,     u_t = exp(lam_t/2) * e_t
    e_t ~ Student-t(nu), standardised to unit variance
    lam_{t+1} = omega + a_s * s_t + b * lam_t
    s_t = w_t * u_t^2 / exp(lam_t) - 1,   w_t = (nu + 1) / (nu - 2 + u_t^2 / exp(lam_t))

The log-variance lam_t is driven by the score of the Student-t density
(Creal, Koopman & Lucas, 2013; this unit-scaled form is Harvey's
Beta-t-EGARCH). The weight w_t shrinks towards zero for extreme u_t, so a
single outlier moves the variance far less than in GARCH -- the property
that motivates GAS-t as the stable replacement candidate for ARX-GARCH-t.

Estimation is two-step, like stochastic_vol.py: OLS for the ARX mean
(arx_mean.py), then exact MLE of (omega, a_s, b, nu) on the residuals via a
numba-jitted filter and L-BFGS-B. The recursion is fully deterministic given
the data, so forecasts are reproducible run to run (fixed simulation seed).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

from .arx_mean import fit_arx_mean, simulate_funding_cost

LAM_BOUNDS = (-30.0, 30.0)


@njit(cache=True)
def _gas_filter(u: np.ndarray, omega: float, a_s: float, b: float, nu: float, lam0: float):
    """Returns (negative log-likelihood, lam_{T+1}) for the GAS-t recursion.

    `u` is on the complete hourly grid; a NaN (unobserved hour) contributes
    no likelihood and advances lam by its conditional expectation (the score
    has mean zero), so a gap of g hours is g elapsed steps, not zero.
    """
    n = u.shape[0]
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    lam = lam0
    nll = 0.0
    for t in range(n):
        lam = min(max(lam, LAM_BOUNDS[0]), LAM_BOUNDS[1])
        if math.isnan(u[t]):
            lam = omega + b * lam
            continue
        sig2 = math.exp(lam)
        z2 = u[t] * u[t] / sig2
        nll -= const - 0.5 * lam - 0.5 * (nu + 1) * math.log(1 + z2 / (nu - 2))
        w = (nu + 1) / (nu - 2 + z2)
        s = w * z2 - 1.0
        lam = omega + a_s * s + b * lam
    return nll, lam


def fit_gas_t(u: np.ndarray, x0: np.ndarray | None = None):
    var_u = float(np.nanvar(u))
    lam0 = math.log(max(var_u, 1e-12))
    if x0 is None:
        x0 = np.array([lam0 * (1 - 0.97), 0.05, 0.97, 6.0])

    def obj(p):
        nll, _ = _gas_filter(u, p[0], p[1], p[2], p[3], lam0)
        return nll if np.isfinite(nll) else 1e20

    bounds = [(-10.0, 10.0), (1e-4, 2.0), (0.0, 0.9999), (2.1, 100.0)]
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds)
    omega, a_s, b, nu = res.x
    _, lam_next = _gas_filter(u, omega, a_s, b, nu, lam0)
    return {"omega": omega, "a_s": a_s, "b": b, "nu": nu, "lam_next": lam_next, "lam0": lam0,
            "converged": bool(res.success), "nll": float(res.fun)}


def simulate_gas_shocks(params: dict, n_sims: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    omega, a_s, b, nu = params["omega"], params["a_s"], params["b"], params["nu"]
    t_scale = math.sqrt((nu - 2) / nu)
    lam = np.full(n_sims, params["lam_next"])
    shocks = np.empty((n_sims, horizon))
    for j in range(horizon):
        lam = np.clip(lam, *LAM_BOUNDS)
        sig2 = np.exp(lam)
        u = np.sqrt(sig2) * rng.standard_t(nu, size=n_sims) * t_scale
        shocks[:, j] = u
        z2 = u * u / sig2
        s = (nu + 1) / (nu - 2 + z2) * z2 - 1.0
        lam = omega + a_s * s + b * lam
    return shocks


def gas_t_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    exog_cols: list[str] | None = None,
    n_sims: int = 1000,
    seed: int = 0,
    min_obs: int = 24 * 90,
) -> float:
    fit = fit_arx_mean(train_full, origin_row, exog_cols=exog_cols, min_obs=min_obs)
    if fit is None:
        return np.nan
    params = fit_gas_t(fit.resid)
    # L-BFGS-B often reports a line-search "failure" at a perfectly good
    # optimum, so `converged` is recorded but not used as a filter; a
    # non-finite likelihood or a persistence estimate pinned at its upper
    # bound (an integrated, non-stationary variance) is flagged instead.
    if not np.isfinite(params["nll"]) or params["b"] >= 0.9999:
        return np.nan
    rng = np.random.default_rng(seed)
    shocks = simulate_gas_shocks(params, n_sims, horizon_hours, rng)
    return float(np.quantile(simulate_funding_cost(fit, shocks), alpha))
