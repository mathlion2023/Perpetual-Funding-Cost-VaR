"""Section 4.6.4 (added 2026-09-28): two-regime Markov-switching GARCH(1,1)-t.

    f_t = a + rho*f_{t-1} + theta'Z_{t-1} + u_t
    u_t | S_t = k  ~  sqrt(h_{k,t}) * e_t,   e_t ~ Student-t(nu), unit variance
    h_{k,t} = omega_k + alpha_k * u_{t-1}^2 + beta_k * h_{k,t-1},   k = 1, 2
    P(S_t = j | S_{t-1} = i) = p_ij  (constant 2x2 transition matrix)

Every regime's variance is updated with the *realised* u_{t-1} whatever
regime was active (Haas, Mittnik & Paolella, 2004), which avoids the
path-dependence problem of Gray-type MS-GARCH and makes the likelihood
computable exactly with the Hamilton (1989) filter. This is the
specification Ardia, Bluteau & Rueede (2019) apply to Bitcoin and that the
MSGARCH package implements (Ardia, Bluteau, Boudt, et al., 2019).

Motivation (proposal Section 4.6.4): VaR exceptions cluster in time for
almost every single-regime model in the first full backtest, and the
preliminary cash-timing evidence ties peak/terminal funding gaps to regime
*transitions* rather than to persistent high-funding states.

Documented simplifications:
  - Two-step estimation (OLS ARX mean first, arx_mean.py), as in
    stochastic_vol.py and gas.py, not joint mean+variance MLE.
  - Constant transition probabilities. The proposal allows them to depend on
    lagged open interest and the high-funding indicator; that
    time-varying-transition extension is not implemented in this pass.
  - Regime labels are normalised after estimation so regime 1 is always the
    low-unconditional-variance regime.
  - Warm start: each origin's optimisation starts from the previous origin's
    estimate (a pure function of earlier data, so no look-ahead), which cuts
    runtime several-fold on an expanding window.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

from .arx_mean import fit_arx_mean, simulate_funding_cost

_WARM_START: dict[str, np.ndarray] = {}


@njit(cache=True)
def _ms_filter(u, w1, a1, b1, w2, a2, b2, p11, p22, nu):
    """Hamilton filter for the Haas-Mittnik-Paolella MS-GARCH-t.

    Returns (negative log-likelihood, filtered P(S_T=1 | F_T), h1_{T+1}, h2_{T+1}).

    `u` is on the complete hourly grid; a NaN (unobserved hour) contributes
    no likelihood, the regime probability is propagated by the transition
    matrix only, and both variances are updated with the predictive
    E[u_t^2 | F_{t-1}] = pred1*h1 + pred2*h2 in place of the unseen u_t^2.
    """
    n = u.shape[0]
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    v = 0.0
    n_obs = 0
    for t in range(n):
        if not math.isnan(u[t]):
            v += u[t] * u[t]
            n_obs += 1
    v /= max(n_obs, 1)
    h1, h2 = v, v
    xi1 = (1 - p22) / (2 - p11 - p22)  # ergodic start
    nll = 0.0
    for t in range(n):
        pred1 = p11 * xi1 + (1 - p22) * (1 - xi1)
        pred2 = 1.0 - pred1
        if math.isnan(u[t]):
            u2 = pred1 * h1 + pred2 * h2
            xi1 = pred1
            h1 = w1 + a1 * u2 + b1 * h1
            h2 = w2 + a2 * u2 + b2 * h2
            continue
        z1 = u[t] * u[t] / h1
        z2 = u[t] * u[t] / h2
        l1 = math.exp(const - 0.5 * math.log(h1) - 0.5 * (nu + 1) * math.log(1 + z1 / (nu - 2)))
        l2 = math.exp(const - 0.5 * math.log(h2) - 0.5 * (nu + 1) * math.log(1 + z2 / (nu - 2)))
        lik = pred1 * l1 + pred2 * l2
        if not (lik > 1e-300):
            return 1e20, 0.5, h1, h2
        nll -= math.log(lik)
        xi1 = pred1 * l1 / lik
        h1 = w1 + a1 * u[t] * u[t] + b1 * h1
        h2 = w2 + a2 * u[t] * u[t] + b2 * h2
    return nll, xi1, h1, h2


def _unpack(p):
    return dict(w1=p[0], a1=p[1], b1=p[2], w2=p[3], a2=p[4], b2=p[5], p11=p[6], p22=p[7], nu=p[8])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# Optimisation runs on an unconstrained, similarly-scaled parameter vector:
# log(omega_k / var(u)), logit(persistence_k = alpha_k + beta_k),
# logit(alpha_k / persistence_k), logit(p_kk), log(nu - 2). Optimising the
# natural parameters directly under box bounds stalled at the starting values
# in the simulated-data recovery check, because omega is orders of magnitude
# smaller than the other parameters.
def _to_natural(z: np.ndarray, v: float) -> np.ndarray:
    out = []
    for k in range(2):
        lw, lpers, lshare = z[3 * k : 3 * k + 3]
        pers = 0.9999 * _sigmoid(lpers)
        a = pers * _sigmoid(lshare)
        out += [v * np.exp(lw), a, pers - a]
    out += [0.5 + 0.4999 * _sigmoid(z[6]), 0.5 + 0.4999 * _sigmoid(z[7]), 2.0 + np.exp(z[8])]
    return np.array(out)


def _to_unconstrained(p: np.ndarray, v: float) -> np.ndarray:
    z = []
    for k in range(2):
        w, a, b = p[3 * k : 3 * k + 3]
        pers = a + b
        z += [np.log(max(w, 1e-12) / v), _logit(pers / 0.9999), _logit(a / max(pers, 1e-9))]
    z += [_logit((p[6] - 0.5) / 0.4999), _logit((p[7] - 0.5) / 0.4999), np.log(max(p[8] - 2.0, 1e-3))]
    return np.array(z)


def fit_ms_garch_t(u: np.ndarray, x0: np.ndarray | None = None) -> dict:
    v = float(np.nanvar(u))
    if x0 is None:
        x0 = np.array([0.02 * v, 0.05, 0.90, 0.10 * v, 0.15, 0.80, 0.98, 0.95, 6.0])

    def obj(z):
        nll, *_ = _ms_filter(u, *_to_natural(z, v))
        return nll if np.isfinite(nll) else 1e20

    res = minimize(obj, _to_unconstrained(x0, v), method="BFGS", options={"gtol": 1e-4, "maxiter": 500})
    p = _to_natural(res.x, v)
    nll, xi1, h1n, h2n = _ms_filter(u, *p)
    out = _unpack(p) | {"xi1": xi1, "h1_next": h1n, "h2_next": h2n, "nll": float(nll), "x": p}

    # Label normalisation: regime 1 = lower unconditional variance.
    uv1 = out["w1"] / max(1 - out["a1"] - out["b1"], 1e-6)
    uv2 = out["w2"] / max(1 - out["a2"] - out["b2"], 1e-6)
    if uv1 > uv2:
        out = {**out,
               "w1": out["w2"], "a1": out["a2"], "b1": out["b2"],
               "w2": out["w1"], "a2": out["a1"], "b2": out["b1"],
               "p11": out["p22"], "p22": out["p11"],
               "xi1": 1 - out["xi1"], "h1_next": out["h2_next"], "h2_next": out["h1_next"]}
        out["x"] = np.array([out[k] for k in ["w1", "a1", "b1", "w2", "a2", "b2", "p11", "p22", "nu"]])
    return out


def simulate_ms_shocks(par: dict, n_sims: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    nu = par["nu"]
    t_scale = math.sqrt((nu - 2) / nu)
    # Regime at T+1 drawn from the one-step prediction of the filtered state.
    p1_next = par["p11"] * par["xi1"] + (1 - par["p22"]) * (1 - par["xi1"])
    in1 = rng.random(n_sims) < p1_next
    h1 = np.full(n_sims, par["h1_next"])
    h2 = np.full(n_sims, par["h2_next"])
    shocks = np.empty((n_sims, horizon))
    for j in range(horizon):
        h = np.where(in1, h1, h2)
        u = np.sqrt(h) * rng.standard_t(nu, size=n_sims) * t_scale
        shocks[:, j] = u
        h1 = par["w1"] + par["a1"] * u * u + par["b1"] * h1
        h2 = par["w2"] + par["a2"] * u * u + par["b2"] * h2
        stay = np.where(in1, par["p11"], par["p22"])
        switch = rng.random(n_sims) >= stay
        in1 = np.where(switch, ~in1, in1)
    return shocks


def ms_garch_t_var(
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
    warm_start_key: str | None = None,
) -> float:
    fit = fit_arx_mean(train_full, origin_row, exog_cols=exog_cols, min_obs=min_obs)
    if fit is None:
        return np.nan
    key = warm_start_key or target_col
    par = fit_ms_garch_t(fit.resid, x0=_WARM_START.get(key))
    if not np.isfinite(par["nll"]) or par["nll"] >= 1e19:
        return np.nan
    _WARM_START[key] = par["x"]
    rng = np.random.default_rng(seed)
    shocks = simulate_ms_shocks(par, n_sims, horizon_hours, rng)
    return float(np.quantile(simulate_funding_cost(fit, shocks), alpha))
