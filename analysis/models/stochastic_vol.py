"""Section 8.4: conventional stochastic volatility (SV-t) and SV with jumps (SVJ-t).

    f_t = mu_t + exp(h_t/2) * eps_t,   eps_t ~ t_nu(0,1)
    h_t = omega + phi*(h_{t-1} - omega) + sigma_h * eta_t,   eta_t ~ N(0,1)
    mu_t = a + rho*f_{t-1} + theta'Z_{t-1}

Estimation method (documented simplification, not full MCMC/particle-filter
Bayesian estimation): a two-step quasi-MLE.

  Step 1 -- mean equation: OLS of f_t on [1, f_{t-1}, Z_{t-1}] gives (a, rho,
  theta) and residuals r_t = f_t - mu_t.

  Step 2 -- volatility equation: following Harvey, Ruiz & Shephard (1994),
  x_t = log(r_t^2 + c) is approximately h_t + a constant + noise with known
  first two moments under Gaussian eps_t; a linear Gaussian Kalman filter on
  x_t then gives a quasi-likelihood for (omega, phi, sigma_h), maximised by
  L-BFGS-B. The bias-correction constants used are the exact Gaussian values
  (E[log(chi^2_1)] = -1.2704, Var = pi^2/2); using them under Student-t eps_t
  is an approximation, not exact -- a known limitation of this lightweight
  estimator relative to a full particle-filter/MCMC treatment, which is left
  as future work (this mirrors this project's own documented status for the
  jump-augmented rough-volatility estimator).

SVJ-t extension: after fitting the baseline SV-t, candidate jumps are flagged
where the standardised residual exceeds a robust threshold; jump size is
fitted as Normal(muJ, sigmaJ^2) on the flagged residuals, and jump
probability is fitted via logistic regression on standardised open interest,
HighFunding, and a realised-volatility stress proxy (Brunnermeier & Pedersen,
2009 motivates this state-dependence). This is a two-step
detect-then-estimate procedure, not joint estimation -- again a documented
simplification pending a full particle-filter implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import njit
from scipy.optimize import minimize
from scipy.stats import t as student_t

GAUSSIAN_LOG_CHISQ1_MEAN = -1.2704
GAUSSIAN_LOG_CHISQ1_VAR = (np.pi**2) / 2

# Bounds for the log-variance level h_t (and its long-run mean omega), shared
# between the Kalman optimizer's own search bounds and any later clipping of
# h_t -- these MUST stay identical. A previous mismatch (optimizer bounds
# (-30, 10) vs. a separate hardcoded (-20, 20) clip used only when
# standardizing residuals for the nu fit) silently clipped every fitted omega
# for this project's funding-rate series (consistently ~-26 to -28, i.e.
# always below -20) up to -20, inflating the implied standardization scale
# ~43x and crushing std_resid's variance to ~0.05. That forced student_t.fit
# (which assumes unit scale) to a degenerate boundary nu=60 (near-Gaussian)
# in every single forecast origin, understating tail risk in the pre-specified
# primary SVJ-t proxy. Verified empirically against the real panel: with
# matching bounds and time-varying h_t standardization (see _fit_sv_core),
# nu comes out ~26 instead of saturating at 60.
OMEGA_BOUNDS = (-30.0, 10.0)

DEFAULT_EXOG = ["basis", "rv_primary", "d_oi", "spread", "ofi", "volume_shock"]


@dataclass
class SVFit:
    a: float
    rho: float
    theta: np.ndarray
    exog_cols: list[str]
    omega: float
    phi: float
    sigma_h: float
    nu: float
    last_level: float
    last_f: float
    resid_std: float


def _mean_equation_ols(df: pd.DataFrame, exog_cols: list[str]) -> tuple[np.ndarray, pd.Series]:
    y = df["f_eff"]
    X = sm.add_constant(df[["f_eff_lag1"] + exog_cols])
    ols = sm.OLS(y, X).fit()
    resid = ols.resid
    return ols.params.to_numpy(), resid


@njit(cache=True)
def _kalman_filter_levels(x: np.ndarray, omega: float, phi: float, sigma_h2: float, obs_mean: float, R: float) -> np.ndarray:
    """Same recursion as _kalman_filter_core, but returns the full per-step
    filtered level h_t instead of only the last one. Used once (after the
    optimizer has converged) to time-varying-standardize residuals for the
    nu fit -- standardizing by a single constant (e.g. the long-run mean
    omega) would leave volatility clustering in the "standardized" series
    and contaminate the fitted tail-shape parameter with it.
    """
    n = len(x)
    level = omega
    P = sigma_h2 / (1 - phi * phi) if abs(phi) < 0.999 else sigma_h2 * 100.0
    levels = np.empty(n)
    for i in range(n):
        level_pred = omega + phi * (level - omega)
        P_pred = phi * phi * P + sigma_h2
        S = P_pred + R
        if np.isnan(x[i]):  # unobserved hour: predict-only step (elapsed time, no update)
            level, P = level_pred, P_pred
            levels[i] = level
            continue
        innovation = (x[i] - obs_mean) - level_pred
        K = P_pred / S
        level = level_pred + K * innovation
        P = (1 - K) * P_pred
        levels[i] = level
    return levels


@njit(cache=True)
def _kalman_filter_core(x: np.ndarray, omega: float, phi: float, sigma_h2: float, obs_mean: float, R: float) -> tuple:
    """One Kalman filter pass. Returns (negative log-likelihood, last filtered level).

    Numba-jitted: this loop runs once per candidate parameter vector inside
    the optimizer (dozens of times) for every forecast origin in a rolling
    backtest, over a training sample that can run to tens of thousands of
    hourly observations for a multi-year window -- a pure-Python loop here
    is the single biggest runtime risk in this model.
    """
    n = len(x)
    level = omega
    P = sigma_h2 / (1 - phi * phi) if abs(phi) < 0.999 else sigma_h2 * 100.0
    nll = 0.0
    for i in range(n):
        level_pred = omega + phi * (level - omega)
        P_pred = phi * phi * P + sigma_h2
        S = P_pred + R
        if np.isnan(x[i]):  # unobserved hour: predict-only step, no likelihood contribution
            level, P = level_pred, P_pred
            continue
        innovation = (x[i] - obs_mean) - level_pred
        nll += 0.5 * (np.log(2 * np.pi * S) + innovation * innovation / S)
        K = P_pred / S
        level = level_pred + K * innovation
        P = (1 - K) * P_pred
    return nll, level


def _kalman_loglik(params: np.ndarray, x: np.ndarray) -> float:
    omega, phi, log_sigma_h = params
    sigma_h2 = np.exp(log_sigma_h) ** 2
    if not (-0.999 < phi < 0.999):
        return 1e10
    nll, _ = _kalman_filter_core(x, omega, phi, sigma_h2, GAUSSIAN_LOG_CHISQ1_MEAN, GAUSSIAN_LOG_CHISQ1_VAR)
    return nll


def _fit_kalman_sv(resid: pd.Series, eps: float | None = None) -> tuple[float, float, float, float, np.ndarray]:
    """Returns (omega, phi, sigma_h, last_filtered_level, h_path).

    `eps` guards log(r^2) against an exact-zero residual; it must be far
    below the smallest realistic |residual|, not a fixed absolute constant.
    A previous fixed default of 1e-8 silently swamped this project's
    ~1e-6-scale funding-rate residuals (typical r^2 ~ 1e-13), because for
    most observations r^2 + eps ~= eps regardless of the true residual --
    biasing the fitted volatility level ~74x too high (log-variance off by
    ~8, i.e. exp(8/2)~=55-74x on the sigma scale) and producing wildly
    oversized VaR forecasts (observed empirically: SV-t/SVJ-t had zero
    exceptions and ~100-200x worse quantile loss than every other model
    across the full 5-year backtest). Deriving eps from the data's own scale
    makes this fail safe for any future series, not just this one.
    """
    r = resid.to_numpy()  # on the hourly grid; NaN = unobserved hour (predict-only in the filter)
    if eps is None:
        nonzero_abs = np.abs(r[(r != 0) & np.isfinite(r)])
        smallest_scale = float(nonzero_abs.min()) if len(nonzero_abs) else 1.0
        eps = (smallest_scale * 1e-6) ** 2

    r2 = r**2 + eps
    x = np.log(r2)

    x0 = np.array([np.log(np.nanmean(r2) + eps), 0.9, np.log(0.2)])
    bounds = [OMEGA_BOUNDS, (-0.995, 0.995), (np.log(1e-4), np.log(5.0))]
    res = minimize(_kalman_loglik, x0, args=(x,), method="L-BFGS-B", bounds=bounds)
    omega, phi, log_sigma_h = res.x
    sigma_h = np.exp(log_sigma_h)

    # one more filter pass to recover the full filtered h_t path (last entry
    # is the last_filtered_level used for forecasting)
    sigma_h2 = sigma_h**2
    R = GAUSSIAN_LOG_CHISQ1_VAR
    h_path = _kalman_filter_levels(x, omega, phi, sigma_h2, GAUSSIAN_LOG_CHISQ1_MEAN, R)

    return float(omega), float(phi), float(sigma_h), float(h_path[-1]), h_path


def _fit_sv_core(train_full: pd.DataFrame, exog_cols: list[str], min_obs: int) -> SVFit | None:
    lag_exog_cols = [c + "_lag1" for c in exog_cols]
    df = train_full[["f_eff"]].copy()
    df["f_eff_lag1"] = train_full["f_eff"].shift(1)
    df[lag_exog_cols] = train_full[exog_cols].shift(1)
    df = df.dropna(subset=["f_eff", "f_eff_lag1"] + lag_exog_cols)
    if len(df) < min_obs:
        return None
    if not np.isfinite(train_full["f_eff"].iloc[-1]):
        # Paths start from f at the origin hour; if that hour is missing, the
        # last observed f is from an earlier hour and the H-hour horizon would
        # be misaligned. Fail the origin visibly instead.
        return None

    mean_df = df.rename(columns=dict(zip(lag_exog_cols, exog_cols)))
    ols_params, resid = _mean_equation_ols(mean_df[["f_eff", "f_eff_lag1"] + exog_cols], exog_cols)
    a, rho = ols_params[0], ols_params[1]
    theta = ols_params[2:]

    if resid.std() == 0 or len(resid) < min_obs:
        return None
    if not (-0.999 < rho < 0.999):
        # Non-stationary AR root in the mean equation -- forward simulation
        # would explode geometrically over the horizon (observed in practice
        # with the ARX-GARCH-t sibling model at full-history scale). Fail
        # this origin rather than return a runaway forecast.
        return None

    # Residuals back on the hourly grid, first fitted hour -> origin hour, so the
    # Kalman filter advances h_t through missing hours as elapsed time and its
    # last level is the origin hour's (lag-audit fix, 2026-11-06; the filter
    # previously ran on the observed rows spliced together).
    resid_grid = resid.reindex(train_full.index[train_full.index.get_loc(resid.index[0]):])
    omega, phi, sigma_h, last_level, h_path = _fit_kalman_sv(resid_grid)
    resid = resid_grid

    std_resid = resid.to_numpy() / np.exp(np.clip(h_path, *OMEGA_BOUNDS) / 2)
    std_resid = std_resid[np.isfinite(std_resid)]
    try:
        nu, _, _ = student_t.fit(std_resid, floc=0, fscale=1)
        nu = float(np.clip(nu, 2.5, 60.0))
    except Exception:
        nu = 8.0

    return SVFit(
        a=a,
        rho=rho,
        theta=theta,
        exog_cols=exog_cols,
        omega=omega,
        phi=phi,
        sigma_h=sigma_h,
        nu=nu,
        last_level=last_level,
        last_f=float(train_full["f_eff"].iloc[-1]),  # the origin hour's own f_eff (checked finite above)
        resid_std=float(resid.std()),
    )


def _simulate_paths(
    fit: SVFit,
    x_origin: np.ndarray,
    horizon_hours: int,
    n_sims: int,
    seed: int,
    jump_params: dict | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t_scale = np.sqrt((fit.nu - 2) / fit.nu) if fit.nu > 2 else 1.0

    sims = np.empty((n_sims, horizon_hours))
    for m in range(n_sims):
        level_prev = fit.last_level
        f_prev = fit.last_f
        for j in range(horizon_hours):
            level_t = fit.omega + fit.phi * (level_prev - fit.omega) + fit.sigma_h * rng.standard_normal()
            sigma_t = np.exp(level_t / 2)
            eps = rng.standard_t(fit.nu) * t_scale
            mu_t = fit.a + fit.rho * f_prev + float(fit.theta @ x_origin)
            f_t = mu_t + sigma_t * eps

            if jump_params is not None:
                p_jump = jump_params["p_jump"]
                if rng.random() < p_jump:
                    f_t += rng.normal(jump_params["muJ"], max(jump_params["sigmaJ"], 1e-8))

            sims[m, j] = f_t
            level_prev, f_prev = level_t, f_t
    return sims


def sv_t_var(
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
    min_obs: int = 24 * 120,
) -> float:
    exog_cols = exog_cols or DEFAULT_EXOG
    fit = _fit_sv_core(train_full, exog_cols, min_obs)
    if fit is None:
        return np.nan

    x_origin = origin_row[exog_cols].to_numpy(dtype=float)
    if np.isnan(x_origin).any():
        return np.nan

    sims = _simulate_paths(fit, x_origin, horizon_hours, n_sims, seed)
    c_paths = np.clip(sims, 0.0, None).sum(axis=1)
    return float(np.quantile(c_paths, alpha))


def svj_t_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    exog_cols: list[str] | None = None,
    jump_threshold_mad: float = 4.0,
    n_sims: int = 1000,
    seed: int = 0,
    min_obs: int = 24 * 120,
) -> float:
    exog_cols = exog_cols or DEFAULT_EXOG
    fit = _fit_sv_core(train_full, exog_cols, min_obs)
    if fit is None:
        return np.nan

    extra_cols = [c for c in ["d_oi"] if c not in exog_cols]
    lag_exog_cols = [c + "_lag1" for c in exog_cols]
    df = train_full[["f_eff"] + extra_cols + exog_cols].copy()
    df["f_eff_lag1"] = df["f_eff"].shift(1)
    df[lag_exog_cols] = train_full[exog_cols].shift(1)
    # Complete-case drop, matching _fit_sv_core's treatment of the identical
    # lagged-exogenous regressors -- filling missing lags with 0.0 (as a
    # previous version did) is not a neutral placeholder for these raw,
    # non-demeaned regressors and would inject spurious mu_hat error (hence
    # spurious jump flags) concentrated in this project's documented 2021
    # missing-data gap.
    df = df.dropna(subset=["f_eff", "f_eff_lag1"] + lag_exog_cols)
    mu_hat = fit.a + fit.rho * df["f_eff_lag1"] + (df[lag_exog_cols].to_numpy() @ fit.theta)
    resid = df["f_eff"] - mu_hat
    sigma_hat = np.exp(fit.omega / 2)  # unconditional average vol as a simple scale reference
    z = resid / max(sigma_hat, 1e-12)

    mad = float(np.median(np.abs(z - np.median(z)))) * 1.4826
    thresh = jump_threshold_mad * max(mad, 1e-8)
    jump_mask = np.abs(z) > thresh

    if jump_mask.sum() >= 5:
        muJ = float(resid[jump_mask].mean())
        sigmaJ = float(resid[jump_mask].std())
        oi_std = (df["d_oi"] - df["d_oi"].mean()) / (df["d_oi"].std() + 1e-12)
        stress = (df[exog_cols[1]] - df[exog_cols[1]].mean()) / (df[exog_cols[1]].std() + 1e-12) if len(exog_cols) > 1 else pd.Series(0.0, index=df.index)
        try:
            X = sm.add_constant(pd.DataFrame({"oi_std": oi_std, "stress": stress}))
            logit = sm.Logit(jump_mask.astype(float), X).fit(disp=0)
            # Jump state at the origin hour itself (not the last observed row,
            # which can be older if hours are missing), standardised like the fit.
            oi_now = (origin_row["d_oi"] - df["d_oi"].mean()) / (df["d_oi"].std() + 1e-12)
            st_col = exog_cols[1] if len(exog_cols) > 1 else None
            st_now = ((origin_row[st_col] - df[st_col].mean()) / (df[st_col].std() + 1e-12)) if st_col else 0.0
            x_now = np.array([1.0, float(oi_now), float(st_now)])
            if not np.isfinite(x_now).all():
                raise ValueError("origin jump-state covariates unobserved")
            p_jump = float(1 / (1 + np.exp(-x_now @ logit.params.to_numpy())))
        except Exception:
            p_jump = float(jump_mask.mean())
        jump_params = {"p_jump": np.clip(p_jump, 0.0, 0.5), "muJ": muJ, "sigmaJ": sigmaJ}
    else:
        jump_params = None  # too few candidate jumps to fit a stable jump component

    x_origin = origin_row[exog_cols].to_numpy(dtype=float)
    if np.isnan(x_origin).any():
        return np.nan

    sims = _simulate_paths(fit, x_origin, horizon_hours, n_sims, seed, jump_params=jump_params)
    c_paths = np.clip(sims, 0.0, None).sum(axis=1)
    return float(np.quantile(c_paths, alpha))
