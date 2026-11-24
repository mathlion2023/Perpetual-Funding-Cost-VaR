"""Proposal Section 4.12: path-level models for the cash-timing stage.

Each model returns, for one forecast origin, forecasts of the q-quantiles of
the path statistics defined in analysis/cash_timing.py -- D^P (price-only
within-horizon draw), D (draw with funding), and Peak (funding-only draw) --
at every level in `levels`. Signature (multi-output, unlike models/base.py):

    fn(train_full, train_labeled, origin_row, *, horizon_hours, levels, **kw)
        -> {"D_P_q0.95": ..., "D_q0.95": ..., "Peak_q0.95": ..., ...}

Three models:

1. joint_bootstrap_path -- naive benchmark. Stationary bootstrap (Politis &
   Romano, 1994) of contiguous blocks of the joint hourly (return, funding)
   history, so co-movement and short-run timing are preserved without any
   parametric model. Unconditional.

2. direct_quantile_path -- linear quantile regression of each path statistic
   on the HAR feature set (har_quantile.py), supervised on train_labeled.

3. joint_sim_path -- the primary path model. Funding: ARX(1) mean
   (arx_mean.py) + two-regime MS-GARCH(1,1)-t variance (ms_garch.py).
   Hourly returns: constant mean + GARCH(1,1)-t (arch). The two standardised
   innovation series are linked by a constant Student-t copula (the proposal
   specifies a time-varying copula; the constant version is a disclosed
   first-pass simplification). Simulated funding is capped at Deribit's
   +/-0.5% per 8h (i.e. +/-0.0625% per hour); the dampener is NOT re-imposed,
   because the model is fitted to already-dampened funding.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from arch import arch_model
from numba import njit
from scipy import stats
from scipy.optimize import minimize_scalar
from statsmodels.regression.quantile_regression import QuantReg

from .. import cash_timing
from .arx_mean import DEFAULT_EXOG as ARX_EXOG, fit_arx_mean
from .block_bootstrap import stationary_bootstrap_paths
from .caviar import DEFAULT_FEATURES
from .har_quantile import add_har_features
from .ms_garch import fit_ms_garch_t

FUNDING_CAP_PER_HOUR = 0.005 / 8
STATS = ("D_P", "D", "Peak")


def _hourly_returns(df: pd.DataFrame) -> pd.Series:
    return df["mark_price"].pct_change(fill_method=None)


def _quantile_dict(sims: dict[str, np.ndarray], levels) -> dict[str, float]:
    return {f"{k}_q{q}": float(np.quantile(sims[k], q)) for k in STATS for q in levels}


def _paths_to_stats(r: np.ndarray, f: np.ndarray) -> dict[str, np.ndarray]:
    """r, f: (n_sims, H) simulated hourly returns and funding (unscaled)."""
    lp = 1.0 - np.cumprod(1.0 + r, axis=1)       # -(M_{t+s} - M_t) / M_t
    C = np.cumsum(f, axis=1)
    return cash_timing.path_stats(lp, C)


# --- 1. Joint stationary bootstrap -------------------------------------------

def joint_bootstrap_path(train_full, train_labeled, origin_row, *, horizon_hours, levels,
                         mean_block_hours: float = 24.0, n_sims: int = 5000, seed: int = 0,
                         min_obs: int = 24 * 90):
    df = pd.DataFrame({"r": _hourly_returns(train_full), "f": train_full["f_eff"]})
    # Index array on the hourly grid with NaN where either series is
    # unobserved, so blocks never splice across a gap (see
    # block_bootstrap.stationary_bootstrap_paths).
    pos = np.where(df.notna().all(axis=1), np.arange(len(df), dtype=float), np.nan)
    if np.isfinite(pos).sum() < min_obs:
        return None
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_paths(pos, n_sims, horizon_hours, mean_block_hours, rng).astype(np.int64)
    r, f = df["r"].to_numpy()[idx], df["f"].to_numpy()[idx]
    return _quantile_dict(_paths_to_stats(r, f), levels)


# --- 2. Direct quantile regression on the path statistics --------------------

def direct_quantile_path(train_full, train_labeled, origin_row, *, horizon_hours, levels,
                         path_targets: pd.DataFrame, min_obs: int = 60, train_stride_hours: int = 4):
    """`path_targets` holds the forward-looking {stat}_H{h} columns for the
    whole panel (cash_timing.build_path_targets); only rows in train_labeled,
    whose windows end at or before the origin, are used for fitting."""
    feats = list(dict.fromkeys(DEFAULT_FEATURES + ["rv_1h"]))
    aug, har_cols = add_har_features(train_full[feats])
    cols = DEFAULT_FEATURES + har_cols
    x0 = aug.iloc[-1][cols]
    if x0.isna().any():
        return None
    X_all = aug.loc[train_labeled.index, cols]
    out = {}
    for k in STATS:
        y = path_targets.loc[train_labeled.index, f"{k}_H{horizon_hours}"]
        # Targets are overlapping h-hour windows, so adjacent hours are nearly
        # duplicate observations; fitting on every `train_stride_hours`-th hour
        # loses little information and cuts the QuantReg cost several-fold.
        d = pd.concat([y.rename("y"), X_all], axis=1).dropna().iloc[::-1].iloc[::train_stride_hours].iloc[::-1]
        if len(d) < min_obs:
            return None
        X = np.column_stack([np.ones(len(d)), d[cols].to_numpy()])
        xrow = np.concatenate(([1.0], x0.to_numpy(dtype=float)))
        for q in levels:
            try:
                res = QuantReg(d["y"].to_numpy(), X).fit(q=q, max_iter=2000)
            except Exception:
                return None
            out[f"{k}_q{q}"] = max(float(res.params @ xrow), 0.0)  # draws are >= 0 by construction
    return out


# --- 3. Joint simulation: MS-GARCH-t funding x GARCH-t returns x t copula ----

@njit(cache=True)
def _ms_predictive_var(u, w1, a1, b1, w2, a2, b2, p11, p22, nu):
    """One-step predictive variance E[h_t | F_{t-1}] from the Hamilton filter,
    for standardising the funding residuals before copula estimation."""
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
    xi1 = (1 - p22) / (2 - p11 - p22)
    out = np.empty(n)
    for t in range(n):
        pred1 = p11 * xi1 + (1 - p22) * (1 - xi1)
        out[t] = pred1 * h1 + (1 - pred1) * h2
        if math.isnan(u[t]):  # unobserved hour: same predict-only step as ms_garch._ms_filter
            u2 = out[t]
            xi1 = pred1
            h1 = w1 + a1 * u2 + b1 * h1
            h2 = w2 + a2 * u2 + b2 * h2
            continue
        l1 = math.exp(const - 0.5 * math.log(h1) - 0.5 * (nu + 1) * math.log(1 + u[t] * u[t] / h1 / (nu - 2)))
        l2 = math.exp(const - 0.5 * math.log(h2) - 0.5 * (nu + 1) * math.log(1 + u[t] * u[t] / h2 / (nu - 2)))
        lik = pred1 * l1 + (1 - pred1) * l2
        xi1 = pred1 * l1 / lik if lik > 1e-300 else 0.5
        h1 = w1 + a1 * u[t] * u[t] + b1 * h1
        h2 = w2 + a2 * u[t] * u[t] + b2 * h2
    return out


def fit_t_copula(z1: np.ndarray, z2: np.ndarray) -> tuple[float, float]:
    """Bivariate Student-t copula: rho from Kendall's tau (rho = sin(pi*tau/2),
    exact for elliptical copulas), then df by maximising the copula
    log-likelihood of the rank pseudo-observations."""
    n = len(z1)
    u1 = stats.rankdata(z1) / (n + 1)
    u2 = stats.rankdata(z2) / (n + 1)
    tau = stats.kendalltau(z1, z2).statistic
    rho = float(np.clip(np.sin(np.pi * tau / 2), -0.99, 0.99))

    def nll(log_df):
        df = 2.0 + math.exp(log_df)
        x1, x2 = stats.t.ppf(u1, df), stats.t.ppf(u2, df)
        q = (x1 * x1 - 2 * rho * x1 * x2 + x2 * x2) / (1 - rho * rho)
        ll = (math.lgamma((df + 2) / 2) + math.lgamma(df / 2) - 2 * math.lgamma((df + 1) / 2)
              - 0.5 * math.log(1 - rho * rho)
              - (df + 2) / 2 * np.log1p(q / df)
              + (df + 1) / 2 * (np.log1p(x1 * x1 / df) + np.log1p(x2 * x2 / df)))
        return -float(np.sum(ll))

    res = minimize_scalar(nll, bounds=(math.log(0.1), math.log(200.0)), method="bounded")
    return rho, 2.0 + math.exp(res.x)


def joint_sim_path(train_full, train_labeled, origin_row, *, horizon_hours, levels,
                   n_sims: int = 2000, seed: int = 0, min_obs: int = 24 * 90,
                   copula_window_hours: int = 24 * 365, return_scale: float = 100.0):
    # Funding marginal.
    fit = fit_arx_mean(train_full, origin_row, min_obs=min_obs)
    if fit is None:
        return None
    par = fit_ms_garch_t(fit.resid, x0=joint_sim_path._warm.get(horizon_hours))
    if not np.isfinite(par["nll"]) or par["nll"] >= 1e19:
        return None
    joint_sim_path._warm[horizon_hours] = par["x"]
    hvar = _ms_predictive_var(fit.resid, par["w1"], par["a1"], par["b1"], par["w2"], par["a2"], par["b2"],
                              par["p11"], par["p22"], par["nu"])
    # fit.resid is on the hourly grid (NaN where unobserved), with its own
    # hour index -- no positional re-alignment needed (lag-audit, 2026-11-06).
    z_f = pd.Series(fit.resid / np.sqrt(hvar), index=fit.resid_index).dropna()

    # Return marginal.
    r = (_hourly_returns(train_full) * return_scale).dropna()
    if len(r) < min_obs:
        return None
    try:
        g = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off", show_warning=False)
    except Exception:
        return None
    gp = g.params
    mu, w, a, b, nu_r = (float(gp["mu"]), float(gp["omega"]), float(gp["alpha[1]"]), float(gp["beta[1]"]), float(gp["nu"]))
    # Hourly BTC returns are routinely fitted as integrated GARCH (alpha+beta
    # ~= 1.000 on this panel). That is harmless for a <=168-step simulation --
    # variance just does not mean-revert -- so only clearly explosive fits are
    # rejected; persistence is recorded in the output.
    if a + b >= 1.01:
        return None
    z_r = g.std_resid.dropna()
    last_eps = float(g.resid.iloc[-1])
    last_s2 = float(g.conditional_volatility.iloc[-1]) ** 2
    s2_next = w + a * last_eps**2 + b * last_s2

    # Copula on the trailing window of jointly-observed standardised residuals.
    zz = pd.concat([z_f.rename("f"), z_r.rename("r")], axis=1).dropna().iloc[-copula_window_hours:]
    if len(zz) < 500:
        return None
    rho, df_c = fit_t_copula(zz["f"].to_numpy(), zz["r"].to_numpy())

    # Simulate.
    rng = np.random.default_rng(seed)
    H = horizon_hours
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    zn = rng.standard_normal((n_sims, H, 2)) @ L.T
    wchi = rng.chisquare(df_c, size=(n_sims, H, 1)) / df_c
    U = stats.t.cdf(zn / np.sqrt(wchi), df_c)
    nu_f = par["nu"]
    e_f = stats.t.ppf(U[..., 0], nu_f) * math.sqrt((nu_f - 2) / nu_f)
    e_r = stats.t.ppf(U[..., 1], nu_r) * math.sqrt((nu_r - 2) / nu_r)

    p1_next = par["p11"] * par["xi1"] + (1 - par["p22"]) * (1 - par["xi1"])
    in1 = rng.random(n_sims) < p1_next
    h1 = np.full(n_sims, par["h1_next"]); h2 = np.full(n_sims, par["h2_next"])
    s2 = np.full(n_sims, s2_next)
    y_prev = np.full(n_sims, fit.last_y)
    shift = fit.mean_shift()
    cap = FUNDING_CAP_PER_HOUR * fit.scale
    f_sim = np.empty((n_sims, H)); r_sim = np.empty((n_sims, H))
    for j in range(H):
        u = np.sqrt(np.where(in1, h1, h2)) * e_f[:, j]
        y = np.clip(shift + fit.rho * y_prev + u, -cap, cap)
        f_sim[:, j] = y / fit.scale
        h1 = par["w1"] + par["a1"] * u * u + par["b1"] * h1
        h2 = par["w2"] + par["a2"] * u * u + par["b2"] * h2
        switch = rng.random(n_sims) >= np.where(in1, par["p11"], par["p22"])
        in1 = np.where(switch, ~in1, in1)
        y_prev = y
        eps = np.sqrt(s2) * e_r[:, j]
        r_sim[:, j] = (mu + eps) / return_scale
        s2 = w + a * eps * eps + b * s2
    out = _quantile_dict(_paths_to_stats(r_sim, f_sim), levels)
    out["copula_rho"], out["copula_df"], out["return_persistence"] = rho, df_c, a + b
    return out


joint_sim_path._warm = {}
