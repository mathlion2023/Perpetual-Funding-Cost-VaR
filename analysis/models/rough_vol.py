"""Section 8.5: rough-volatility feature.

Two pieces, matching the methodology's two-tier "Primary" / "Advanced
robustness" split (see the model-role table in the methodology doc):

1. estimate_hurst(): Hurst exponent H from the scaling of log-realised-
   volatility increments across multiple lags (Gatheral, Jaisson &
   Rosenbaum, 2018) -- a diagnostic on whether Deribit BTC funding-relevant
   volatility looks rough (H << 0.5), in line with Takaishi (2020) but
   evaluated cautiously per Pontiggia (2025)'s multifractality critique
   already flagged in the methodology doc.

2. direct_rough_vol_var(): the "Primary" direct funding-rate model,
   f_t = a + rho*f_{t-1} + theta'Z_{t-1} + delta*sigma_hat_rough_t + u_t,
   where sigma_hat_rough_t is a fractional (power-law-weighted) smoothing of
   past log-realised-volatility using the fitted H -- a fast, OLS-estimable
   proxy for "the rough-volatility state" that is a genuinely distinct
   predictor from the plain rolling-window RV already in Z_t (not merely
   re-including RV under a new name). Forecasting uses a residual bootstrap
   (resampling the fitted OLS residuals with replacement) rather than an
   assumed parametric innovation distribution.

The "Advanced robustness" jump-augmented / structural rough-price-to-funding
simulation (Bennedsen, Lunde & Pakkanen, 2017 hybrid scheme, funding
cap/clamp applied) is NOT implemented here -- per this project's own
Response-to-Reviewers status for Item 5, that estimator is "named and
scheduled for simulated-data validation, but not yet built or tested."
rough_vol_jump_structural_var() below is a stub that raises
NotImplementedError rather than silently returning a low-quality result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

DEFAULT_EXOG = ["basis", "d_oi", "spread", "ofi", "volume_shock"]


def estimate_hurst(log_rv: pd.Series, lags: range = range(1, 21)) -> tuple[float, pd.DataFrame]:
    """H from the slope of log(E|log RV_t - log RV_{t-lag}|) on log(lag), q=1.

    Returns (H, diagnostic_table) where diagnostic_table has one row per lag
    with the estimated first absolute moment, for inspecting fit quality
    (Pontiggia, 2025's critique is precisely that a single H may fit poorly
    across the lag range -- this table lets a caller check that directly).
    """
    # Increments over `lag` *elapsed hours*: log_rv must be on the complete
    # hourly grid, and NaN is dropped only after differencing (lag-audit fix,
    # 2026-11-06). Dropping first made x.shift(lag) the lag-th previous
    # *observed* hour, so any increment spanning a gap was measured over a
    # longer, unknown time span.
    from .base import assert_hourly_grid

    assert_hourly_grid(log_rv.index, "estimate_hurst input")
    x = log_rv
    rows = []
    for lag in lags:
        diffs = (x - x.shift(lag)).dropna()
        if len(diffs) < 30:
            continue
        rows.append({"lag": lag, "m1": diffs.abs().mean()})
    diag = pd.DataFrame(rows)
    if len(diag) < 5:
        return float("nan"), diag

    X = sm.add_constant(np.log(diag["lag"]))
    y = np.log(diag["m1"])
    ols = sm.OLS(y, X).fit()
    H = float(ols.params.iloc[1])
    diag["fitted_log_m1"] = ols.fittedvalues.to_numpy()
    return H, diag


def rough_vol_state(log_rv: pd.Series, H: float, max_lag: int = 168) -> pd.Series:
    """Power-law-weighted (fractional-kernel) smoothing of past log-RV.

    Weights ~ k^(H - 0.5) for k = 1..max_lag, normalised to sum to 1 -- the
    kernel exponent that represents a fractional-Brownian-motion-consistent
    long-memory average (Gatheral et al., 2018). H < 0.5 => weights decay
    faster than a simple average (short effective memory, consistent with
    "rough" dynamics); H close to 0.5 => close to a flat/simple average.
    """
    if not np.isfinite(H):
        H = 0.1  # literature-typical BTC estimate as a fallback (Takaishi, 2020)
    H = float(np.clip(H, 0.01, 0.49))
    k = np.arange(1, max_lag + 1)
    w = k ** (H - 0.5)
    w = w / w.sum()

    x = log_rv.to_numpy()
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(max_lag, n):
        window = x[t - max_lag : t]  # lags 1..max_lag relative to t (excludes x[t] itself)
        if np.isnan(window).any():
            continue
        out[t] = float(np.dot(w[::-1], window))
    return pd.Series(out, index=log_rv.index)


def rough_vol_state_next(log_rv: pd.Series, H: float, max_lag: int = 168) -> float:
    """sigma_rough for the hour *after* the last row of `log_rv`, i.e. built
    from lags 1..max_lag ending at the last row inclusive -- the value that
    enters the regression for f_{t+1} when the last row is the origin t."""
    ext = pd.concat([log_rv.iloc[-max_lag:], pd.Series([np.nan])], ignore_index=True)
    return float(rough_vol_state(ext, H, max_lag=max_lag).iloc[-1])


def _fit_direct_rough_vol(train_full: pd.DataFrame, exog_cols: list[str], hurst_lags: range, kernel_max_lag: int, min_obs: int):
    # rv_1h (raw per-hour realised vol), NOT rv_primary (a 24h rolling
    # aggregate): Hurst estimation needs the raw high-frequency proxy --
    # fitting it to an already-smoothed rolling window inflates H spuriously
    # (empirically H~0.86 on rv_primary vs H~0.19 on rv_1h, and only the
    # latter is consistent with Gatheral et al. 2018 / Takaishi 2020).
    df = train_full[["f_eff", "rv_1h"] + exog_cols].copy()
    log_rv = np.log(df["rv_1h"].replace(0, np.nan))
    H, hurst_diag = estimate_hurst(log_rv, lags=hurst_lags)
    df["sigma_rough"] = rough_vol_state(log_rv, H, max_lag=kernel_max_lag)

    lag_cols = [c + "_lag1" for c in exog_cols]
    df["f_eff_lag1"] = df["f_eff"].shift(1)
    df[lag_cols] = df[exog_cols].shift(1)
    reg_df = df.dropna(subset=["f_eff", "f_eff_lag1", "sigma_rough"] + lag_cols)
    if len(reg_df) < min_obs:
        return None

    X = sm.add_constant(reg_df[["f_eff_lag1"] + lag_cols + ["sigma_rough"]])
    y = reg_df["f_eff"]
    ols = sm.OLS(y, X).fit()

    rho = float(ols.params.get("f_eff_lag1", 0.0))
    if not (-0.999 < rho < 0.999):
        # Non-stationary AR root -- see the identical guard in
        # stochastic_vol.py / arx_garch.py; the bootstrap-residual forward
        # simulation below would explode geometrically over the horizon.
        return None

    return ols, reg_df, H, hurst_diag, df


def direct_rough_vol_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    exog_cols: list[str] | None = None,
    hurst_lags: range = range(1, 21),
    kernel_max_lag: int = 168,
    n_sims: int = 1000,
    seed: int = 0,
    min_obs: int = 24 * 150,  # needs enough history for both the Hurst fit and the kernel warm-up
) -> float:
    exog_cols = exog_cols or DEFAULT_EXOG
    fitted = _fit_direct_rough_vol(train_full, exog_cols, hurst_lags, kernel_max_lag, min_obs)
    if fitted is None:
        return np.nan
    ols, reg_df, H, hurst_diag, df = fitted

    x_origin = origin_row[exog_cols].to_numpy(dtype=float)
    # Alignment fix (2026-11-06): f_{t+1} is regressed on sigma_rough_{t+1},
    # which uses log-RV through hour t (the origin). The old code used
    # df["sigma_rough"].iloc[-1] = sigma_rough_t (log-RV only through t-1), one
    # hour stale relative to f_eff_t and Z_t, which are both at the origin.
    log_rv = np.log(train_full["rv_1h"].replace(0, np.nan))
    sigma_rough_origin = rough_vol_state_next(log_rv, H, max_lag=kernel_max_lag)
    f_origin = float(train_full["f_eff"].iloc[-1])
    if np.isnan(x_origin).any() or not np.isfinite(sigma_rough_origin) or not np.isfinite(f_origin):
        # f at the origin hour itself must be observed: the last observed f
        # could be hours older, which would misalign the H-hour horizon.
        return np.nan

    resid = ols.resid.to_numpy()
    rng = np.random.default_rng(seed)
    params = ols.params  # index: const, f_eff_lag1, <exog>_lag1..., sigma_rough

    n_sims_eff = n_sims
    sims = np.empty((n_sims_eff, horizon_hours))
    for m in range(n_sims_eff):
        f_prev = f_origin
        boot_resid = rng.choice(resid, size=horizon_hours, replace=True)
        for j in range(horizon_hours):
            row = np.concatenate(([1.0], [f_prev], x_origin, [sigma_rough_origin]))
            f_t = float(np.dot(params.to_numpy(), row))
            f_t += boot_resid[j]
            sims[m, j] = f_t
            f_prev = f_t

    c_paths = np.clip(sims, 0.0, None).sum(axis=1)
    return float(np.quantile(c_paths, alpha))


def rough_vol_jump_structural_var(*args, **kwargs) -> float:
    """Advanced-robustness model: jump-augmented rough volatility with a
    structural price-to-funding simulation applying the exchange's funding
    cap/clamp (Bennedsen, Lunde & Pakkanen, 2017 hybrid scheme for the
    fractional-volatility simulation).

    Deliberately NOT implemented in this pass -- per
    Topic1_Response_to_Reviewers.md Item 5, this estimator is scheduled for
    simulated-data validation before being applied to real data, and is not
    yet built. Raising here (rather than silently approximating it) matches
    that documented status.
    """
    raise NotImplementedError(
        "rough_vol_jump_structural_var is not yet implemented -- see Item 5 in "
        "Topic1_Response_to_Reviewers.md. Validate the joint jump/rough-vol/"
        "cap-clamp estimator on simulated data with known parameters first."
    )
