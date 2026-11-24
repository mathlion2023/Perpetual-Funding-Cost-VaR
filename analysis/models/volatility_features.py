"""Volatility-information features for the Q4 extension.

Tina, 2026-10-07 email: "Does volatility information improve prediction of
future collateral requirements?" -- four information sources, each a single
origin-dated scalar feature, held to the same forecast origins, horizons,
baseline predictors, and downstream collateral-prediction method (reusing
collateral_erosion.py's _hurdle_fit unchanged) so only the feature itself
differs between competing models:

  historical : rv_primary, already a panel column (24h trailing realised
               vol of BTC mark price) -- no new code.
  garch      : GARCH(1,1)-t conditional volatility on hourly BTC log-returns,
               refit walk-forward (expanding window) at each evaluation
               origin.
  har        : Corsi (2009) HAR-RV -- OLS forecast of next-hour rv_1h from
               trailing daily/weekly/monthly rv_1h averages, refit
               walk-forward.
  rough      : rough_vol.estimate_hurst/rough_vol_state's sigma_rough,
               reused as-is -- it is already a generic fractional-kernel
               smoothing of price realised vol (rv_1h), not specific to the
               existing funding-rate model that happens to consume it.

GARCH and HAR are genuinely new here: every existing GARCH/HAR-flavoured
model in this package (arx_garch.py, har_quantile.py, gas.py, ms_garch.py)
forecasts funding-rate VaR, not volatility itself -- the GARCH conditional
variance / HAR-style trailing averages are internal machinery, never
previously exposed as a standalone volatility feature.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from . import rough_vol

HAR_WINDOWS_HOURS = {"d": 24, "w": 168, "m": 720}


def walk_forward_feature(panel: pd.DataFrame, origins: pd.DatetimeIndex,
                          feature_fn: Callable[[pd.DataFrame], float], min_obs: int) -> pd.Series:
    """Expanding-window walk-forward: at each origin, feature_fn sees only
    panel rows <= origin (no look-ahead)."""
    idx = panel.index
    pos_of = {ts: i for i, ts in enumerate(idx)}
    out = {}
    for origin in origins:
        pos = pos_of[origin]
        train = panel.iloc[: pos + 1]
        if len(train) < min_obs:
            out[origin] = np.nan
            continue
        try:
            out[origin] = feature_fn(train)
        except Exception:
            out[origin] = np.nan
    return pd.Series(out)


def _garch_feature_fn(train: pd.DataFrame) -> float:
    """Unlike arx_garch_t_var (models/arx_garch.py), this only reads the
    1-step-ahead conditional variance off the fitted recursion -- it does not
    simulate forward over the holding horizon, so the alpha+beta>=1 guard
    used there (to stop a multi-step simulation exploding under a
    non-stationary variance process) does not apply here and is deliberately
    not replicated: hourly BTC log-returns fit GARCH(1,1) essentially at the
    IGARCH boundary (alpha+beta approx 1) on almost every origin checked
    (near-unit-root volatility persistence is the expected finding for crypto
    at this frequency, not a fitting failure), so requiring alpha+beta<1
    would reject nearly every origin."""
    r = 100.0 * np.log(train["mark_price"]).diff().dropna()
    am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off", show_warning=False)
    omega = float(res.params.get("omega", np.nan))
    a_arch = float(res.params.get("alpha[1]", np.nan))
    b_arch = float(res.params.get("beta[1]", np.nan))
    if not np.isfinite(omega) or omega <= 0 or not (0 <= a_arch <= 1) or not (0 <= b_arch <= 1):
        return np.nan
    fcast = res.forecast(horizon=1, reindex=False)
    var_1step = float(fcast.variance.iloc[-1, 0])
    if not np.isfinite(var_1step) or var_1step <= 0:
        return np.nan
    return float(np.sqrt(var_1step)) / 100.0


def garch_vol_feature(panel: pd.DataFrame, origins: pd.DatetimeIndex, min_obs: int = 24 * 180) -> pd.Series:
    return walk_forward_feature(panel, origins, _garch_feature_fn, min_obs)


def _har_feature_fn(train: pd.DataFrame) -> float:
    rv = train["rv_1h"]
    df = pd.DataFrame({"rv": rv})
    for label, n in HAR_WINDOWS_HOURS.items():
        df[f"rv_{label}"] = rv.rolling(n, min_periods=n).mean()
    df["rv_fwd"] = df["rv"].shift(-1)
    predictor_cols = [f"rv_{label}" for label in HAR_WINDOWS_HOURS]
    reg = df.dropna(subset=predictor_cols + ["rv_fwd"])
    if len(reg) < 200:
        return np.nan
    X = sm.add_constant(reg[predictor_cols])
    y = reg["rv_fwd"]
    ols = sm.OLS(y, X).fit()
    x_last = df[predictor_cols].iloc[-1]
    if x_last.isna().any():
        return np.nan
    x_row = np.concatenate(([1.0], x_last.to_numpy(dtype=float)))
    return float(np.dot(ols.params.to_numpy(), x_row))


def har_vol_feature(panel: pd.DataFrame, origins: pd.DatetimeIndex, min_obs: int = 24 * 180) -> pd.Series:
    return walk_forward_feature(panel, origins, _har_feature_fn, min_obs)


def _rough_feature_fn(train: pd.DataFrame, max_lag: int = 168) -> float:
    """Equivalent to rough_vol.rough_vol_state(log_rv, H, max_lag).iloc[-1],
    computed directly rather than via the full historical sweep -- that
    function evaluates every t in the series, which is wasteful when a
    walk-forward loop only ever needs the single most recent value."""
    log_rv = np.log(train["rv_1h"].replace(0, np.nan))
    H, _ = rough_vol.estimate_hurst(log_rv)
    if not np.isfinite(H):
        H = 0.1
    H = float(np.clip(H, 0.01, 0.49))
    x = log_rv.to_numpy()
    if len(x) <= max_lag:
        return np.nan
    window = x[-max_lag:]
    if np.isnan(window).any():
        return np.nan
    k = np.arange(1, max_lag + 1)
    w = k ** (H - 0.5)
    w = w / w.sum()
    return float(np.dot(w[::-1], window))


def rough_vol_feature(panel: pd.DataFrame, origins: pd.DatetimeIndex, min_obs: int = 24 * 180) -> pd.Series:
    return walk_forward_feature(panel, origins, _rough_feature_fn, min_obs)
