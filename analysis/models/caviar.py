"""Section 8.2: direct conditional-quantile and CAViaR-style funding models.

Both are directly supervised on target_col, so both must fit on
train_labeled only (see base.py's docstring for why).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from statsmodels.regression.quantile_regression import QuantReg

DEFAULT_FEATURES = ["f_eff", "basis", "rv_primary", "d_oi", "spread", "ofi", "volume_shock"]


def check_loss(u: np.ndarray, alpha: float) -> float:
    return float(np.sum(u * (alpha - (u < 0))))


def _design(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X.insert(0, "const", 1.0)
    return X


def direct_quantile_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    feature_cols: list[str] | None = None,
    min_obs: int = 60,
    clip_nonnegative: bool = True,
) -> float:
    """Q_alpha[C_{t->t+H} | X_t] = X_t'beta, estimated by minimising check loss.

    `clip_nonnegative` (peer-review-flagged bug fix, 2026-09-29): the original
    unconditional `max(forecast, 0.0)` floor is only correct when target_col is
    a non-negative-by-construction quantity (gross paid funding, C_long_H{h} --
    the model's original purpose). This function is also reused, via
    stage1_omission.price_only_var_backtest and omission_gap.net_loss_var_backtest,
    to forecast L_price_H{h} and L_net_H{h}, both explicitly *signed* targets
    (funding_cost.py never floors them) -- a genuine negative 95th-percentile
    "loss" is possible and, empirically, ~50% of realised L_price/L_net outcomes
    in this dataset are negative. Flooring those VaR forecasts at zero was
    silently inflating them whenever the true fitted quantile was negative.
    Callers forecasting a signed target MUST pass clip_nonnegative=False; the
    default True preserves the original (correct) behaviour for the gross
    funding-cost target used in Section 8's model registry.
    """
    feature_cols = feature_cols or DEFAULT_FEATURES
    df = train_labeled[[target_col] + feature_cols].dropna()
    if len(df) < min_obs:
        return np.nan

    X = _design(df, feature_cols)
    y = df[target_col]
    try:
        res = QuantReg(y, X).fit(q=alpha, max_iter=2000)
    except Exception:
        return np.nan

    x0 = pd.Series({"const": 1.0, **{c: origin_row[c] for c in feature_cols}})
    if x0.isna().any():
        return np.nan
    forecast = float(np.dot(res.params.values, x0[res.params.index].values))
    return max(forecast, 0.0) if clip_nonnegative else forecast


@njit(cache=True)
def _caviar_recurse(xb: np.ndarray, omega: float, phi: float, q0: float) -> np.ndarray:
    """q_t = omega + phi*q_{t-1} + xb_t. Numba-jitted: this recursion is
    evaluated by the optimizer (Nelder-Mead, up to maxiter x n_restarts
    times) at every forecast origin -- a pure-Python loop here made a full
    5-year expanding-window CAViaR backtest take ~2.7 hours for 541 origins
    (observed in practice) instead of minutes."""
    n = len(xb)
    q = np.empty(n)
    q_prev = q0
    for t in range(n):
        if np.isnan(xb[t]):
            # Unobserved hour (xb on the complete hourly grid): q_t undefined,
            # and the recursion restarts from q0 at the next observed hour
            # rather than treating that hour as adjacent to the last observed one.
            q[t] = np.nan
            q_prev = q0
            continue
        q_prev = omega + phi * q_prev + xb[t]
        q[t] = q_prev
    return q


def caviar_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    feature_cols: list[str] | None = None,
    min_obs: int = 120,
    burn_in: int = 30,
    n_restarts: int = 3,
    seed: int = 0,
    clip_nonnegative: bool = True,
) -> float:
    """CAViaR-style persistence extension: q_t = omega + phi * q_{t-1} + beta'X_t.

    Estimated by minimising the sum of check loss over the labeled training
    sample via direct numerical optimisation (Engle & Manganelli, 2004).

    `clip_nonnegative`: see the identical parameter on `direct_quantile_var`
    above -- only correct for a non-negative target (gross funding cost).
    Not currently called with a signed target anywhere in this pipeline, but
    fixed here too so the same bug cannot resurface if `caviar_var` is ever
    reused for a signed target the way `direct_quantile_var` was.
    """
    feature_cols = feature_cols or DEFAULT_FEATURES
    # Lag-audit fix (2026-11-06). The recursion now runs on train_full's
    # complete hourly grid, from the first hour through the origin hour itself:
    #   - q_{t-1} is the previous *elapsed* hour (it was the previous complete
    #     row after dropna(), however many hours earlier);
    #   - the forecast is q at the origin, computed from q_{origin-1}. The old
    #     code plugged in q at the last *labeled* row, i.e. H hours before the
    #     origin, as if it were q_{origin-1}. Rows between the label cutoff and
    #     the origin have observed X but no realised target, so they advance the
    #     recursion without entering the check loss.
    n_lab = len(train_labeled)  # train_labeled is a prefix of train_full (base.run_rolling_backtest)
    Xraw_full = train_full[feature_cols].to_numpy(dtype=float)
    y_full = train_full[target_col].to_numpy(dtype=float, copy=True)  # copy: masked below; must not write into the panel
    y_full[n_lab:] = np.nan  # not yet realised at the origin -- never used in the loss
    usable = np.isfinite(y_full) & np.isfinite(Xraw_full).all(axis=1)
    usable_pos = np.flatnonzero(usable)
    if len(usable_pos) < max(min_obs, burn_in + 10):
        return np.nan

    Xraw = Xraw_full[usable]
    mu, sd = Xraw.mean(axis=0), Xraw.std(axis=0)
    sd[sd == 0] = 1.0
    X = (Xraw_full - mu) / sd  # NaN rows stay NaN -> recursion restarts there

    y_use = y_full[usable_pos]
    q0 = float(np.quantile(y_use[:burn_in], alpha))
    loss_pos = usable_pos[burn_in:]
    n, k = X.shape

    def recurse(params: np.ndarray) -> np.ndarray:
        omega, phi = params[0], params[1]
        beta = params[2:]
        xb = X @ beta
        return _caviar_recurse(xb, omega, phi, q0)

    def objective(params: np.ndarray) -> float:
        q = recurse(params)
        u = y_full[loss_pos] - q[loss_pos]
        return check_loss(u, alpha) / max(1, len(loss_pos))

    rng = np.random.default_rng(seed)
    best_val, best_params = np.inf, None
    base_init = np.concatenate(([q0 * 0.1, 0.7], np.zeros(k)))
    for i in range(n_restarts):
        jitter = rng.normal(scale=0.05, size=base_init.shape) if i > 0 else 0.0
        x0 = base_init + jitter
        res = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8})
        if res.fun < best_val:
            best_val, best_params = res.fun, res.x

    if best_params is None:
        return np.nan

    forecast = float(recurse(best_params)[-1])  # q at the origin hour
    if not np.isfinite(forecast):  # origin X unobserved
        return np.nan
    return max(forecast, 0.0) if clip_nonnegative else forecast
