"""Section 9: the funding-risk omission gap.

Gap_t = VaR_net_t - VaR_price_t, where both are direct conditional-quantile
(Section 8.2) forecasts of L_net_H and L_price_H respectively -- this
preserves the realised empirical dependence between price P&L and funding
cost, since both targets are estimated directly rather than combined from
separately-fit marginals. The regime test is Gap_t = alpha + delta*HighFunding_t
+ gamma'X_t + u_t, prediction delta > 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config
from .models import base, caviar, expected_shortfall
from .stage1_omission import DEFAULT_CONTROLS

# Section 9 (this module) keeps spread_binc; Section 7 (stage1_omission.py)
# does not -- see the comment on stage1_omission.DEFAULT_CONTROLS for why
# (Binance's 2022-03+ coverage window left Section 7's rare-event logit
# unidentifiable on the full sample). Section 9's regressions tolerate the
# shorter window, so spread_binc stays in here.
SECTION9_CONTROLS = DEFAULT_CONTROLS + ["spread_binc"]


def net_loss_var_backtest(
    panel: pd.DataFrame,
    horizon_hours: int,
    alpha: float = config.VAR_ALPHA,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    target_col = f"L_net_H{horizon_hours}"
    feature_cols = feature_cols or SECTION9_CONTROLS
    # clip_nonnegative=False (bug fix, 2026-09-29): L_net_H{h} is signed
    # (-R^(h)_t + C_signed_H{h}); see caviar.direct_quantile_var's docstring.
    return base.run_rolling_backtest(
        panel,
        caviar.direct_quantile_var,
        target_col,
        horizon_hours,
        alpha=alpha,
        feature_cols=feature_cols,
        clip_nonnegative=False,
        **kwargs,
    )


def build_gap_dataset(
    panel: pd.DataFrame,
    price_var_result: pd.DataFrame,
    net_var_result: pd.DataFrame,
    control_cols: list[str] | None = None,
) -> pd.DataFrame:
    control_cols = control_cols or SECTION9_CONTROLS
    idx = price_var_result.index.intersection(net_var_result.index)
    df = pd.DataFrame(index=idx)
    df["VaR_price"] = price_var_result.loc[idx, "var_forecast"]
    df["VaR_net"] = net_var_result.loc[idx, "var_forecast"]
    df["gap"] = df["VaR_net"] - df["VaR_price"]
    df["high_funding"] = panel.loc[idx, "high_funding"]
    for c in control_cols:
        df[c] = panel.loc[idx, c]
    return df.dropna().sort_index()


def paired_historical_gap_backtest(
    panel: pd.DataFrame,
    horizon_hours: int,
    alpha: float = config.VAR_ALPHA,
    min_train_hours: int = config.MIN_TRAIN_HOURS,
    min_obs: int = 60,
    origins: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Bound-consistent alternative to `build_gap_dataset` (referee report,
    item 1, 2026-11-24). `build_gap_dataset`'s Gap_t = VaR_net_t - VaR_price_t
    combines two INDEPENDENTLY fitted quantile regressions (net_loss_var_backtest
    and stage1_omission.price_only_var_backtest); nothing ties their fitted
    quantiles to the pathwise identity L_net = L_price + C_signed, so Corollary
    1's bound on the true gap need not transfer to this empirical construction
    -- which is exactly why the reported 99th-percentile gap (14.3%/27.3%)
    exceeded the deterministic bound (4.5%/10.5%).

    Here VaR_price and VaR_net are both the alpha-quantile of the REALISED
    L_price_H{h}/L_net_H{h} values over the identical expanding training
    sample, restricted to rows where both are observed. Every row feeding
    either quantile satisfies L_net = L_price + C_signed exactly (it is the
    same realised panel identity), so the Corollary 1 monotonicity argument --
    applied to this empirical distribution instead of two separate model
    fits -- guarantees Gap_t = VaR_net_t - VaR_price_t in [-b_H, b_H] at every
    origin, where b_H = horizon_hours * kappa_bar / 8.
    """
    price_col, net_col = f"L_price_H{horizon_hours}", f"L_net_H{horizon_hours}"
    base.assert_hourly_grid(panel.index)
    if origins is None:
        origins = base.non_overlapping_origins(panel.index, horizon_hours, min_train_hours)
    pos_of = {ts: i for i, ts in enumerate(panel.index)}
    rows = []
    for origin in origins:
        pos = pos_of[origin]
        label_cutoff = pos - horizon_hours  # same cutoff convention as models.base.run_rolling_backtest
        train_labeled = panel.iloc[0 : max(0, label_cutoff + 1)]
        d = train_labeled[[price_col, net_col]].dropna()
        if len(d) < min_obs:
            rows.append({"origin": origin, "VaR_price": np.nan, "VaR_net": np.nan, "n_train": len(d)})
            continue
        rows.append({
            "origin": origin,
            "VaR_price": float(np.quantile(d[price_col], alpha)),
            "VaR_net": float(np.quantile(d[net_col], alpha)),
            "n_train": len(d),
        })
    out = pd.DataFrame(rows).set_index("origin")
    out["L_price_realized"] = panel.loc[out.index, price_col]
    out["L_net_realized"] = panel.loc[out.index, net_col]
    out["gap"] = out["VaR_net"] - out["VaR_price"]
    return out.dropna(subset=["VaR_price", "VaR_net"]).sort_index()


def price_es_backtest(
    panel: pd.DataFrame,
    horizon_hours: int,
    alpha: float = config.VAR_ALPHA,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Section 11 item 5: ES_price, mirroring stage1_omission.price_only_var_backtest
    (same target, same default feature set) but with the ES model_fn."""
    from .stage1_omission import BASELINE_PRICE_FEATURES

    target_col = f"L_price_H{horizon_hours}"
    feature_cols = feature_cols or BASELINE_PRICE_FEATURES
    # clip_nonnegative=False (bug fix, 2026-09-29): L_price_H{h} is signed.
    return base.run_rolling_backtest(
        panel,
        expected_shortfall.direct_quantile_es,
        target_col,
        horizon_hours,
        alpha=alpha,
        feature_cols=feature_cols,
        clip_nonnegative=False,
        **kwargs,
    )


def net_es_backtest(
    panel: pd.DataFrame,
    horizon_hours: int,
    alpha: float = config.VAR_ALPHA,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Section 11 item 5: ES_net, mirroring net_loss_var_backtest above (same
    target, same default feature set) but with the ES model_fn."""
    target_col = f"L_net_H{horizon_hours}"
    feature_cols = feature_cols or SECTION9_CONTROLS
    # clip_nonnegative=False (bug fix, 2026-09-29): L_net_H{h} is signed.
    return base.run_rolling_backtest(
        panel,
        expected_shortfall.direct_quantile_es,
        target_col,
        horizon_hours,
        alpha=alpha,
        feature_cols=feature_cols,
        clip_nonnegative=False,
        **kwargs,
    )


def build_es_gap_dataset(
    panel: pd.DataFrame,
    price_es_result: pd.DataFrame,
    net_es_result: pd.DataFrame,
) -> pd.DataFrame:
    """ESGap_t = ES_net_t - ES_price_t, the ES analogue of `build_gap_dataset`."""
    idx = price_es_result.index.intersection(net_es_result.index)
    df = pd.DataFrame(index=idx)
    df["ES_price"] = price_es_result.loc[idx, "var_forecast"]
    df["ES_net"] = net_es_result.loc[idx, "var_forecast"]
    df["es_gap"] = df["ES_net"] - df["ES_price"]
    return df.dropna().sort_index()


SUMMARY_PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _percentile_summary(series_by_label: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for label, s in series_by_label.items():
        row = {"measure": label, "n": int(s.notna().sum()), "mean": float(s.mean())}
        for q in SUMMARY_PERCENTILES:
            row[f"p{int(q * 100)}"] = float(s.quantile(q))
        rows.append(row)
    return pd.DataFrame(rows).set_index("measure")


def gap_and_rur_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Section 11 item 3: descriptive summary of the continuous omission gap.

    Gap_t = VaR_net_t - VaR_price_t (already a column of `build_gap_dataset`'s
    output) and the proportional risk-understatement ratio
    RUR_t = Gap_t / VaR_price_t, summarised as mean, median, 75th/90th/95th/99th
    percentiles. This is a pure descriptive pass over the existing gap
    dataset -- no new estimation, just the summary Tina asked for as a
    standalone table before the market-conditions regression (item 4).
    """
    out = df.copy()
    out["rur"] = out["gap"] / out["VaR_price"]
    return _percentile_summary({"gap": out["gap"], "rur": out["rur"]})


def es_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Section 11 item 5: descriptive summary of the ES gap, the same
    mean/median/p75/p90/p95/p99 treatment as `gap_and_rur_summary`, applied
    to ESGap_t = ES_net_t - ES_price_t and its proportional counterpart
    ES_RUR_t = ESGap_t / ES_price_t.
    """
    out = df.copy()
    out["es_rur"] = out["es_gap"] / out["ES_price"]
    return _percentile_summary({"es_gap": out["es_gap"], "es_rur": out["es_rur"]})


def fit_gap_regime_test(df: pd.DataFrame, control_cols: list[str] | None = None):
    control_cols = control_cols or SECTION9_CONTROLS
    X = sm.add_constant(df[["high_funding"] + control_cols])
    y = df["gap"]
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    delta = float(ols.params["high_funding"])
    p_two_sided = float(ols.pvalues["high_funding"])
    p_one_sided = p_two_sided / 2 if delta > 0 else 1 - p_two_sided / 2
    return {
        "fit": ols,
        "delta": delta,
        "delta_pvalue_one_sided_H0_delta_le_0": p_one_sided,
    }


MARKET_CONDITION_CONTROLS = SECTION9_CONTROLS  # rv_primary, basis, d_oi, spread, ofi, volume_shock, spread_binc
# Tina's naming <-> panel column names, for report-facing output only.
MARKET_CONDITION_LABELS = {
    "FundingLevel": "FundingLevel",
    "FundingVolatility": "FundingVolatility",
    "rv_primary": "RV",
    "basis": "Basis",
    "d_oi": "DeltaOI",
    "spread": "SpreadDeribit",
    "spread_binc": "SpreadBinance",
    "ofi": "OrderFlow",
    "volume_shock": "VolumeShock",
}


def _funding_level_and_volatility(
    panel: pd.DataFrame, lookback: int = config.HIGH_FUNDING_LOOKBACK_HOURS
) -> pd.DataFrame:
    """FundingLevel_t and FundingVolatility_t for item 4's market-conditions
    gap regression -- not part of the Section 6 X_t forecast set (that
    vector is frozen per the methodology), so kept local to this analysis
    rather than added to forecast_set.py's PREDICTOR_COLUMNS.

    FundingLevel_t reuses forecast_set.py's `f_bar_24h` (trailing `lookback`
    -hour mean of f_eff, causal by construction) when already present;
    FundingVolatility_t is the trailing `lookback`-hour standard deviation of
    f_eff over the same causal window (rolling, so it only uses information
    dated at or before t, consistent with every other X_t component).
    """
    out = pd.DataFrame(index=panel.index)
    if "f_bar_24h" in panel.columns:
        out["FundingLevel"] = panel["f_bar_24h"]
    else:
        out["FundingLevel"] = panel["f_eff"].rolling(lookback, min_periods=lookback).mean()
    out["FundingVolatility"] = panel["f_eff"].rolling(lookback, min_periods=lookback).std()
    return out


def build_gap_market_conditions_dataset(
    panel: pd.DataFrame,
    price_var_result: pd.DataFrame,
    net_var_result: pd.DataFrame,
    control_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Section 11 item 4: explain Gap_t via the underlying market environment
    (continuous funding level/volatility plus the same controls already used
    elsewhere) rather than by regressing it directly on FundingVaR, and
    rather than the binary HighFunding regime dummy used in
    `fit_gap_regime_test`.
    """
    control_cols = control_cols or MARKET_CONDITION_CONTROLS
    base_df = build_gap_dataset(panel, price_var_result, net_var_result, control_cols=control_cols)
    fv = _funding_level_and_volatility(panel)
    idx = base_df.index.intersection(fv.index)
    df = base_df.loc[idx].copy()
    df["FundingLevel"] = fv.loc[idx, "FundingLevel"]
    df["FundingVolatility"] = fv.loc[idx, "FundingVolatility"]
    return df.dropna().sort_index()


def fit_gap_market_conditions_regression(df: pd.DataFrame, control_cols: list[str] | None = None):
    """Tina's item-4 specification:

    Gap_t = alpha + b1*FundingLevel_t + b2*FundingVolatility_t + b3*RV_t
            + b4*Basis_t + b5*DeltaOI_t + b6*Spread_t + b7*OrderFlow_t
            + b8*VolumeShock_t + eps_t

    HAC (maxlags=2) standard errors, matching `fit_gap_regime_test`: forecast
    origins are non-overlapping (so serial dependence is reduced, per her
    note) but not eliminated, so heteroskedasticity/autocorrelation-robust
    errors are still used. Reported two-sided, since this is an exploratory
    decomposition across eight regressors rather than a single directional
    test.
    """
    control_cols = control_cols or MARKET_CONDITION_CONTROLS
    regressors = ["FundingLevel", "FundingVolatility"] + control_cols
    X = sm.add_constant(df[regressors])
    y = df["gap"]
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return {
        "fit": ols,
        "n": int(ols.nobs),
        "r_squared": float(ols.rsquared),
        "params": {MARKET_CONDITION_LABELS.get(k, k): float(v) for k, v in ols.params.items() if k != "const"},
        "const": float(ols.params["const"]),
        "pvalues_two_sided": {MARKET_CONDITION_LABELS.get(k, k): float(v) for k, v in ols.pvalues.items() if k != "const"},
    }
