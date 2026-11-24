"""Q4 extension (Tina, 2026-10-07 email): "Does volatility information
improve prediction of future collateral requirements?"

Dependent variable is total funding-inclusive PACRRatio (long, 10x leverage
-- the same target as H3/H4, built via collateral_erosion.build_h3_h4_dataset,
already funding-inclusive per the Q1 confirmation of net_max_arrays). A
separate participation (occurrence) model predicts PACRRatio>0; a severity
(quantile) model predicts its size conditional on a shortfall. Every
competing model uses the identical baseline predictors (basis, d_oi, spread,
ofi, volume_shock) and the identical collateral-prediction method (logit +
quantile regression, mirroring collateral_erosion._hurdle_fit's hurdle
design) -- only the volatility feature differs. This duplicates _hurdle_fit's
fitting logic (rather than importing it directly) because this module also
needs the fitted model objects for genuine out-of-sample prediction on a
held-out test period, which _hurdle_fit (built for H3/H4's in-sample
reporting) does not expose.

Explicitly distinct from analysis/models/rough_vol.py's direct_rough_vol_var
("P8" in the proposal's informal numbering, though no file in this repo uses
that literal label): that model's target is f_eff (funding RATE) as a
candidate reserve-policy input. This module's target is PACRRatio (the
funding-inclusive peak collateral requirement), asking whether volatility
information of various kinds predicts *that* -- a different question, reusing
only the feature-construction logic (rough_vol.estimate_hurst/
rough_vol_state), not the funding-rate model itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.regression.quantile_regression import QuantReg

from . import collateral_erosion as ce
from .chronological_split import chronological_split

BASELINE_PREDICTORS = ["basis", "d_oi", "spread", "ofi", "volume_shock"]
QUANTILE_TAUS = ce.QUANTILE_TAUS


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def fit_hurdle_oos(df_fit: pd.DataFrame, predictor_cols: list[str], taus=QUANTILE_TAUS) -> dict:
    """Fits participation logit + severity quantile regression on df_fit,
    standardising predictors by df_fit's own mean/std (stored for consistent
    application to out-of-sample data -- test-period features must never be
    restandardised against their own, different, mean/std)."""
    X = df_fit[predictor_cols]
    mean, std = X.mean(), X.std().replace(0, 1.0)
    Xs = (X - mean) / std
    Xc = sm.add_constant(Xs, has_constant="add")
    y = df_fit["PACRRatio"]
    participate = (y > 0).astype(float)
    logit = sm.Logit(participate, Xc).fit(disp=0)

    pos = df_fit[y > 0]
    min_obs_severity = len(predictor_cols) + 2  # need more rows than regressors for QuantReg to be identified
    qr = {}
    if len(pos) >= min_obs_severity:
        Xp = (pos[predictor_cols] - mean) / std
        Xpc = sm.add_constant(Xp, has_constant="add")
        yp = pos["PACRRatio"]
        for tau in taus:
            try:
                qr[tau] = QuantReg(yp, Xpc).fit(q=tau, max_iter=5000)
            except Exception:
                qr[tau] = None
    else:
        qr = {tau: None for tau in taus}
    return {"logit": logit, "qr": qr, "mean": mean, "std": std, "predictor_cols": predictor_cols,
            "n_fit": len(df_fit), "n_fit_positive": int(participate.sum())}


def _design(fit: dict, df: pd.DataFrame, params_index: pd.Index) -> pd.DataFrame:
    X = (df[fit["predictor_cols"]] - fit["mean"]) / fit["std"]
    Xc = sm.add_constant(X, has_constant="add")
    return Xc.reindex(columns=params_index, fill_value=0.0)


def evaluate_oos(fit: dict, df_test: pd.DataFrame, taus=QUANTILE_TAUS) -> dict:
    y = df_test["PACRRatio"]
    participate = (y > 0).astype(float)
    X_logit = _design(fit, df_test, fit["logit"].params.index)
    p_hat = fit["logit"].predict(X_logit)
    auc = float(roc_auc_score(participate, p_hat)) if participate.nunique() > 1 else float("nan")
    brier = float(np.mean((participate.to_numpy() - p_hat.to_numpy()) ** 2))

    pos = df_test[y > 0]
    sev = {}
    for tau in taus:
        qr = fit["qr"].get(tau)
        if qr is None or not len(pos):
            sev[tau] = np.nan
            continue
        X_qr = _design(fit, pos, qr.params.index)
        y_pred = qr.predict(X_qr)
        sev[tau] = pinball_loss(pos["PACRRatio"].to_numpy(), y_pred.to_numpy(), tau)

    return {"n_test": len(df_test), "n_test_positive": int(participate.sum()),
            "auc": auc, "brier": brier, **{f"pinball_tau{tau}": v for tau, v in sev.items()}}


def build_dataset(panel: pd.DataFrame, h: int, vol_feature: str | pd.Series | None,
                   origins: pd.DatetimeIndex | None = None, vol_name: str = "vol") -> pd.DataFrame:
    """H3/H4's dataset (already funding-inclusive PACRRatio, long, 10x,
    baseline predictors), restricted to `origins` (non-overlapping,
    horizon-spaced -- walk-forward volatility-feature computation at every
    overlapping hourly origin, tens of thousands of them, is computationally
    infeasible for GARCH/HAR refitting; non-overlapping origins also avoid
    overstating predictive-comparison precision the same way overlap
    inflates apparent sample size elsewhere in this project, see H5/Q2),
    with a volatility feature swapped in/out. vol_feature is either a panel
    column name (e.g. "rv_primary", the historical-vol information source)
    or an already-computed Series indexed by origin timestamp (a
    walk-forward garch/har/rough feature, only defined at `origins`) --
    joined by index under `vol_name`."""
    base = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)
    if origins is not None:
        base = base.loc[base.index.intersection(origins)]
    out = base[["PACRRatio"]].copy()
    for c in BASELINE_PREDICTORS:
        out[c] = panel.loc[out.index, c]
    if vol_feature is not None:
        if isinstance(vol_feature, str):
            out[vol_name] = panel.loc[out.index, vol_feature]
        else:
            out[vol_name] = vol_feature.reindex(out.index)
    return out.dropna()


def build_dataset_multi(panel: pd.DataFrame, h: int, vol_features: dict[str, "str | pd.Series"],
                         origins: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Like build_dataset, but for the nested rough-vol test, which needs two
    volatility features (the best conventional one plus rough) on the same
    rows simultaneously."""
    base = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)
    if origins is not None:
        base = base.loc[base.index.intersection(origins)]
    out = base[["PACRRatio"]].copy()
    for c in BASELINE_PREDICTORS:
        out[c] = panel.loc[out.index, c]
    for name, feature in vol_features.items():
        if isinstance(feature, str):
            out[name] = panel.loc[out.index, feature]
        else:
            out[name] = feature.reindex(out.index)
    return out.dropna()


def regime_stratification(panel: pd.DataFrame, h: int, split: dict, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Low/medium/high volatility terciles of rv_primary, thresholds fixed
    from the TRAIN split only, applied forward to classify TEST-period
    origins; compares subsequent PACRRatio and shortfall occurrence.
    Restricted to `origins` (non-overlapping, horizon-spaced) for the same
    reason as build_dataset."""
    train_rv = panel.loc[panel.index.intersection(split["train"]).intersection(origins), "rv_primary"].dropna()
    q1, q2 = train_rv.quantile([1 / 3, 2 / 3])

    ds = ce.build_pacr_dataset(panel, h)
    ds = ds.loc[ds.index.intersection(origins)]
    pr = pd.Series(ce.pacr_ratio(ds["net_max_long"].to_numpy(), ds["mark_price_origin"].to_numpy(), ce.PRIMARY_LEVERAGE),
                    index=ds.index)
    test_idx = ds.index.intersection(split["test"])
    rv_test = panel.loc[test_idx, "rv_primary"]
    pr_test = pr.loc[test_idx]
    df = pd.DataFrame({"rv": rv_test, "pacr_ratio": pr_test}).dropna()
    df["regime"] = pd.cut(df["rv"], bins=[-np.inf, q1, q2, np.inf], labels=["low", "medium", "high"])

    rows = []
    for regime, g in df.groupby("regime", observed=True):
        rows.append({
            "horizon": h, "regime": regime, "n": len(g),
            "rv_threshold_lo": float(q1), "rv_threshold_hi": float(q2),
            "share_positive": float((g["pacr_ratio"] > 0).mean()) if len(g) else np.nan,
            "mean_pacr_ratio": float(g["pacr_ratio"].mean()) if len(g) else np.nan,
            "p95_pacr_ratio": float(g["pacr_ratio"].quantile(0.95)) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)
