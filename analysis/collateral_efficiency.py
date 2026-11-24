"""Q5 extension (Tina, 2026-10-07 email): "Can funding and volatility
information provide similar protection with less collateral, or better
protection with the same collateral?"

Unlike H5 (collateral_erosion.py's P0-P3/P7 reserve policies, each an
additive price-reserve + funding-reserve sum) and unlike Q4's two-part
hurdle model (built for hypothesis testing / information-value comparison),
here the reserve IS the predicted upper-quantile of total PACRRatio
directly: "use the predicted upper quantile of total PACR as the additional
reserve; do not add it on top of the existing price reserve, because it
already predicts the total additional requirement." Five models isolate
funding's and volatility's separate contributions, holding everything else
(forecast origins, horizons, baseline predictors, leverage) fixed:

  price_only              : baseline predictors only
  price_funding            : baseline + f_eff (current funding rate)
  price_vol                 : baseline + rv_primary (historical volatility)
  price_funding_vol          : baseline + f_eff + rv_primary
  price_funding_vol_rough     : baseline + f_eff + rv_primary + rough-vol feature

("historical" is used as the single conventional-volatility representative
here, per the Q4 finding that no one of historical/GARCH/HAR dominated --
Q5 asks about funding/volatility/rough as information *categories*, not
about re-selecting the best conventional model.)

Reused machinery: collateral_erosion.build_h3_h4_dataset (PACRRatio target,
baseline predictors), chronological_split, models.base.non_overlapping_origins,
models.volatility_features.rough_vol_feature (walk-forward).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.stats.proportion import proportion_confint

from . import collateral_erosion as ce

BASELINE_PREDICTORS = ["basis", "d_oi", "spread", "ofi", "volume_shock"]

MODEL_FEATURES = {
    "price_only": [],
    "price_funding": ["funding"],
    "price_vol": ["vol"],
    "price_funding_vol": ["funding", "vol"],
    "price_funding_vol_rough": ["funding", "vol", "rough"],
}


def build_dataset(panel: pd.DataFrame, h: int, leverage: int, origins: pd.DatetimeIndex | None,
                   extra_features: dict) -> pd.DataFrame:
    """H3/H4's dataset (PACRRatio target, long direction, baseline
    predictors) at the given leverage, restricted to `origins`, with
    additional named features joined on (panel column name, or an
    already-computed Series indexed by origin -- e.g. the walk-forward
    rough-vol feature, only defined at the evaluation origins)."""
    base = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=leverage)
    if origins is not None:
        base = base.loc[base.index.intersection(origins)]
    out = base[["PACRRatio"]].copy()
    for c in BASELINE_PREDICTORS:
        out[c] = panel.loc[out.index, c]
    for name, feat in extra_features.items():
        if isinstance(feat, str):
            out[name] = panel.loc[out.index, feat]
        else:
            out[name] = feat.reindex(out.index)
    return out.dropna()


def fit_quantile_reserve(df_fit: pd.DataFrame, predictor_cols: list[str], tau: float) -> dict:
    X = df_fit[predictor_cols]
    mean, std = X.mean(), X.std().replace(0, 1.0)
    Xs = (X - mean) / std
    Xc = sm.add_constant(Xs, has_constant="add")
    y = df_fit["PACRRatio"]
    qr = QuantReg(y, Xc).fit(q=tau, max_iter=5000)
    return {"qr": qr, "mean": mean, "std": std, "predictor_cols": predictor_cols, "tau": tau, "n_fit": len(df_fit)}


def predict_reserve(fit: dict, df: pd.DataFrame) -> pd.Series:
    X = (df[fit["predictor_cols"]] - fit["mean"]) / fit["std"]
    Xc = sm.add_constant(X, has_constant="add").reindex(columns=fit["qr"].params.index, fill_value=0.0)
    pred = fit["qr"].predict(Xc)
    return pred.clip(lower=0.0)  # a reserve cannot be negative


def evaluate_reserve(reserve: pd.Series, pacr: pd.Series) -> dict:
    df = pd.DataFrame({"reserve": reserve, "pacr": pacr}).dropna()
    n = len(df)
    if n == 0:
        return {"n": 0}
    shortfall = df["pacr"] > df["reserve"]
    k = int(shortfall.sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    unmet = (df["pacr"] - df["reserve"]).clip(lower=0.0)
    sev = unmet[shortfall]
    return {
        "n": n, "mean_reserve": float(df["reserve"].mean()),
        "shortfalls": k, "shortfall_rate": k / n,
        "shortfall_rate_ci95_lo": float(lo), "shortfall_rate_ci95_hi": float(hi),
        "mean_unmet_all": float(unmet.mean()),
        "conditional_severity": float(sev.mean()) if k else 0.0,
        "p95_unmet_all": float(unmet.quantile(0.95)),
        "p99_unmet_all": float(unmet.quantile(0.99)),
    }


def bootstrap_diff_ci(reserve_a: pd.Series, reserve_b: pd.Series, pacr: pd.Series, metric: str,
                       n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap CI for (metric_b - metric_a), resampling origins with
    replacement -- origins are non-overlapping here, so an iid bootstrap over
    origins is a defensible approximation (unlike the overlapping-origin case
    elsewhere in this project, which needed a block bootstrap, see Section
    15.4b of the main report)."""
    df = pd.DataFrame({"a": reserve_a, "b": reserve_b, "pacr": pacr}).dropna()
    n = len(df)
    if n == 0:
        return {"diff": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}

    def _metric(res: pd.Series, p: pd.Series) -> float:
        sf = p > res
        if metric == "shortfall_rate":
            return float(sf.mean())
        unmet = (p - res).clip(lower=0.0)
        if metric == "mean_unmet_all":
            return float(unmet.mean())
        raise ValueError(metric)

    point = _metric(df["b"], df["pacr"]) - _metric(df["a"], df["pacr"])
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n, n)
        d = df.iloc[sel]
        boots[i] = _metric(d["b"], d["pacr"]) - _metric(d["a"], d["pacr"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(point), "ci95_lo": float(lo), "ci95_hi": float(hi)}
