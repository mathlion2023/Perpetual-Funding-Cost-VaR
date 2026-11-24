"""Section 7: testing whether price-only VaR omits funding-cost risk.

Pipeline:
  1. Price-only VaR forecast, via quantile regression on a baseline feature
     set (realised volatility, basis, open-interest growth, Deribit quoted
     spread, volume shock) -- reuses models.caviar.direct_quantile_var with
     target_col = "L_price_H{h}". (spread_binc is not in this baseline --
     see the module-level comment on DEFAULT_CONTROLS/BASELINE_PRICE_FEATURES.)
  2. Omission event B_t = 1[L_net_t > VaR_price_t].
  3. Three nested out-of-sample logit specifications for Pr(B_t=1):
       Model 1 (baseline)          : intercept only
       Model 2 (+ controls)        : intercept + X_t
       Model 3 (+ funding VaR)     : intercept + X_t + VaR_funding_t
     Model 3 is Section 7's Pr(B_t=1|F_t) = Lambda(alpha + beta*VaR_funding_t
     + gamma'X_t); H0: beta=0 vs Ha: beta>0 is tested via Model 2 vs Model 3
     (nested likelihood-ratio test, a one-sided-motivated encompassing test).
  4. Out-of-sample Brier score, log score, AUC, and Cox calibration
     (slope/intercept of outcome ~ logit(fitted_prob)) for each model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import config
from .models import base, caviar

# spread_binc deliberately excluded here (2026-09-26): restricting Section 7's
# omission dataset to Binance's 2022-03+ coverage window (via .dropna() in
# build_omission_dataset) left too few rare omission events for the
# full-sample encompassing logit to be identifiable (singular Hessian on both
# H72 and H168 -- see run_stage1_out_of_sample's full_sample_fit_error path).
# Section 7 therefore stays Deribit-only (the longer, all-history sample);
# spread_binc is still used in Section 9's market-conditions regression
# (omission_gap.SECTION9_CONTROLS), which already accepts a shorter window.
BASELINE_PRICE_FEATURES = ["rv_primary", "basis", "d_oi", "spread", "volume_shock"]
DEFAULT_CONTROLS = ["rv_primary", "basis", "d_oi", "spread", "ofi", "volume_shock"]


def price_only_var_backtest(
    panel: pd.DataFrame,
    horizon_hours: int,
    alpha: float = config.VAR_ALPHA,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    target_col = f"L_price_H{horizon_hours}"
    feature_cols = feature_cols or BASELINE_PRICE_FEATURES
    # clip_nonnegative=False (bug fix, 2026-09-29): L_price_H{h} is a signed
    # target (-R^(h)_t, can be negative on net price gains) -- the model's
    # default zero-floor is only correct for the non-negative funding-cost
    # target it was originally written for. See caviar.direct_quantile_var's
    # docstring.
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


def build_omission_dataset(
    panel: pd.DataFrame,
    price_var_result: pd.DataFrame,
    funding_var_result: pd.DataFrame,
    horizon_hours: int,
    control_cols: list[str] | None = None,
) -> pd.DataFrame:
    control_cols = control_cols or DEFAULT_CONTROLS
    idx = price_var_result.index.intersection(funding_var_result.index)
    df = pd.DataFrame(index=idx)
    df["L_net"] = panel.loc[idx, f"L_net_H{horizon_hours}"]
    df["VaR_price"] = price_var_result.loc[idx, "var_forecast"]
    df["VaR_funding"] = funding_var_result.loc[idx, "var_forecast"]
    df["omission_event"] = (df["L_net"] > df["VaR_price"]).astype(float)
    for c in control_cols:
        df[c] = panel.loc[idx, c]
    return df.dropna().sort_index()


def sample_accounting(
    panel: pd.DataFrame,
    price_var_result: pd.DataFrame,
    funding_var_result: pd.DataFrame,
    horizon_hours: int,
    control_cols: list[str] | None = None,
) -> dict:
    """Why scheduled Stage 1 origins are not in the omission dataset (lag
    audit, 2026-11-06): build_omission_dataset's dropna() otherwise hides how
    many origins each of these removes. Counts are not mutually exclusive."""
    control_cols = control_cols or DEFAULT_CONTROLS
    idx = price_var_result.index.intersection(funding_var_result.index)
    return {
        "scheduled_origins": int(len(idx)),
        "L_net_unrealised": int(panel.loc[idx, f"L_net_H{horizon_hours}"].isna().sum()),
        "VaR_price_missing": int(price_var_result.loc[idx, "var_forecast"].isna().sum()),
        "VaR_funding_missing": int(funding_var_result.loc[idx, "var_forecast"].isna().sum()),
        "controls_missing": int(panel.loc[idx, control_cols].isna().any(axis=1).sum()),
    }


def _score(y_true: np.ndarray, p_hat: np.ndarray) -> dict:
    """Peer review 2026-09-26, item R1 -- AUC here is legitimately allowed to
    sit below 0.5 and is legitimately allowed to come out numerically
    identical across two different models' predictions; neither is a bug.

    AUC < 0.5 for the intercept-only walk-forward baseline (m1_baseline)
    happens because that "intercept" is re-estimated at every expanding-
    window origin as the running historical event rate, not a single global
    constant -- if the true event rate is higher earlier in a window's own
    history than in what immediately follows (event clustering / regime
    change, both plausible with only 3-14 total events over 5 years), the
    recursive-mean forecast is too high exactly when subsequent risk is
    lower, which is a real rank-inversion, not noise. Verified directly at
    H=168 on 2026-09-26: the 3 held-out events have mean percentile rank
    0.413 (H=72, 12 events) / an even more skewed pattern at H=168 under
    m1_baseline, both below the 0.5 a well-calibrated classifier would need.

    Two models tying exactly on AUC (m1_baseline and m3_controls_funding both
    gave 0.3819951338199513 at H=168) is a coincidence of AUC's coarseness
    with only 3 positive cases: AUC depends only on the *set* of relative
    ranks the positive cases occupy among the negatives, not on which
    specific case holds which rank. Verified directly: the two models'
    predicted probabilities for the 3 events differ substantially, but each
    model places the 3 events at percentile ranks {~0.7%, ~32%, ~83-84%} --
    the same three slots, populated by a different permutation of which
    event lands where. That produces bit-identical AUC by construction of
    the statistic, independent of any shared computation path.
    """
    p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
    brier = float(np.mean((y_true - p_hat) ** 2))
    log_score = float(np.mean(y_true * np.log(p_hat) + (1 - y_true) * np.log(1 - p_hat)))
    try:
        auc = float(roc_auc_score(y_true, p_hat)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    logit_p = np.log(p_hat / (1 - p_hat))
    try:
        cal_X = sm.add_constant(logit_p)
        cal_fit = sm.Logit(y_true, cal_X).fit(disp=0)
        cal_intercept, cal_slope = cal_fit.params.iloc[0], cal_fit.params.iloc[1]
    except Exception:
        cal_intercept, cal_slope = float("nan"), float("nan")

    return {
        "n": len(y_true),
        "n_events": int(y_true.sum()),
        "brier": brier,
        "log_score": log_score,
        "auc": auc,
        "calibration_intercept": float(cal_intercept),
        "calibration_slope": float(cal_slope),
    }


def _standardize(reference: pd.DataFrame, cols: list[str], *frames: pd.DataFrame) -> list[pd.DataFrame]:
    """z-score `cols` in each of `frames` using `reference`'s own mean/std
    (never the target frame's), so this stays a pure reparameterization with
    no look-ahead. A constant column (std=0) is left unscaled (divide by 1)
    rather than producing inf/NaN.

    Standardizing before every sm.Logit(...).fit() call below is a numerical
    stability fix, not a modeling choice -- see the 2026-09-26 finding that
    this project's MLE solver gave substantively different beta_funding, AUC,
    and LR-test results depending on whether `spread` was in raw-ratio or bps
    units, even though rescaling one covariate should be a no-op for every
    *other* coefficient and every fit statistic in a correctly converged
    logistic regression. Standardizing removes that arbitrary-scale
    sensitivity; coefficients on the standardized scale are converted back to
    the original units where reported (see beta_funding below).
    """
    means = reference[cols].mean()
    stds = reference[cols].std().replace(0, 1.0)
    return [(f[cols] - means) / stds for f in frames] + [stds]


def run_stage1_out_of_sample(df: pd.DataFrame, control_cols: list[str] | None = None, min_train: int = 40) -> dict:
    """Expanding-window out-of-sample fit of the three nested logit specs.

    Returns dict with per-model prediction series (for external inspection),
    scoring metrics, and the in-sample-on-full-data Model2-vs-Model3
    likelihood-ratio (encompassing) test using all available data (reported
    separately from the out-of-sample scores, since a LR test needs a single
    fixed fit, not per-origin refits).
    """
    control_cols = control_cols or DEFAULT_CONTROLS
    df = df.sort_index()
    y = df["omission_event"].to_numpy()
    n = len(df)
    if n <= min_train + 5:
        raise ValueError(f"Not enough Stage-1 origins ({n}) for min_train={min_train}.")

    specs = {
        "m1_baseline": [],
        "m2_controls": control_cols,
        "m3_controls_funding": control_cols + ["VaR_funding"],
    }

    preds = {name: np.full(n, np.nan) for name in specs}
    for i in range(min_train, n):
        train = df.iloc[:i]
        for name, cols in specs.items():
            if cols:
                train_z, new_z, _ = _standardize(train, cols, train, df.iloc[[i]])
                X_train = sm.add_constant(train_z)
            else:
                X_train = sm.add_constant(pd.DataFrame(index=train.index))
            try:
                fit = sm.Logit(train["omission_event"], X_train).fit(disp=0)
                x_new = new_z if cols else pd.DataFrame(index=[df.index[i]])
                X_new = sm.add_constant(x_new, has_constant="add")
                X_new = X_new.reindex(columns=X_train.columns, fill_value=1.0)
                p_hat = fit.predict(X_new).iloc[0]
            except Exception:
                p_hat = train["omission_event"].mean()
            preds[name][i] = p_hat

    scores = {}
    valid = ~np.isnan(preds["m3_controls_funding"])
    for name in specs:
        scores[name] = _score(y[valid], preds[name][valid])

    # The full-sample (all n) encompassing fit below is a rare-event logit
    # (omission_event rate is typically a few percent) with len(control_cols)+2
    # parameters -- thin enough on some horizons/date ranges (e.g. H72's
    # ~18 events / 391 origins after spread_binc's Binance-coverage start date
    # trims the sample) that the Hessian can be exactly singular
    # (quasi-/complete separation), independent of any per-origin fit in the
    # loop above (which already degrades gracefully via its own try/except).
    # Rather than let that crash the whole multi-hour pipeline run and lose
    # every other already-computed section, this degrades to NaN/None with
    # `full_sample_fit_error` set, so the failure is visible in the output
    # instead of silent or fatal.
    m2_cols = control_cols
    m3_cols = control_cols + ["VaR_funding"]
    m2_z, stds_m2 = _standardize(df, m2_cols, df)
    m3_z, stds_m3 = _standardize(df, m3_cols, df)
    X_full_m2 = sm.add_constant(m2_z)
    X_full_m3 = sm.add_constant(m3_z)
    try:
        fit_m2 = sm.Logit(df["omission_event"], X_full_m2).fit(disp=0)
        fit_m3 = sm.Logit(df["omission_event"], X_full_m3).fit(disp=0)
        lr_stat = 2 * (fit_m3.llf - fit_m2.llf)
        lr_pvalue = float(stats.chi2.sf(lr_stat, df=1))
        # The LR p-value above is two-sided: it rejects for beta_funding of
        # either sign, so a small p with beta_funding < 0 is evidence *against*
        # Ha: beta > 0, not for it. The signed-root version tests the one-sided
        # Ha directly (lag audit, 2026-11-06 -- reporting fix only).
        lr_signed_root = float(np.sign(fit_m3.params["VaR_funding"]) * np.sqrt(max(lr_stat, 0.0)))
        lr_pvalue_one_sided = float(stats.norm.sf(lr_signed_root))
        # Convert the standardized-scale coefficient back to original
        # VaR_funding units (divide by its own std) so the reported number
        # means "effect per unit of VaR_funding", matching how it was always
        # reported -- the p-value is scale-invariant either way (standardizing
        # only rescales beta and its SE by the same factor).
        beta_funding = float(fit_m3.params["VaR_funding"]) / float(stds_m3["VaR_funding"])
        beta_pvalue_two_sided = float(fit_m3.pvalues["VaR_funding"])
        beta_pvalue_one_sided = beta_pvalue_two_sided / 2 if beta_funding > 0 else 1 - beta_pvalue_two_sided / 2
        full_sample_fit_error = None
    except np.linalg.LinAlgError as exc:
        fit_m3 = None
        lr_stat, lr_pvalue, lr_pvalue_one_sided = float("nan"), float("nan"), float("nan")
        beta_funding, beta_pvalue_one_sided = float("nan"), float("nan")
        full_sample_fit_error = f"{type(exc).__name__}: {exc}"

    # Firth bias-reduced robustness check (peer review 2026-09-26, item R2):
    # ordinary MLE is finite-sample biased under rare events (as few as 3-14
    # positive cases feed this fit) -- a distinct problem from the scale-
    # conditioning issue the standardization above fixes. Runs independently
    # of the try/except above (Firth's penalization keeps the Hessian
    # non-singular even where ordinary MLE fails), so this is reported even
    # when full_sample_fit_error is set.
    from .models.firth import firth_logit

    firth_m2 = firth_logit(df["omission_event"].to_numpy(), X_full_m2.to_numpy())
    firth_m3 = firth_logit(df["omission_event"].to_numpy(), X_full_m3.to_numpy())
    var_funding_idx = list(X_full_m3.columns).index("VaR_funding")
    firth_beta_funding = float(firth_m3["beta"][var_funding_idx]) / float(stds_m3["VaR_funding"])
    firth_pvalue_two_sided = float(firth_m3["pvalues"][var_funding_idx])
    firth_pvalue_one_sided = firth_pvalue_two_sided / 2 if firth_beta_funding > 0 else 1 - firth_pvalue_two_sided / 2
    firth_lr_stat = 2 * (firth_m3["penalized_loglik"] - firth_m2["penalized_loglik"])
    firth_lr_pvalue = float(stats.chi2.sf(firth_lr_stat, df=1))
    firth_lr_pvalue_one_sided = float(stats.norm.sf(np.sign(firth_beta_funding) * np.sqrt(max(firth_lr_stat, 0.0))))

    return {
        "predictions": {name: pd.Series(p, index=df.index) for name, p in preds.items()},
        "scores": scores,
        "full_sample_fit_m3": fit_m3,
        "full_sample_fit_error": full_sample_fit_error,
        "beta_funding": beta_funding,
        "beta_funding_pvalue_one_sided_H0_beta_le_0": beta_pvalue_one_sided,
        "lr_test_m2_vs_m3": {"statistic": float(lr_stat), "pvalue": lr_pvalue, "df": 1,
                             "pvalue_one_sided_signed_root": lr_pvalue_one_sided},
        "n_origins": int(n),
        "n_events": int(y.sum()),
        "firth_robustness": {
            "beta_funding": firth_beta_funding,
            "beta_funding_pvalue_one_sided_H0_beta_le_0": firth_pvalue_one_sided,
            "lr_test_m2_vs_m3": {"statistic": float(firth_lr_stat), "pvalue": firth_lr_pvalue, "df": 1,
                                 "pvalue_one_sided_signed_root": firth_lr_pvalue_one_sided},
        },
    }
