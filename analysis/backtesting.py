"""Section 10: out-of-sample VaR backtesting suite.

Operates on the DataFrame produced by models.base.run_rolling_backtest
(columns: var_forecast, realized, exception), indexed by non-overlapping
forecast origins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def check_loss(u: np.ndarray, alpha: float) -> np.ndarray:
    return u * (alpha - (u < 0).astype(float))


def quantile_loss(df: pd.DataFrame, alpha: float) -> float:
    d = df.dropna(subset=["var_forecast", "realized"])
    u = d["realized"].to_numpy() - d["var_forecast"].to_numpy()
    return float(check_loss(u, alpha).mean())


def kupiec_uc_test(exceptions: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) unconditional-coverage test. H0: exception rate == 1-alpha."""
    p = 1 - alpha
    n = len(exceptions)
    x = int(exceptions.sum())
    if n == 0:
        return {"n": 0, "x": 0, "phat": float("nan"), "LR_uc": float("nan"), "pvalue": float("nan")}
    phat = x / n

    def _loglik(prob):
        if prob <= 0 or prob >= 1:
            return -np.inf
        return (n - x) * np.log(1 - prob) + x * np.log(prob)

    lr = -2 * (_loglik(p) - _loglik(phat if 0 < phat < 1 else min(max(phat, 1e-8), 1 - 1e-8)))
    pvalue = float(stats.chi2.sf(lr, df=1))
    return {"n": n, "x": x, "phat": phat, "p_nominal": p, "LR_uc": float(lr), "pvalue": pvalue}


def christoffersen_independence_test(exceptions: np.ndarray) -> dict:
    """Christoffersen (1998) independence test on the exception sequence."""
    e = exceptions.astype(int)
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(e[:-1], e[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / max(n0 + n1, 1)

    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    ll_restricted = (n0 + n1 - n01 - n11) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_unrestricted = n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01) + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    lr = -2 * (ll_restricted - ll_unrestricted)
    pvalue = float(stats.chi2.sf(lr, df=1))
    return {"LR_ind": float(lr), "pvalue": pvalue, "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_conditional_coverage_test(exceptions: np.ndarray, alpha: float) -> dict:
    uc = kupiec_uc_test(exceptions, alpha)
    ind = christoffersen_independence_test(exceptions)
    lr_cc = uc["LR_uc"] + ind["LR_ind"]
    pvalue = float(stats.chi2.sf(lr_cc, df=2))
    return {"LR_cc": float(lr_cc), "pvalue": pvalue, "uc": uc, "independence": ind}


def christoffersen_pelletier_duration_test(exceptions: np.ndarray) -> dict:
    """Simplified continuous-time Weibull duration test (Christoffersen & Pelletier, 2004).

    Durations between exceptions (in units of forecast origins) are treated
    as a continuous Weibull hazard; H0: shape parameter b=1 (exponential,
    i.e. memoryless / correctly calibrated hazard). LR test, chi2(1).

    This is a simplified discrete-origin implementation of the original
    continuous-time formulation -- adequate as a first pass, but a documented
    approximation, not the full continuous-time censored-duration MLE.
    """
    idx_exc = np.where(exceptions == 1)[0]
    if len(idx_exc) < 4:
        return {"LR_dur": float("nan"), "pvalue": float("nan"), "n_durations": len(idx_exc), "note": "too few exceptions for a duration test"}

    durations = np.diff(idx_exc).astype(float)
    durations = durations[durations > 0]
    n = len(durations)

    lam_hat = 1.0 / durations.mean()
    ll_exp = n * np.log(lam_hat) - lam_hat * durations.sum()

    def neg_ll_weibull(params):
        a, b = params
        if a <= 0 or b <= 0:
            return np.inf
        d = durations
        ll = n * np.log(a / b) + (a - 1) * np.log(d / b).sum() - ((d / b) ** a).sum()
        return -ll

    from scipy.optimize import minimize

    res = minimize(neg_ll_weibull, x0=[1.0, durations.mean()], method="Nelder-Mead")
    ll_weibull = -res.fun if res.success else ll_exp

    lr = max(0.0, -2 * (ll_exp - ll_weibull))
    pvalue = float(stats.chi2.sf(lr, df=1))
    return {"LR_dur": float(lr), "pvalue": pvalue, "n_durations": n, "mean_duration_origins": float(durations.mean())}


def exceedance_severity(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["var_forecast", "realized", "exception"])
    exc = d[d["exception"] == 1]
    if len(exc) == 0:
        return {"n_exceptions": 0, "mean_severity_ratio": float("nan")}
    ratio = exc["realized"] / exc["var_forecast"].replace(0, np.nan)
    return {
        "n_exceptions": len(exc),
        "mean_severity_ratio": float(ratio.mean()),
        "median_severity_ratio": float(ratio.median()),
        "max_severity_ratio": float(ratio.max()),
    }


def es_robustness_check(df: pd.DataFrame) -> dict:
    """Simplified Expected-Shortfall-oriented check (McNeil & Frey, 2000 motivation).

    A full McNeil-Frey ES-regression backtest needs a model-implied ES
    forecast; this pipeline's models currently only produce VaR forecasts.
    As a documented proxy, this tests whether the standardised exceedance
    residual (realized - VaR)/VaR has mean zero across exceptions (a
    necessary, not sufficient, condition for a well-calibrated tail beyond
    the VaR threshold itself). Treat as a lightweight diagnostic, not a
    substitute for a proper ES-regression backtest.
    """
    d = df.dropna(subset=["var_forecast", "realized", "exception"])
    exc = d[d["exception"] == 1]
    if len(exc) < 5:
        return {"n_exceptions": len(exc), "mean_std_residual": float("nan"), "tstat": float("nan"), "pvalue": float("nan")}
    resid = (exc["realized"] - exc["var_forecast"]) / exc["var_forecast"].replace(0, np.nan)
    resid = resid.dropna()
    tstat, pvalue = stats.ttest_1samp(resid, popmean=0.0)
    return {"n_exceptions": len(resid), "mean_std_residual": float(resid.mean()), "tstat": float(tstat), "pvalue": float(pvalue)}


def diebold_mariano_test(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha: float, h: int = 1) -> dict:
    """Diebold & Mariano (1995) test on the check-loss differential between two models.

    H0: equal predictive accuracy. Positive DM stat => model A has higher
    loss (worse) than model B.
    """
    idx = df_a.index.intersection(df_b.index)
    a = df_a.loc[idx].dropna(subset=["var_forecast", "realized"])
    b = df_b.loc[idx].dropna(subset=["var_forecast", "realized"])
    idx2 = a.index.intersection(b.index)
    a, b = a.loc[idx2], b.loc[idx2]

    loss_a = check_loss(a["realized"].to_numpy() - a["var_forecast"].to_numpy(), alpha)
    loss_b = check_loss(b["realized"].to_numpy() - b["var_forecast"].to_numpy(), alpha)
    d = loss_a - loss_b
    n = len(d)
    if n < 10:
        return {"n": n, "DM_stat": float("nan"), "pvalue": float("nan")}

    d_mean = d.mean()
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1] if n > lag else 0.0
        var_d += 2 * cov
    var_d = max(var_d, 1e-12) / n
    dm_stat = d_mean / np.sqrt(var_d)
    pvalue = float(2 * stats.norm.sf(abs(dm_stat)))
    return {"n": n, "DM_stat": float(dm_stat), "pvalue": pvalue, "mean_loss_diff": float(d_mean)}


def backtest_report(df: pd.DataFrame, alpha: float) -> dict:
    """Bundle all Section 10 diagnostics for a single model's backtest result."""
    d = df.dropna(subset=["var_forecast", "realized", "exception"])
    exceptions = d["exception"].to_numpy()
    return {
        "n_forecasts": len(d),
        "expected_exceptions": len(d) * (1 - alpha),
        "realized_exceptions": int(exceptions.sum()),
        "kupiec_uc": kupiec_uc_test(exceptions, alpha),
        "christoffersen_independence": christoffersen_independence_test(exceptions),
        "christoffersen_conditional_coverage": christoffersen_conditional_coverage_test(exceptions, alpha),
        "christoffersen_pelletier_duration": christoffersen_pelletier_duration_test(exceptions),
        "exceedance_severity": exceedance_severity(d),
        "es_robustness_check": es_robustness_check(d),
        "quantile_loss": quantile_loss(d, alpha),
    }


def availability_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per model: how many scheduled origins produced a usable (forecast,
    realised) pair, and why the rest did not -- an exception raised inside the
    model, a NaN forecast returned without an exception (a model guard, e.g.
    non-stationary root or unobserved origin hour), or an unrealised target
    (forward window touches a missing hour). Makes model failures visible
    instead of letting them silently change each model's sample (lag audit,
    2026-11-06)."""
    rows = []
    for name, res in results.items():
        fc_ok = res["var_forecast"].notna()
        err = res["error"].notna() if "error" in res else pd.Series(False, index=res.index)
        rows.append({
            "model": name,
            "scheduled_origins": len(res),
            "usable": int((fc_ok & res["realized"].notna()).sum()),
            "model_exception": int(err.sum()),
            "model_returned_nan": int((~fc_ok & ~err).sum()),
            "target_unrealised": int(res["realized"].isna().sum()),
        })
    return pd.DataFrame(rows).set_index("model")


def common_sample(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model to the origins at which *all* models have a usable
    forecast and realised outcome, so that cross-model comparisons are on the
    same dates."""
    idx = None
    for res in results.values():
        ok = res.dropna(subset=["var_forecast", "realized"]).index
        idx = ok if idx is None else idx.intersection(ok)
    return {name: res.loc[idx] for name, res in results.items()}


def compare_models(results: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    """Pairwise Diebold-Mariano matrix (row model vs column model, check-loss basis)."""
    names = list(results.keys())
    rows = []
    for a_name in names:
        row = {"model": a_name}
        for b_name in names:
            if a_name == b_name:
                row[b_name] = 0.0
                continue
            dm = diebold_mariano_test(results[a_name], results[b_name], alpha)
            row[b_name] = dm["DM_stat"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
