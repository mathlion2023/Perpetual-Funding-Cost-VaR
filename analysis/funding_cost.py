"""Sections 4-5: funding-cost outcome, net loss, and collateral drain.

All quantities here are *forward-looking targets* built from the hourly
panel's f_eff and mark_price columns -- they must never be used as
predictors dated t (that is forecast_set.py's job, which only uses
information available by the end of hour t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def funding_rate_distribution_summary(panel: pd.DataFrame) -> dict:
    """Descriptive statistics of the raw hourly effective funding rate f_eff
    itself (Tina, 2026-10-07 email, "funding distribution statistics") --
    distinct from the derived gross/net funding-effect and PACR tables
    elsewhere in this module, which describe the *consequence* of funding
    for collateral. This describes funding as a time series in its own
    right: level, dispersion, persistence, and the split between
    paying/receiving hours.
    """
    f = panel["f_eff"].dropna()
    return {
        "n": int(len(f)),
        "mean_hourly": float(f.mean()),
        "std_hourly": float(f.std()),
        "skew": float(f.skew()),
        "min": float(f.min()),
        "max": float(f.max()),
        "share_positive": float((f > 0).mean()),
        "share_negative": float((f < 0).mean()),
        "share_zero": float((f == 0).mean()),
        "acf_lag1": float(f.autocorr(1)),
        "acf_lag8": float(f.autocorr(8)),
        "annualised_mean_pct": float(f.mean() * 24 * 365 * 100),
    }


def forward_sum(s: pd.Series, horizon: int) -> pd.Series:
    """out[t] = sum(s[t+1 .. t+horizon]) -- i.e. horizon hours strictly after t.

    Assumes `s` is indexed on a complete, regularly-spaced hourly grid.
    NaN wherever the full forward window would run past the end of the series,
    or if any input value in the window is NaN.
    """
    from .models.base import assert_hourly_grid

    assert_hourly_grid(s.index, "forward_sum input")
    x = s.to_numpy(dtype=float)
    n = len(x)
    csum =np.concatenate(([0.0], np.cumsum(np.nan_to_num(x, nan=0.0))))
    nan_mask = np.concatenate(([0.0], np.cumsum(np.isnan(x).astype(float))))

    # csum[k] = sum(x[0 .. k-1]), so sum(x[t+1 .. t+horizon]) = csum[t+horizon+1] - csum[t+1].
    # Bug fix (2026-09-28): this previously subtracted csum[lo - 1] = csum[t],
    # which summed x[t .. t+horizon] -- horizon+1 hours including the origin
    # hour itself, contradicting this docstring. Every C_*_H{h} target (and so
    # L_net_H{h}) carried the origin hour's already-observed funding.
    out = np.full(n, np.nan)
    valid_t = np.arange(n - horizon)
    lo, hi = valid_t + 1, valid_t + horizon + 1
    has_nan_in_window = (nan_mask[hi] - nan_mask[lo]) > 0
    sums = csum[hi] - csum[lo]
    sums[has_nan_in_window] = np.nan
    out[valid_t] = sums
    return pd.Series(out, index=s.index)


def forward_value(s: pd.Series, horizon: int) -> pd.Series:
    """out[t] = s[t+horizon]."""
    from .models.base import assert_hourly_grid

    assert_hourly_grid(s.index, "forward_value input")
    return s.shift(-horizon)


def add_funding_cost_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Section 4: per-hour long-side funding cost, plus signed/short-side robustness variants."""
    out = panel.copy()
    out["c_long"] = out["f_eff"].clip(lower=0.0)
    out["c_signed"] = out["f_eff"]
    out["c_short"] = (-out["f_eff"]).clip(lower=0.0)

    for h in config.HORIZONS_HOURS:
        out[f"C_long_H{h}"] = forward_sum(out["c_long"], h)
        out[f"C_signed_H{h}"] = forward_sum(out["c_signed"], h)
        out[f"C_short_H{h}"] = forward_sum(out["c_short"], h)
    return out


def add_price_and_loss_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Section 5: holding-period return, net loss, gross downside loss.

    L_net_H{h} (Tina, 2026-09-27 email, "Signed funding and net loss") is the
    actual net P&L a long position experiences: price loss plus *signed*
    funding cash flow, so a period where longs receive funding (f_eff<0)
    reduces net loss rather than being floored at zero. This must use
    c_signed, not c_long -- c_long is deliberately clipped to >=0 and is only
    appropriate for the separately-labelled gross-payment/collateral-drain
    measures below, which describe "how much funding alone can cost you" and
    are not meant to net against funding receipts.
    """
    out = panel.copy()
    for h in config.HORIZONS_HOURS:
        m_fwd = forward_value(out["mark_price"], h)
        r_h = (m_fwd - out["mark_price"]) / out["mark_price"]
        out[f"R_H{h}"] = r_h
        out[f"L_price_H{h}"] = -r_h  # Section 7's price-only loss target for VaR^price

        c_long_h = out[f"C_long_H{h}"] if f"C_long_H{h}" in out.columns else forward_sum(out["c_long"], h)
        c_signed_h = out[f"C_signed_H{h}"] if f"C_signed_H{h}" in out.columns else forward_sum(out["c_signed"], h)
        out[f"L_net_H{h}"] = -r_h + c_signed_h
        out[f"L_gross_downside_H{h}"] = (-r_h).clip(lower=0.0) + c_long_h
    return out


def add_collateral_drain(panel: pd.DataFrame) -> pd.DataFrame:
    """Section 5: funding-driven collateral drain at each configured leverage level."""
    out = panel.copy()
    for h in config.HORIZONS_HOURS:
        c_col = f"C_long_H{h}"
        for k in config.LEVERAGE_LEVELS:
            out[f"CollateralDrain_H{h}_k{k}"] = k * out[c_col]
    return out


def collateral_drain_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Descriptive table: mean/median/95th/99th pct collateral drain and share of
    windows exceeding the configured thresholds, for each horizon x leverage pair.

    This is a *gross funding-payment* measure (built on C_long_H, clipped to
    >=0 -- see add_collateral_drain), not net P&L: it answers "how much can
    funding payments alone cost, ignoring any offsetting receipts or price
    moves", by design. It should never be read as, or relabelled as, a net-
    loss figure -- that is L_net_H{h} (funding_cost.add_price_and_loss_outcomes),
    which nets signed funding receipts against price P&L.

    Tina, 2026-09-27 email: also report the mean drain *within* the worst 1%
    of cases (mean_p99_tail), distinct from the p99 threshold value itself --
    the two answer different questions ("what's the cutoff for the worst 1%"
    vs "how bad is it on average once you're in that worst 1%").
    """
    rows = []
    for h in config.HORIZONS_HOURS:
        for k in config.LEVERAGE_LEVELS:
            col = f"CollateralDrain_H{h}_k{k}"
            s = panel[col].dropna()
            p99_threshold = s.quantile(0.99)
            row = {
                "horizon_h": h,
                "leverage": k,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "p95": s.quantile(0.95),
                "p99": p99_threshold,
                "mean_p99_tail": float(s[s >= p99_threshold].mean()),
            }
            for thr in config.COLLATERAL_DRAIN_THRESHOLDS:
                row[f"share_ge_{thr:.0%}"] = float((s >= thr).mean())
            rows.append(row)
    return pd.DataFrame(rows)
