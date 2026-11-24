"""Serial-correlation-robust re-estimation of the H3/H4 hurdle models.

Addresses the CRITICAL finding in Topic1_Collateral_Erosion_Peer_Review_Report.md:
collateral_erosion.py::_hurdle_fit fits sm.Logit and QuantReg on the full
overlapping-origin sample (consecutive hourly origins share all but one hour
of their horizon-hour outcome window) with no correction for the resulting
serial dependence, unlike the H5/McNemar comparison, which avoids overlap
entirely by construction (base.py::non_overlapping_origins).

This script keeps the original point estimates (they are consistent; only
the standard errors/p-values are invalid) and re-derives inference two ways:

1. Participation (logit): cluster-robust covariance, clusters defined by
   non-overlapping horizon-width time blocks -- the same block width used by
   the H5 comparison, applied here as a clustering variable rather than a
   subsampling scheme, so the full sample is retained.
2. Severity (quantile regression): block bootstrap. Each bootstrap draw
   resamples whole horizon-width blocks (not individual rows) with
   replacement, refits QuantReg, and the empirical bootstrap standard
   deviation of each coefficient is used for a normal-approximation p-value.
   This mirrors the moving-block-bootstrap requirement already stated in the
   project's own source design document (Sec 8.3) for rolling-horizon
   inference, which was applied to H5 but not previously to H3/H4.

Usage:
    python -m analysis.run_collateral_erosion_robust --out analysis/output/collateral_erosion_2026_09_29_robust
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.regression.quantile_regression import QuantReg

from . import collateral_erosion as ce

N_BOOT = 1000
SEED = 0


def cluster_ids(index: pd.DatetimeIndex, block_width_hours: float, offset_hours: float = 0.0,
                 anchor: pd.Timestamp | None = None) -> np.ndarray:
    """Non-overlapping time blocks of width block_width_hours, with the
    boundary grid shifted earlier by offset_hours (sensitivity to block
    length -- offset_hours != 0 tests whether inference is sensitive to the
    arbitrary choice of where the partition starts, not just its width).
    Defaults (block_width_hours=horizon_hours, offset_hours=0) reproduce the
    original partition.

    anchor fixes the calendar-time origin the block grid is measured from.
    Pass the FULL panel's first timestamp here when clustering an already-
    filtered subsample (e.g. block_bootstrap_quantreg's PACRRatio>0 rows) --
    anchoring on the filtered subsample's own first timestamp instead (the
    pre-2026-10-07 default, `anchor=None` falling back to `index.min()`) is
    an uncontrolled, data-dependent phase shift of the block grid, not a
    genuine difference in dependence structure, and was found to flip at
    least one severity-cell's significance classification (see the Q2
    peer-review note in Topic1_Collateral_Erosion_Pilot_Results.tex,
    Section 15.4b)."""
    if anchor is None:
        anchor = index.min()
    elapsed_hours = (index - anchor) / pd.Timedelta(hours=1)
    return ((elapsed_hours + offset_hours) // block_width_hours).astype(int).to_numpy()


def cluster_robust_logit(df: pd.DataFrame, predictor_cols: list[str], horizon_hours: int,
                          standardise: bool, block_width_hours: float | None = None,
                          offset_hours: float = 0.0) -> dict:
    if block_width_hours is None:
        block_width_hours = horizon_hours
    X = df[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = sm.add_constant(X)
    y = (df["PACRRatio"] > 0).astype(float)
    groups = cluster_ids(df.index, block_width_hours, offset_hours)

    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    out = {"n": int(fit.nobs), "n_clusters": int(len(np.unique(groups))),
           "share_positive": float(y.mean()), "pseudo_r2": float(fit.prsquared)}
    for c in predictor_cols:
        out[c] = float(fit.params[c])
        out[f"{c}_se_cluster"] = float(fit.bse[c])
        out[f"{c}_p_cluster"] = float(fit.pvalues[c])
    return out


def _fit_quantreg(X: pd.DataFrame, y: pd.Series, tau: float, predictor_cols: list[str]) -> dict | None:
    try:
        res = QuantReg(y, X).fit(q=tau, max_iter=5000)
    except Exception:
        return None
    return {c: float(res.params[c]) for c in (["const"] + predictor_cols)}


def block_bootstrap_quantreg(df: pd.DataFrame, predictor_cols: list[str], horizon_hours: int,
                              taus, standardise: bool, n_boot: int = N_BOOT, seed: int = SEED,
                              block_width_hours: float | None = None, offset_hours: float = 0.0) -> pd.DataFrame:
    if block_width_hours is None:
        block_width_hours = horizon_hours
    pos = df[df["PACRRatio"] > 0].copy()
    Xp = pos[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            Xp[c] = (Xp[c] - Xp[c].mean()) / Xp[c].std()
    Xp = sm.add_constant(Xp)
    yp = pos["PACRRatio"]
    groups = cluster_ids(pos.index, block_width_hours, offset_hours, anchor=df.index.min())
    unique_groups = np.unique(groups)
    group_rowidx = {g: np.where(groups == g)[0] for g in unique_groups}
    n_target = len(pos)
    rng = np.random.default_rng(seed)

    rows = []
    for tau in taus:
        point = _fit_quantreg(Xp, yp, tau, predictor_cols)
        boot_coefs = {c: [] for c in (["const"] + predictor_cols)}
        for _ in range(n_boot):
            picked_rows = []
            while sum(len(r) for r in picked_rows) < n_target:
                g = rng.choice(unique_groups)
                picked_rows.append(group_rowidx[g])
            sel = np.concatenate(picked_rows)[:n_target]
            Xb = Xp.iloc[sel].reset_index(drop=True)
            yb = yp.iloc[sel].reset_index(drop=True)
            fit_b = _fit_quantreg(Xb, yb, tau, predictor_cols)
            if fit_b is None:
                continue
            for c in boot_coefs:
                boot_coefs[c].append(fit_b[c])
        row = {"tau": tau, "n": len(pos), "n_boot_ok": len(boot_coefs["const"])}
        for c in predictor_cols:
            arr = np.array(boot_coefs[c])
            se_boot = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
            coef = point[c] if point else np.nan
            z = coef / se_boot if se_boot and se_boot > 0 else np.nan
            p_boot = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
            ci_lo = float(np.percentile(arr, 2.5)) if len(arr) > 1 else np.nan
            ci_hi = float(np.percentile(arr, 97.5)) if len(arr) > 1 else np.nan
            row[c] = coef
            row[f"{c}_se_boot"] = se_boot
            row[f"{c}_p_boot"] = p_boot
            row[f"{c}_ci_lo_pct"] = ci_lo
            row[f"{c}_ci_hi_pct"] = ci_hi
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")

    h3_predictors = ["paying_side", "adverse_move", "paying_x_adverse"]
    h4_predictors = ce.MARKET_CONDITION_PREDICTORS

    for h in ce.HORIZONS_HOURS:
        df = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)

        print(f"\n=== H{h}: H3 participation, cluster-robust (clusters = {h}h blocks) ===", flush=True)
        h3_part = cluster_robust_logit(df, h3_predictors, h, standardise=False)
        pd.DataFrame([{"horizon": h, **h3_part}]).to_csv(args.out / f"h3_participation_robust_H{h}.csv", index=False)
        print({k: round(v, 6) if isinstance(v, float) else v for k, v in h3_part.items()})

        print(f"=== H{h}: H3 severity, block bootstrap (n_boot={N_BOOT}) ===", flush=True)
        h3_sev = block_bootstrap_quantreg(df, h3_predictors, h, ce.QUANTILE_TAUS, standardise=False)
        h3_sev.insert(0, "horizon", h)
        h3_sev.to_csv(args.out / f"h3_severity_robust_H{h}.csv", index=False)
        print(h3_sev.to_string(index=False))

        print(f"=== H{h}: H4 participation, cluster-robust (clusters = {h}h blocks) ===", flush=True)
        h4_part = cluster_robust_logit(df, h4_predictors, h, standardise=True)
        pd.DataFrame([{"horizon": h, **h4_part}]).to_csv(args.out / f"h4_participation_robust_H{h}.csv", index=False)
        print({k: round(v, 6) if isinstance(v, float) else v for k, v in h4_part.items()})

        print(f"=== H{h}: H4 severity, block bootstrap (n_boot={N_BOOT}) ===", flush=True)
        h4_sev = block_bootstrap_quantreg(df, h4_predictors, h, ce.QUANTILE_TAUS, standardise=True)
        h4_sev.insert(0, "horizon", h)
        h4_sev.to_csv(args.out / f"h4_severity_robust_H{h}.csv", index=False)
        print(h4_sev.to_string(index=False))

    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
