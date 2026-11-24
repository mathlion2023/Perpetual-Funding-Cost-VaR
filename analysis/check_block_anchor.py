"""One-off verification for peer-review item R4: does anchoring the H3/H4
severity block-bootstrap's clusters on the full calendar-time series (not the
already-PACRRatio>0-filtered subsequence) change any of Table 10's
stable/not-stable classifications?

Note (2026-10-19): `run_collateral_erosion_robust.py::block_bootstrap_quantreg`
was already fixed on 2026-10-07 to anchor on `df.index.min()` (the H3/H4
dataset's own first timestamp), not the PACRRatio>0-filtered subsample's start
this docstring originally described. This script's "original" spec anchored
on `panel.index.min()` instead -- a third reference point, 24h earlier than
`df.index.min()` -- never reconciled with production until now. It is fixed
to anchor on `df.index.min()`, matching production exactly.

Usage:
    python -m analysis.check_block_anchor
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from . import collateral_erosion as ce

N_BOOT = 1000
SEED = 0

SPECS = {
    "original": dict(width_mult=1.0, offset_mult=0.0),
    "half_width": dict(width_mult=0.5, offset_mult=0.0),
    "double_width": dict(width_mult=2.0, offset_mult=0.0),
    "half_offset": dict(width_mult=1.0, offset_mult=0.5),
}

TARGET_CELLS = [
    # (model, horizon, tau, predictor) -- the cells Table 10 actually reports
    ("h3", 72, None, "adverse_move_participation"),
    ("h3", 168, None, "paying_side_participation"),
    ("h3", 72, 0.50, "paying_side"),
    ("h3", 72, 0.90, "paying_side"),
    ("h3", 168, 0.90, "paying_side"),
    ("h3", "both", 0.95, "paying_side"),
    ("h3", "both", None, "adverse_move_severity_90_95"),
    ("h3", 72, 0.95, "paying_x_adverse"),
    ("h4", "both", None, "rv_primary_participation"),
    ("h4", 72, 0.50, "ofi"),
    ("h4", 168, 0.50, "volume_shock"),
]


def cluster_ids_anchored(index: pd.DatetimeIndex, anchor: pd.Timestamp, block_width_hours: float,
                          offset_hours: float = 0.0) -> np.ndarray:
    elapsed_hours = (index - anchor) / pd.Timedelta(hours=1)
    return ((elapsed_hours + offset_hours) // block_width_hours).astype(int).to_numpy()


def block_bootstrap_quantreg_anchored(df: pd.DataFrame, predictor_cols: list[str], tau: float,
                                       standardise: bool, anchor: pd.Timestamp,
                                       block_width_hours: float, offset_hours: float,
                                       n_boot: int = N_BOOT, seed: int = SEED):
    pos = df[df["PACRRatio"] > 0].copy()
    Xp = pos[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            Xp[c] = (Xp[c] - Xp[c].mean()) / Xp[c].std()
    Xp = sm.add_constant(Xp)
    yp = pos["PACRRatio"]
    groups = cluster_ids_anchored(pos.index, anchor, block_width_hours, offset_hours)
    unique_groups = np.unique(groups)
    group_rowidx = {g: np.where(groups == g)[0] for g in unique_groups}
    n_target = len(pos)
    rng = np.random.default_rng(seed)

    point = QuantReg(yp, Xp).fit(q=tau, max_iter=5000)
    boot_coefs = {c: [] for c in (["const"] + predictor_cols)}
    for _ in range(n_boot):
        picked_rows = []
        while sum(len(r) for r in picked_rows) < n_target:
            g = rng.choice(unique_groups)
            picked_rows.append(group_rowidx[g])
        sel = np.concatenate(picked_rows)[:n_target]
        Xb = Xp.iloc[sel].reset_index(drop=True)
        yb = yp.iloc[sel].reset_index(drop=True)
        try:
            fit_b = QuantReg(yb, Xb).fit(q=tau, max_iter=5000)
        except Exception:
            continue
        for c in boot_coefs:
            boot_coefs[c].append(float(fit_b.params[c]))

    from scipy.stats import norm
    out = {}
    for c in predictor_cols:
        arr = np.array(boot_coefs[c])
        se_boot = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
        coef = float(point.params[c])
        z = coef / se_boot if se_boot and se_boot > 0 else np.nan
        p_boot = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        out[c] = {"coef": coef, "p": p_boot}
    return out


def cluster_robust_logit_anchored(df: pd.DataFrame, predictor_cols: list[str], standardise: bool,
                                   anchor: pd.Timestamp, block_width_hours: float, offset_hours: float):
    X = df[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = sm.add_constant(X)
    y = (df["PACRRatio"] > 0).astype(float)
    groups = cluster_ids_anchored(df.index, anchor, block_width_hours, offset_hours)
    fit = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    return {c: {"coef": float(fit.params[c]), "p": float(fit.pvalues[c])} for c in predictor_cols}


def main():
    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")

    h3_predictors = ["paying_side", "adverse_move", "paying_x_adverse"]
    h4_predictors = ce.MARKET_CONDITION_PREDICTORS

    rows = []
    for h in ce.HORIZONS_HOURS:
        df = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)
        # Anchor on df.index.min() (the H3/H4 dataset's own first timestamp),
        # matching run_collateral_erosion_robust.py's current cluster_ids(...,
        # anchor=df.index.min()) exactly -- not panel.index.min(), which this
        # script used until 2026-10-19. Production was fixed on 2026-10-07
        # (see cluster_ids's docstring) to anchor on the full H3/H4 dataset
        # rather than the PACRRatio>0-filtered subsample; this script's prior
        # "original" spec anchored on the raw panel instead, a third, never
        # reconciled reference point one day (24h) earlier.
        full_anchor = df.index.min()
        for spec_name, spec in SPECS.items():
            bw = h * spec["width_mult"]
            off = h * spec["offset_mult"]

            part_h3 = cluster_robust_logit_anchored(df, h3_predictors, False, full_anchor, bw, off)
            part_h4 = cluster_robust_logit_anchored(df, h4_predictors, True, full_anchor, bw, off)
            for pred, v in part_h3.items():
                rows.append({"model": "h3_participation", "horizon": h, "spec": spec_name, "tau": None,
                             "predictor": pred, "anchored_p": v["p"], "anchored_coef": v["coef"]})
            for pred, v in part_h4.items():
                rows.append({"model": "h4_participation", "horizon": h, "spec": spec_name, "tau": None,
                             "predictor": pred, "anchored_p": v["p"], "anchored_coef": v["coef"]})

            for tau in ce.QUANTILE_TAUS:
                sev_h3 = block_bootstrap_quantreg_anchored(df, h3_predictors, tau, False, full_anchor, bw, off)
                sev_h4 = block_bootstrap_quantreg_anchored(df, h4_predictors, tau, True, full_anchor, bw, off)
                for pred, v in sev_h3.items():
                    rows.append({"model": "h3_severity", "horizon": h, "spec": spec_name, "tau": tau,
                                 "predictor": pred, "anchored_p": v["p"], "anchored_coef": v["coef"]})
                for pred, v in sev_h4.items():
                    rows.append({"model": "h4_severity", "horizon": h, "spec": spec_name, "tau": tau,
                                 "predictor": pred, "anchored_p": v["p"], "anchored_coef": v["coef"]})
            print(f"done: h={h} spec={spec_name}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("analysis/output/block_anchor_check.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
