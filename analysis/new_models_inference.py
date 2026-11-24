"""Inference and robustness checks for the 2026-09-28 model additions.

Three checks, the first two matching methods already used for the original
seven models in Topic1_Full_History_Results_Report.md:

1. Corrected Diebold-Mariano comparisons (report Section 16 R2 convention):
   every model pair at both horizons, check-loss differential on the pair's
   common origins, 2 unweighted autocovariance lags
   (backtesting.diebold_mariano_test with h=3), then Benjamini-Hochberg FDR
   within each horizon (see dm_family for why "within" rather than pooled).
   Reported twice: for the original 7-model family (21 pairs per horizon,
   reproducing the published R2 files exactly as a check on this code) and
   for the full 11-model family (55 pairs per horizon).

2. Common-sample robustness (report Section 6.3 convention): every model
   restricted to the origins where all compared models produced a forecast,
   both for all 11 models and excluding arx_garch_t (whose stability guard
   leaves it only ~45% of origins and would halve the common sample).
   Raw and ACI-recalibrated (gamma = 0.02) forecasts are both scored.

3. Is each model's recalibration gain significant? Same DM test, recalibrated
   vs raw forecasts of the same model on the same origins, FDR within
   horizon across the 11 models.

Usage:
    python -m analysis.new_models_inference
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import backtesting, config, recalibration

OUT = config.OUTPUT_DIR
# Environment overrides (as in rerun_2026_09_27.py) let this run on a newer
# base run without overwriting the published outputs.
BASE_RUN = Path(os.environ.get("FCVAR_BASE_RUN", OUT / "full_run_2026_09_28"))
NEW_RUN = Path(os.environ.get("FCVAR_NEW_RUN", OUT / "new_models_2026_09_28"))
# The Section 16 R2 reproduction check only makes sense on the pre-forward_sum-fix base run.
CHECK_R2_REPRODUCTION = "FCVAR_BASE_RUN" not in os.environ
ORIGINAL = ["historical_simulation", "direct_quantile", "caviar", "arx_garch_t", "sv_t", "svj_t", "rough_vol"]
NEW = ["har_quantile", "gas_t", "ms_garch_t", "stationary_bootstrap"]
HAC_LAGS = 2
RECAL_GAMMA = 0.02


def load_backtests(H: int) -> dict[str, pd.DataFrame]:
    res = {}
    for m in ORIGINAL:
        res[m] = pd.read_csv(BASE_RUN / f"H{H}" / f"backtest_{m}_H{H}.csv", index_col="origin", parse_dates=["origin"])
    for m in NEW:
        res[m] = pd.read_csv(NEW_RUN / f"H{H}" / f"backtest_{m}_H{H}.csv", index_col="origin", parse_dates=["origin"])
    return res


def dm_hac(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha: float, lags: int) -> dict:
    """The report's R2 convention is backtesting.diebold_mariano_test with
    h = lags + 1, i.e. `lags` unweighted autocovariance terms (verified to
    reproduce the report's published DM2 values exactly). A Bartlett-weighted
    Newey-West variance was tried first and did NOT reproduce them: it gives
    larger |DM| (e.g. 4.01 vs the report's 3.64 for historical_simulation vs
    direct_quantile at H=72)."""
    r = backtesting.diebold_mariano_test(df_a, df_b, alpha, h=lags + 1)
    return {"n": r["n"], "DM": r["DM_stat"], "p": r["pvalue"], "mean_loss_diff": r.get("mean_loss_diff")}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def dm_family(backtests: dict[int, dict[str, pd.DataFrame]], models: list[str]) -> pd.DataFrame:
    rows = []
    for H, res in backtests.items():
        for a, b in itertools.combinations(models, 2):
            r0 = dm_hac(res[a], res[b], config.VAR_ALPHA, 0)
            r2 = dm_hac(res[a], res[b], config.VAR_ALPHA, HAC_LAGS)
            rows.append({"horizon": H, "a": a, "b": b, "n": r2["n"], "DM0": r0["DM"], "p0": r0["p"],
                         "DM2": r2["DM"], "p2": r2["p"]})
    df = pd.DataFrame(rows)
    # Benjamini-Hochberg applied separately within each horizon: this is what
    # the published full_run_svfix/H{h}/dm_matrix_hac_fdr_pairs.csv files do
    # (reproduced to 7e-16), although the report's Section 16 R2 text says the
    # correction was pooled across both horizons. The pooled variant is kept
    # alongside as p2_fdr_pooled / sig_hac_fdr_pooled.
    df["p2_fdr"] = np.nan
    for _, g in df.groupby("horizon"):
        df.loc[g.index, "p2_fdr"] = bh_fdr(g["p2"].to_numpy())
    df["p2_fdr_pooled"] = bh_fdr(df["p2"].to_numpy())
    df["sig_uncorrected"] = df["p0"] < 0.05
    df["sig_hac_only"] = df["p2"] < 0.05
    df["sig_hac_fdr"] = df["p2_fdr"] < 0.05
    df["sig_hac_fdr_pooled"] = df["p2_fdr_pooled"] < 0.05
    return df


def common_sample(res: dict[str, pd.DataFrame], models: list[str], recal: bool) -> pd.DataFrame:
    frames = {}
    for m in models:
        d = res[m].dropna(subset=["var_forecast", "realized"])
        if recal:
            d = recalibration.aci_recalibrate(res[m], config.VAR_ALPHA, gamma=RECAL_GAMMA)
        frames[m] = d
    idx = None
    for d in frames.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    rows = []
    for m, d in frames.items():
        r = backtesting.backtest_report(d.loc[idx], config.VAR_ALPHA)
        rows.append({"model": m, "n": r["n_forecasts"], "exceptions": r["realized_exceptions"],
                     "expected": r["expected_exceptions"], "kupiec_p": r["kupiec_uc"]["pvalue"],
                     "cc_p": r["christoffersen_conditional_coverage"]["pvalue"], "qloss": r["quantile_loss"]})
    return pd.DataFrame(rows)


def main():
    backtests = {H: load_backtests(H) for H in config.HORIZONS_HOURS}
    out_dir = NEW_RUN / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corrected DM: original family first, as a reproduction check on the report's Section 16 R2.
    fam7 = dm_family(backtests, ORIGINAL)
    fam7.to_csv(out_dir / "dm_hac_fdr_pairs_original7.csv", index=False)
    if CHECK_R2_REPRODUCTION:
        ref = pd.concat([pd.read_csv(OUT / "full_run_svfix" / f"H{H}" / "dm_matrix_hac_fdr_pairs.csv") for H in config.HORIZONS_HOURS])
        chk = fam7.merge(ref, on=["horizon", "a", "b"], suffixes=("", "_ref"))
        print(f"Reproduction check vs report R2 ({len(chk)} pairs): "
              f"max |DM2 diff| = {np.abs(chk.DM2 - chk.DM2_ref).max():.2e}, "
              f"max |p2_fdr diff| = {np.abs(chk.p2_fdr - chk.p2_fdr_ref).max():.2e}, "
              f"significance calls identical: {bool((chk.sig_hac_fdr == chk.sig_hac_fdr_ref).all())}")
    else:
        print(f"Base run {BASE_RUN} (override): Section 16 R2 reproduction check skipped.")

    fam11 = dm_family(backtests, ORIGINAL + NEW)
    fam11.to_csv(out_dir / "dm_hac_fdr_pairs_all11.csv", index=False)
    summary = {}
    for H in config.HORIZONS_HOURS:
        f = fam11[fam11.horizon == H]
        summary[f"H{H}"] = {"pairs": len(f), "sig_uncorrected": int(f.sig_uncorrected.sum()), "sig_hac_fdr_pooled": int(f.sig_hac_fdr_pooled.sum()),
                            "sig_hac_only": int(f.sig_hac_only.sum()), "sig_hac_fdr": int(f.sig_hac_fdr.sum())}
    print("\n11-model family (55 pairs per horizon; FDR within horizon, pooled variant also counted):", json.dumps(summary))
    involving_new = fam11[fam11.a.isin(NEW) | fam11.b.isin(NEW)]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(involving_new[["horizon", "a", "b", "n", "DM0", "DM2", "p2", "p2_fdr", "sig_hac_fdr"]].to_string(index=False))

    # 2. Common-sample robustness.
    for H, res in backtests.items():
        for label, models in [("all11", ORIGINAL + NEW), ("excl_arx", [m for m in ORIGINAL + NEW if m != "arx_garch_t"])]:
            for recal in (False, True):
                cs = common_sample(res, models, recal)
                tag = f"common_sample_{label}_{'recal' if recal else 'raw'}_H{H}.csv"
                cs.to_csv(out_dir / tag, index=False)
                if label == "excl_arx":
                    print(f"\n--- H{H} common sample excl. arx_garch_t, {'recalibrated (gamma=0.02)' if recal else 'raw'}: n={cs.n.iloc[0]}")
                    with pd.option_context("display.float_format", "{:.4g}".format):
                        print(cs.to_string(index=False))
                else:
                    print(f"(H{H} common sample, all 11 models: n={cs.n.iloc[0]})")

    # 3. Recalibrated vs raw, per model.
    rows = []
    for H, res in backtests.items():
        for m, bt in res.items():
            rec = recalibration.aci_recalibrate(bt, config.VAR_ALPHA, gamma=RECAL_GAMMA)
            r = dm_hac(rec, bt, config.VAR_ALPHA, HAC_LAGS)
            rows.append({"horizon": H, "model": m, "n": r["n"], "DM2_recal_vs_raw": r["DM"], "p2": r["p"]})
    rg = pd.DataFrame(rows)
    rg["p2_fdr"] = np.nan
    for _, g in rg.groupby("horizon"):
        rg.loc[g.index, "p2_fdr"] = bh_fdr(g["p2"].to_numpy())
    rg["sig_hac_fdr"] = rg["p2_fdr"] < 0.05
    rg.to_csv(out_dir / "recalibration_gain_dm.csv", index=False)
    print("\n--- Recalibration gain (negative DM => recalibrated has lower loss)")
    with pd.option_context("display.float_format", "{:.3g}".format):
        print(rg.to_string(index=False))

    with open(out_dir / "dm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll output written to {out_dir}")


if __name__ == "__main__":
    main()
