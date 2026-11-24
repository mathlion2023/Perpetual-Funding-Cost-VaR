"""Q4 extension driver (Tina, 2026-10-07 email): "Does volatility information
improve prediction of future collateral requirements?"

Three pieces, all on non-overlapping (horizon-spaced) origins, chronological
60/20/20 train/val/test split (analysis/chronological_split.py), evaluated
only on the held-out test period after fitting on train+val:

1. Volatility-regime stratification (low/med/high terciles of rv_primary,
   thresholds fixed from TRAIN only).
2. Four-way information-set comparison for the shared participation+severity
   collateral-prediction method: none (baseline), historical (rv_primary),
   GARCH, HAR -- out-of-sample AUC/Brier (participation) and pinball loss
   (severity) on the test period.
3. Nested rough-volatility test: rough added on top of whichever of
   {historical, GARCH, HAR} has the best test-period severity (tau=0.95)
   pinball loss.

Usage:
    python -m analysis.run_vol_extension --out analysis/output/vol_extension_2026_10_07
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import config
from . import vol_extension as ve
from .chronological_split import chronological_split
from .models import base
from .models import volatility_features as vf

INFO_SETS = ["none", "historical", "garch", "har"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")
    split = chronological_split(panel.index, train_frac=0.6, val_frac=0.2)
    print(f"Split boundaries: train_end={split['train_end']}, val_end={split['val_end']}", flush=True)

    comparison_rows = []
    regime_rows = []
    nested_rows = []

    for h in config.HORIZONS_HOURS:
        origins_all = base.non_overlapping_origins(panel.index, h, config.MIN_TRAIN_HOURS)
        origins_train = origins_all.intersection(split["train"])
        origins_val = origins_all.intersection(split["val"])
        origins_test = origins_all.intersection(split["test"])
        origins_fit = origins_train.union(origins_val)
        print(f"\n[{h}h] origins: all={len(origins_all)}, train={len(origins_train)}, "
              f"val={len(origins_val)}, test={len(origins_test)}", flush=True)

        t0 = time.time()
        garch_feat = vf.garch_vol_feature(panel, origins_all)
        har_feat = vf.har_vol_feature(panel, origins_all)
        rough_feat = vf.rough_vol_feature(panel, origins_all)
        print(f"  [{h}h] walk-forward vol features: {time.time()-t0:.1f}s "
              f"(garch n_nan={garch_feat.isna().sum()}, har n_nan={har_feat.isna().sum()}, "
              f"rough n_nan={rough_feat.isna().sum()})", flush=True)

        regime_rows.append(ve.regime_stratification(panel, h, split, origins_all))

        feature_map = {"none": None, "historical": "rv_primary", "garch": garch_feat, "har": har_feat}

        # Common-sample fix (peer-review R2): build every info-set's fit/test frame
        # first, then restrict ALL FOUR to the intersection of origins usable by
        # every one of them, so the comparison is on one shared denominator per
        # horizon rather than each info-set's own largest available sample (the
        # same fix already applied to H5 in Topic1_Collateral_Erosion_Pilot_Results
        # Section 15.5b, for the same reason: HAR's 720h rolling-window requirement
        # drops a few more origins than none/historical/garch).
        raw_fit = {}
        raw_test = {}
        for name, feature in feature_map.items():
            raw_fit[name] = ve.build_dataset(panel, h, feature, origins=origins_fit)
            raw_test[name] = ve.build_dataset(panel, h, feature, origins=origins_test)

        common_fit_idx = raw_fit["none"].index
        common_test_idx = raw_test["none"].index
        for name in feature_map:
            common_fit_idx = common_fit_idx.intersection(raw_fit[name].index)
            common_test_idx = common_test_idx.intersection(raw_test[name].index)
        per_info_test_n = {k: len(v) for k, v in raw_test.items()}
        print(f"  [{h}h] per-info-set n_test (before common-sample restriction): {per_info_test_n}", flush=True)
        print(f"  [{h}h] common sample across all four info sets: n_fit={len(common_fit_idx)}, n_test={len(common_test_idx)}",
              flush=True)

        fits = {}
        for name, feature in feature_map.items():
            predictor_cols = ve.BASELINE_PREDICTORS + ([] if feature is None else ["vol"])
            df_fit = raw_fit[name].loc[common_fit_idx]
            df_test = raw_test[name].loc[common_test_idx]
            fit = ve.fit_hurdle_oos(df_fit, predictor_cols)
            perf = ve.evaluate_oos(fit, df_test)
            fits[name] = fit
            comparison_rows.append({"horizon": h, "info_set": name, **perf})
            print(f"  [{h}h] {name}: n_fit={fit['n_fit']}, {perf}", flush=True)

        conventional = [r for r in comparison_rows if r["horizon"] == h and r["info_set"] in ("historical", "garch", "har")]
        best = min(conventional, key=lambda r: (pd.isna(r["pinball_tau0.95"]), r["pinball_tau0.95"]))
        best_name = best["info_set"]
        best_feature = feature_map[best_name]
        print(f"  [{h}h] best conventional by test tau=0.95 pinball loss: {best_name}", flush=True)

        vol_features_nested = {best_name: best_feature if isinstance(best_feature, str) else best_feature,
                                "rough": rough_feat}
        predictor_cols_nested = ve.BASELINE_PREDICTORS + [best_name, "rough"]
        df_fit_n = ve.build_dataset_multi(panel, h, vol_features_nested, origins=origins_fit)
        df_test_n = ve.build_dataset_multi(panel, h, vol_features_nested, origins=origins_test)
        fit_n = ve.fit_hurdle_oos(df_fit_n, predictor_cols_nested)
        perf_n = ve.evaluate_oos(fit_n, df_test_n)
        rough_coef = {f"rough_coef_tau{tau}": float(fit_n["qr"][tau].params.get("rough", float("nan")))
                      for tau in ve.QUANTILE_TAUS}
        rough_p = {f"rough_p_tau{tau}": float(fit_n["qr"][tau].pvalues.get("rough", float("nan")))
                   for tau in ve.QUANTILE_TAUS}
        nested_rows.append({"horizon": h, "anchor": best_name, "n_fit": fit_n["n_fit"],
                             **{f"anchor_{k}": v for k, v in best.items() if k not in ("horizon", "info_set")},
                             **{f"nested_{k}": v for k, v in perf_n.items()},
                             **rough_coef, **rough_p})
        print(f"  [{h}h] nested ({best_name}+rough): {perf_n}", flush=True)

    comparison_df = pd.DataFrame(comparison_rows)
    regime_df = pd.concat(regime_rows, ignore_index=True)
    nested_df = pd.DataFrame(nested_rows)

    comparison_df.to_csv(args.out / "vol_info_set_comparison.csv", index=False)
    regime_df.to_csv(args.out / "vol_regime_stratification.csv", index=False)
    nested_df.to_csv(args.out / "vol_rough_nested_test.csv", index=False)

    with pd.option_context("display.width", 240, "display.float_format", "{:.4g}".format):
        print("\n=== Volatility-regime stratification ===")
        print(regime_df.to_string(index=False))
        print("\n=== Information-set comparison ===")
        print(comparison_df.to_string(index=False))
        print("\n=== Nested rough-volatility test ===")
        print(nested_df.to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
