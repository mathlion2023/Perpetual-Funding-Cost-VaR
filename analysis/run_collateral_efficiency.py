"""Q5 driver (Tina, 2026-10-07 email): "Can funding and volatility
information provide similar protection with less collateral, or better
protection with the same collateral?"

Chronological 60/20/20 split (shared with Q4), non-overlapping origins,
calibrated on train+val, evaluated on held-out test only. Five models
(price_only / +funding / +volatility / +funding+volatility / +funding
+volatility+rough), each leverage (10x, 20x) and horizon (72h, 168h)
separately, each at a quantile grid (0.90/0.95/0.975/0.99) so "comparable
protection target" (same tau) and "comparable collateral budget" (same
mean reserve, read off the grid) can both be answered. Two predefined
secondary subgroups (high entry-volatility; paying-side-and-high-funding
periods, both defined from information known at the origin) are evaluated
at tau=0.95 with the full metric set, at both leverages.

Usage:
    python -m analysis.run_collateral_efficiency --out analysis/output/collateral_efficiency_2026_10_07
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import collateral_efficiency as cef
from . import config
from .chronological_split import chronological_split
from .models import base
from .models import volatility_features as vf

TAUS = (0.90, 0.95, 0.975, 0.99)
LEVERAGES = (10, 20)
SECONDARY_TAU = 0.95


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")
    split = chronological_split(panel.index, train_frac=0.6, val_frac=0.2)

    main_rows = []
    secondary_rows = []
    diff_rows = []

    for h in config.HORIZONS_HOURS:
        origins_all = base.non_overlapping_origins(panel.index, h, config.MIN_TRAIN_HOURS)
        origins_train = origins_all.intersection(split["train"])
        origins_val = origins_all.intersection(split["val"])
        origins_test = origins_all.intersection(split["test"])
        origins_fit = origins_train.union(origins_val)
        print(f"\n[{h}h] origins: all={len(origins_all)}, fit={len(origins_fit)}, test={len(origins_test)}", flush=True)

        rough_feat = vf.rough_vol_feature(panel, origins_all)

        train_rv = panel.loc[origins_train, "rv_primary"].dropna()
        high_vol_cut = train_rv.quantile(2 / 3)
        high_vol_origins = panel.index[panel["rv_primary"] > high_vol_cut]
        paying_high_funding_origins = panel.index[(panel["f_eff"] > 0) & (panel["high_funding"] == 1)]
        print(f"  [{h}h] high-vol cutoff (train 2/3 quantile of rv_primary) = {high_vol_cut:.5f}", flush=True)

        for leverage in LEVERAGES:
            reserve_by_model_tau = {}
            for model_name, feature_list in cef.MODEL_FEATURES.items():
                extra_features = {}
                if "funding" in feature_list:
                    extra_features["funding"] = "f_eff"
                if "vol" in feature_list:
                    extra_features["vol"] = "rv_primary"
                if "rough" in feature_list:
                    extra_features["rough"] = rough_feat
                predictor_cols = cef.BASELINE_PREDICTORS + feature_list

                df_fit = cef.build_dataset(panel, h, leverage, origins_fit, extra_features)
                df_test = cef.build_dataset(panel, h, leverage, origins_test, extra_features)

                for tau in TAUS:
                    fit = cef.fit_quantile_reserve(df_fit, predictor_cols, tau)
                    reserve_test = cef.predict_reserve(fit, df_test)
                    perf = cef.evaluate_reserve(reserve_test, df_test["PACRRatio"])
                    main_rows.append({"horizon": h, "leverage": leverage, "model": model_name, "tau": tau,
                                       "n_fit": fit["n_fit"], **perf})
                    if tau == SECONDARY_TAU:
                        reserve_by_model_tau[model_name] = reserve_test

                        for subgroup_name, subgroup_origins in (
                            ("high_volatility", high_vol_origins), ("paying_side_high_funding", paying_high_funding_origins),
                        ):
                            idx = reserve_test.index.intersection(subgroup_origins)
                            perf_sub = cef.evaluate_reserve(reserve_test.loc[idx], df_test["PACRRatio"].loc[idx])
                            secondary_rows.append({"horizon": h, "leverage": leverage, "model": model_name,
                                                    "subgroup": subgroup_name, **perf_sub})

                print(f"  [{h}h L{leverage}] {model_name}: n_fit={fit['n_fit']}", flush=True)

            # PACRRatio itself doesn't depend on any model's extra features, only on
            # baseline predictors (bug fix, code audit 2026-11-25: this used to be
            # `df_test["PACRRatio"]` left over from the loop's last iteration, so it
            # was silently bound to whichever model happened to be last in
            # cef.MODEL_FEATURES rather than a model-agnostic series. Built fresh here
            # with no extra features so its index is the superset every per-model
            # reserve series can align against).
            pacr_test_series = cef.build_dataset(panel, h, leverage, origins_test, {})["PACRRatio"]
            baseline_reserve = reserve_by_model_tau["price_only"]
            for model_name in ("price_funding", "price_vol", "price_funding_vol", "price_funding_vol_rough"):
                for metric in ("shortfall_rate", "mean_unmet_all"):
                    d = cef.bootstrap_diff_ci(baseline_reserve, reserve_by_model_tau[model_name], pacr_test_series, metric)
                    diff_rows.append({"horizon": h, "leverage": leverage, "model": model_name,
                                       "tau": SECONDARY_TAU, "metric": metric, **d})

    main_df = pd.DataFrame(main_rows)
    secondary_df = pd.DataFrame(secondary_rows)
    diff_df = pd.DataFrame(diff_rows)

    main_df.to_csv(args.out / "efficiency_main.csv", index=False)
    secondary_df.to_csv(args.out / "efficiency_secondary.csv", index=False)
    diff_df.to_csv(args.out / "efficiency_diff_vs_price_only.csv", index=False)

    with pd.option_context("display.width", 240, "display.float_format", "{:.4g}".format):
        print("\n=== Main efficiency comparison (all taus) ===")
        print(main_df.to_string(index=False))
        print("\n=== Secondary subgroups (tau=0.95) ===")
        print(secondary_df.to_string(index=False))
        print("\n=== Bootstrap CI, difference vs. price_only (tau=0.95) ===")
        print(diff_df.to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
