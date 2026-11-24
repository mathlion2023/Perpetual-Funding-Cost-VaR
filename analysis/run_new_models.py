"""Backtest the Stage 3 models added to the proposal on 2026-09-28
(Sections 4.6.4-4.6.6) against the existing seven, without rebuilding the
panel.

Reuses the panel.parquet and backtest_<model>_H{h}.csv files written by a
completed run_pipeline.py run, so the new models are scored on exactly the
same panel, target, and forecast origins as the original seven.

Usage:
    python -m analysis.run_new_models --horizon 72 \
        --base-run analysis/output/full_run_2026_09_28/H72 \
        --out analysis/output/new_models_2026_09_28/H72
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import backtesting, config, recalibration
from .models import base, block_bootstrap, gas, har_quantile, ms_garch

NEW_MODEL_REGISTRY = {
    # name: (model_fn, model_kwargs, window_hours) -- same convention as run_pipeline.MODEL_REGISTRY
    "har_quantile": (har_quantile.har_quantile_var, {}, None),
    "gas_t": (gas.gas_t_var, {"n_sims": 1000}, None),
    "ms_garch_t": (ms_garch.ms_garch_t_var, {"n_sims": 1000}, None),
    "stationary_bootstrap": (block_bootstrap.stationary_bootstrap_var, {"n_sims": 5000}, None),
}

ORIGINAL_MODELS = ["historical_simulation", "direct_quantile", "caviar", "arx_garch_t", "sv_t", "svj_t", "rough_vol"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, required=True, choices=config.HORIZONS_HOURS)
    ap.add_argument("--base-run", type=Path, required=True, help="A completed run_pipeline.py output dir for this horizon.")
    ap.add_argument("--models", nargs="+", default=list(NEW_MODEL_REGISTRY), choices=list(NEW_MODEL_REGISTRY))
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    H = args.horizon
    target_col = f"C_long_H{H}"

    panel = pd.read_parquet(args.base_run / "panel.parquet")
    print(f"panel: {panel.shape[0]} hours, from {args.base_run}", flush=True)

    results: dict[str, pd.DataFrame] = {}
    for name in ORIGINAL_MODELS:
        f = args.base_run / f"backtest_{name}_H{H}.csv"
        if f.exists():
            results[name] = pd.read_csv(f, index_col="origin", parse_dates=["origin"])

    # New models not being re-run this call keep their previous backtest, so a
    # partial re-run (e.g. --models har_quantile) still reports all models.
    for name in NEW_MODEL_REGISTRY:
        f = args.out / f"backtest_{name}_H{H}.csv"
        if name not in args.models and f.exists():
            results[name] = pd.read_csv(f, index_col="origin", parse_dates=["origin"])

    print(f"\n=== New Stage 3 models: horizon={H}h, models={args.models} ===", flush=True)
    for name in args.models:
        fn, kwargs, window = NEW_MODEL_REGISTRY[name]
        t0 = time.time()
        res = base.run_rolling_backtest(panel, fn, target_col, H, min_train_hours=args.min_train_hours,
                                        window_hours=window, **kwargs)
        print(f"  [{name}] {len(res)} origins, {res['var_forecast'].notna().sum()} successful forecasts, "
              f"{time.time() - t0:.1f}s", flush=True)
        errs = res["error"].dropna()
        if len(errs):
            print(f"    first error: {errs.iloc[0]}", flush=True)
        res.to_csv(args.out / f"backtest_{name}_H{H}.csv")
        results[name] = res

    print("\n=== Backtest reports (all models) ===")
    reports = {}
    for name, res in results.items():
        reports[name] = backtesting.backtest_report(res, config.VAR_ALPHA)
        r = reports[name]
        print(f"  [{name}] exceptions {r['realized_exceptions']}/{r['n_forecasts']} (expected {r['expected_exceptions']:.1f}); "
              f"Kupiec p={r['kupiec_uc']['pvalue']:.3f}; cond. cov. p={r['christoffersen_conditional_coverage']['pvalue']:.3f}; "
              f"quantile loss={r['quantile_loss']:.6g}")
    with open(args.out / "backtest_reports.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)

    dm = backtesting.compare_models(results, config.VAR_ALPHA)
    dm.to_csv(args.out / "diebold_mariano_matrix.csv")
    print("\nDiebold-Mariano matrix (row vs column, +ve => row worse):")
    print(dm.round(2).to_string())

    print("\n=== Section 4.6.6 recalibration layer (adaptive conformal inference) ===")
    rows = []
    for gamma in recalibration.DEFAULT_GAMMAS:
        for name, res in results.items():
            rec = recalibration.aci_recalibrate(res, config.VAR_ALPHA, gamma=gamma)
            rec.to_csv(args.out / f"recalibrated_{name}_H{H}_gamma{gamma}.csv")
            raw_r, rec_r = reports[name], backtesting.backtest_report(rec, config.VAR_ALPHA)
            rows.append({
                "model": name, "gamma": gamma, "n": rec_r["n_forecasts"],
                "exceptions_raw": raw_r["realized_exceptions"], "exceptions_recal": rec_r["realized_exceptions"],
                "kupiec_p_raw": raw_r["kupiec_uc"]["pvalue"], "kupiec_p_recal": rec_r["kupiec_uc"]["pvalue"],
                "cc_p_raw": raw_r["christoffersen_conditional_coverage"]["pvalue"],
                "cc_p_recal": rec_r["christoffersen_conditional_coverage"]["pvalue"],
                "qloss_raw": raw_r["quantile_loss"], "qloss_recal": rec_r["quantile_loss"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "recalibration_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(summary[summary["gamma"] == 0.02].to_string(index=False))

    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
