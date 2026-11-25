"""Adaptive-conformal recalibration layer (Section 4.6.6 / recalibration.py)
applied to the corrected 2022-2025 backtests from the 2026-11-06 lag/gap
audit (full_run_2026_11_06_lagfix_2022_2025). No recalibration run exists
yet on this corrected data (the only prior recalibration runs, under
full_run_2026_09_29_corrected and new_models_2026_09_28, predate both the
2026-10-20 f_eff field-swap fix and this lag fix). This script only
post-processes the already-computed backtest_<model>_H{h}.csv files in that
run directory -- it does not refit any model.

Usage:
    python -m analysis.run_recalibration_lagfix_2022_2025 --out analysis/output/collateral_erosion_2026_11_25_lagfix_2022_2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import backtesting, config, recalibration

BASE_RUN_DIR = Path("analysis/output/full_run_2026_11_06_lagfix_2022_2025")
MODELS = ["historical_simulation", "direct_quantile", "caviar", "har_quantile", "arx_garch_t",
          "gas_t", "ms_garch_t", "sv_t", "svj_t", "rough_vol", "stationary_bootstrap"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for H in config.HORIZONS_HOURS:
        for name in MODELS:
            f = BASE_RUN_DIR / f"H{H}" / f"backtest_{name}_H{H}.csv"
            res = pd.read_csv(f, index_col="origin", parse_dates=["origin"])
            raw_r = backtesting.backtest_report(res, config.VAR_ALPHA)
            for gamma in recalibration.DEFAULT_GAMMAS:
                rec = recalibration.aci_recalibrate(res, config.VAR_ALPHA, gamma=gamma)
                rec.to_csv(args.out / f"recalibrated_{name}_H{H}_gamma{gamma}.csv")
                rec_r = backtesting.backtest_report(rec, config.VAR_ALPHA)
                rows.append({
                    "horizon": H, "model": name, "gamma": gamma, "n": rec_r["n_forecasts"],
                    "exceptions_raw": raw_r["realized_exceptions"], "exceptions_recal": rec_r["realized_exceptions"],
                    "kupiec_p_raw": raw_r["kupiec_uc"]["pvalue"], "kupiec_p_recal": rec_r["kupiec_uc"]["pvalue"],
                    "cc_p_raw": raw_r["christoffersen_conditional_coverage"]["pvalue"],
                    "cc_p_recal": rec_r["christoffersen_conditional_coverage"]["pvalue"],
                    "qloss_raw": raw_r["quantile_loss"], "qloss_recal": rec_r["quantile_loss"],
                })
            print(f"[H={H}] {name} done", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "recalibration_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(summary[summary["gamma"] == 0.02].to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
