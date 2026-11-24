"""Follow-up to rerun_2026_09_27.py, addressing Tina's point: the Section 9
gap Gap_t = VaR_net_t - VaR_price_t should compare two VaR models built on
the SAME forecast dates, model architecture, and information set -- differing
only in whether the target includes funding cost (L_price vs L_net), not
also in which controls are included.

The existing pipeline does not do this: price_only_var_backtest defaults to
BASELINE_PRICE_FEATURES (no ofi, no spread_binc) while net_loss_var_backtest
defaults to SECTION9_CONTROLS (adds ofi and spread_binc). This script refits
VaR_price on SECTION9_CONTROLS (matching VaR_net's information set exactly)
so the gap isolates the effect of the target definition alone, and reports
that alongside the original (mismatched-feature) gap for comparison.

Run after rerun_2026_09_27.py has completed (reuses its price/net-loss
backtests are NOT reused here since VaR_price must be refit on a different
feature set -- only the corrected L_net panel construction is shared).

Usage: python -m analysis.rerun_2026_09_27_matched
"""

from __future__ import annotations

import json

import pandas as pd

from . import config, backtesting, omission_gap, stage1_omission
from .rerun_2026_09_27 import load_panel, gap_economic_magnitude, NEW_OUT


def run_horizon(h: int, panel: pd.DataFrame) -> dict:
    out_dir = NEW_OUT / f"H{h}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"horizon": h}

    print(f"[H{h}] matched-information-set price-only VaR (SECTION9_CONTROLS)...", flush=True)
    price_res_matched = stage1_omission.price_only_var_backtest(
        panel, h, min_train_hours=config.MIN_TRAIN_HOURS, feature_cols=omission_gap.SECTION9_CONTROLS
    )
    price_res_matched.to_csv(out_dir / "price_only_var_backtest_matched_features.csv")
    result["price_only_matched_backtest_report"] = backtesting.backtest_report(price_res_matched, config.VAR_ALPHA)

    net_res = omission_gap.net_loss_var_backtest(panel, h, min_train_hours=config.MIN_TRAIN_HOURS)

    gap_ds_matched = omission_gap.build_gap_dataset(panel, price_res_matched, net_res)
    gap_ds_matched.to_csv(out_dir / "omission_gap_dataset_matched_features.csv")
    result["gap_n_matched"] = len(gap_ds_matched)
    if len(gap_ds_matched) > 10:
        regime = omission_gap.fit_gap_regime_test(gap_ds_matched)
        result["gap_regime_test_matched"] = {
            "delta": regime["delta"],
            "p_one_sided": regime["delta_pvalue_one_sided_H0_delta_le_0"],
        }
        rur_summary = omission_gap.gap_and_rur_summary(gap_ds_matched)
        rur_summary.to_csv(out_dir / "gap_rur_summary_matched_features.csv")
        econ = gap_economic_magnitude(gap_ds_matched)
        econ.to_csv(out_dir / "gap_economic_magnitude_matched_features.csv")

    dm_price_matched_vs_net = backtesting.diebold_mariano_test(price_res_matched, net_res, config.VAR_ALPHA)
    result["dm_price_matched_vs_net"] = dm_price_matched_vs_net
    print(f"  matched-feature gap: n={result['gap_n_matched']}, regime={result.get('gap_regime_test_matched')}", flush=True)

    with open(out_dir / "reanalysis_summary_matched.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


def main():
    panel = load_panel()
    all_results = {}
    for h in config.HORIZONS_HOURS:
        all_results[h] = run_horizon(h, panel)
    with open(NEW_OUT / "reanalysis_summary_matched_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("MATCHED-FEATURE RERUN DONE", flush=True)


if __name__ == "__main__":
    main()
