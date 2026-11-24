"""Block-length sensitivity check for the H3/H4 serial-correlation correction.

Addresses Tina's 2026-10-07 email: "please confirm that sensitivity to block
length has been checked. Horizon-width blocks address substantial dependence,
but observations near adjacent block boundaries can still share future
outcomes. Non-overlapping evaluation windows also do not automatically
establish independence."

run_collateral_erosion_robust.py's cluster_ids() used exactly one partition
(block width = horizon, offset = 0, anchored at the series' first timestamp).
This script re-derives the same cluster-robust logit (participation) and
block-bootstrap quantile regression (severity) fits under three alternative
partitions, holding everything else identical, and reports whether the
report's significance conclusions are stable:

  original    : block_width = horizon,   offset = 0   (as in the main report)
  half_width  : block_width = horizon/2, offset = 0
  double_width: block_width = 2*horizon, offset = 0
  half_offset : block_width = horizon,   offset = horizon/2

half_width/double_width test sensitivity to the arbitrary choice of block
width; half_offset tests sensitivity to the arbitrary choice of where the
partition starts (Tina's "boundary-adjacent origins" concern) while holding
width fixed.

Usage:
    python -m analysis.run_collateral_erosion_block_sensitivity --out analysis/output/collateral_erosion_2026_10_07_block_sensitivity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import collateral_erosion as ce
from .run_collateral_erosion_robust import cluster_robust_logit, block_bootstrap_quantreg

SPECS = {
    "original": dict(width_mult=1.0, offset_mult=0.0),
    "half_width": dict(width_mult=0.5, offset_mult=0.0),
    "double_width": dict(width_mult=2.0, offset_mult=0.0),
    "half_offset": dict(width_mult=1.0, offset_mult=0.5),
}


def run_one_spec(df: pd.DataFrame, predictor_cols: list[str], h: int, standardise: bool,
                  spec_name: str, width_mult: float, offset_mult: float) -> tuple[dict, pd.DataFrame]:
    block_width = h * width_mult
    offset = h * offset_mult
    part = cluster_robust_logit(df, predictor_cols, h, standardise,
                                 block_width_hours=block_width, offset_hours=offset)
    sev = block_bootstrap_quantreg(df, predictor_cols, h, ce.QUANTILE_TAUS, standardise,
                                    block_width_hours=block_width, offset_hours=offset)
    part["spec"] = spec_name
    sev.insert(0, "spec", spec_name)
    return part, sev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")

    models = {
        "h3": (["paying_side", "adverse_move", "paying_x_adverse"], False),
        "h4": (ce.MARKET_CONDITION_PREDICTORS, True),
    }

    for h in ce.HORIZONS_HOURS:
        for model_name, (predictor_cols, standardise) in models.items():
            df = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)
            part_rows, sev_rows = [], []
            for spec_name, spec in SPECS.items():
                print(f"=== H{h} {model_name} spec={spec_name} ===", flush=True)
                part, sev = run_one_spec(df, predictor_cols, h, standardise, spec_name, **spec)
                part_rows.append(part)
                sev_rows.append(sev)
            part_df = pd.DataFrame(part_rows)
            sev_df = pd.concat(sev_rows, ignore_index=True)
            part_df.insert(0, "horizon", h)
            sev_df.insert(0, "horizon", h)
            part_df.to_csv(args.out / f"{model_name}_participation_block_sensitivity_H{h}.csv", index=False)
            sev_df.to_csv(args.out / f"{model_name}_severity_block_sensitivity_H{h}.csv", index=False)
            print(part_df.to_string(index=False))
            print(sev_df.to_string(index=False))

    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
