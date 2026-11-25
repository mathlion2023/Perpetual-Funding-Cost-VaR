"""Re-run of run_collateral_erosion.py (P0-P3 reserve-policy comparison, H1-H4
descriptives) against the corrected full-history panel from the 2026-11-06
lag/gap-alignment audit (Topic1_Code_Audit_Record_2026-11-06.md, section 25
of Topic1_Full_History_Results_Report.md). The original script's panel and
LONG_RUN_DIR both point at pre-audit runs (full_run_fwdsum_fix /
full_run_2026_09_29_corrected); this clone repoints both at
full_run_2026_11_06_lagfix_2022_2025 so the paper's
Table 3 (collateral drain, PACRRatio, reserve shortfalls) uses the same
corrected data as Tables 1-2. No methodology changed from
run_collateral_erosion.py -- only the two input paths.

Usage:
    python -m analysis.run_collateral_erosion_lagfix_fullhistory --out analysis/output/collateral_erosion_2026_11_24_lagfix_fullhistory
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from . import collateral_erosion as ce
from . import config, stage1_omission
from .models import base, caviar, historical_simulation, mean_funding

PANEL_PATH = Path("analysis/output/full_run_2026_11_06_lagfix_2022_2025/H72/panel.parquet")
LONG_RUN_DIR = Path("analysis/output/full_run_2026_11_06_lagfix_2022_2025")


def price_only_var_backtest_short(panel: pd.DataFrame, horizon_hours: int, alpha: float = config.VAR_ALPHA, **kwargs):
    target_col = f"R_H{horizon_hours}"
    return base.run_rolling_backtest(
        panel, caviar.direct_quantile_var, target_col, horizon_hours,
        alpha=alpha, feature_cols=stage1_omission.BASELINE_PRICE_FEATURES, clip_nonnegative=False, **kwargs,
    )


def evaluate_reserve_policy(reserve_frac: pd.Series, pacr_frac: pd.Series) -> dict:
    df = pd.DataFrame({"reserve": reserve_frac, "pacr": pacr_frac}).dropna()
    if not len(df):
        return {"n": 0}
    shortfall = df["pacr"] > df["reserve"]
    severity = (df["pacr"] - df["reserve"])[shortfall]
    return {
        "n": len(df),
        "shortfalls": int(shortfall.sum()),
        "shortfall_rate": float(shortfall.mean()),
        "mean_shortfall_severity_pct_notional": float(severity.mean() * 100) if len(severity) else 0.0,
        "max_shortfall_pct_notional": float(severity.max() * 100) if len(severity) else 0.0,
        "mean_reserve_pct_notional": float(df["reserve"].mean() * 100),
        "p95_reserve_pct_notional": float(df["reserve"].quantile(0.95) * 100),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    print(f"panel: {panel.shape[0]} hours, from {PANEL_PATH}", flush=True)

    print("=== H1: funding erosion of starting collateral ===", flush=True)
    h1 = ce.h1_funding_erosion_summary(panel)
    h1.to_csv(args.out / "h1_funding_erosion_summary.csv", index=False)
    print(h1.to_string(index=False))

    print("\n=== H2: PACR / PACRRatio grid (leverage x horizon x direction) ===", flush=True)
    h2 = ce.h2_pacr_grid_summary(panel)
    h2.to_csv(args.out / "h2_pacr_grid_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print(h2.to_string(index=False))

    print("\n=== H3/H4: hurdle model (participation logit + severity quantile regression) of PACRRatio (long, 10x) ===", flush=True)
    for h in ce.HORIZONS_HOURS:
        df = ce.build_h3_h4_dataset(panel, h, direction="long", leverage=ce.PRIMARY_LEVERAGE)
        h3_part, h3_sev = ce.fit_h3_interaction(df)
        pd.DataFrame([{"horizon": h, **h3_part}]).to_csv(args.out / f"h3_participation_H{h}.csv", index=False)
        h3_sev.insert(0, "horizon", h)
        h3_sev.to_csv(args.out / f"h3_severity_H{h}.csv", index=False)
        h4_part, h4_sev = ce.fit_h4_market_conditions(df)
        pd.DataFrame([{"horizon": h, **h4_part}]).to_csv(args.out / f"h4_participation_H{h}.csv", index=False)
        h4_sev.insert(0, "horizon", h)
        h4_sev.to_csv(args.out / f"h4_severity_H{h}.csv", index=False)
        print(f"\n  H{h} (n={len(df)}, share PACRRatio>0 = {h3_part['share_positive']:.3f}):")
        print("  H3 participation (logit):", {k: round(v, 4) if isinstance(v, float) else v for k, v in h3_part.items()})
        print("  H3 severity | shortfall>0:", h3_sev.to_string(index=False))
        print("  H4 participation (logit):", {k: round(v, 4) if isinstance(v, float) else v for k, v in h4_part.items()})
        print("  H4 severity | shortfall>0:", h4_sev.to_string(index=False))

    print("\n=== Reserve-policy comparison (P0/P1/P2/P3), both directions, primary leverage 10x ===", flush=True)
    rows = []
    for h in ce.HORIZONS_HOURS:
        ds = ce.build_pacr_dataset(panel, h)
        pacr_frac = {
            d: pd.Series(ce.pacr_frac_notional(ds[f"net_max_{d}"].to_numpy(), ds["mark_price_origin"].to_numpy(),
                                                ce.PRIMARY_LEVERAGE),
                          index=ds.index)
            for d in ("long", "short")
        }

        t0 = time.time()
        price_long = stage1_omission.price_only_var_backtest(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        price_short = price_only_var_backtest_short(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        print(f"  [{h}h] price-only VaR (long+short): {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        p1_long = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_long_H{h}", h,
                                             min_train_hours=args.min_train_hours)["var_forecast"]
        p1_short = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        print(f"  [{h}h] P1 mean-funding (long+short): {time.time() - t0:.1f}s", flush=True)

        long_file = LONG_RUN_DIR / f"H{h}" / f"backtest_historical_simulation_H{h}.csv"
        p2_long = pd.read_csv(long_file, parse_dates=["origin"]).set_index("origin")["var_forecast"]
        t0 = time.time()
        p2_short = base.run_rolling_backtest(panel, historical_simulation.historical_simulation_var, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        print(f"  [{h}h] P2 historical-tail short: {time.time() - t0:.1f}s", flush=True)

        p3_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_direct_quantile_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        t0 = time.time()
        p3_short = base.run_rolling_backtest(panel, caviar.direct_quantile_var, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        print(f"  [{h}h] P3 direct-quantile short: {time.time() - t0:.1f}s", flush=True)

        p7_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_svj_t_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]

        policies = {
            "P0_price_only": {"long": price_long * 0, "short": price_short * 0},
            "P1_mean_funding": {"long": p1_long, "short": p1_short},
            "P2_historical_tail": {"long": p2_long, "short": p2_short},
            "P3_direct_quantile": {"long": p3_long, "short": p3_short},
            "P7_svj_t": {"long": p7_long, "short": None},  # short filled in later by the extra-run script
        }
        price_component = {"long": price_long, "short": price_short}

        for policy, fund_by_dir in policies.items():
            for direction, fund in fund_by_dir.items():
                if fund is None:
                    continue
                reserve = price_component[direction] + fund
                e = evaluate_reserve_policy(reserve, pacr_frac[direction])
                rows.append({"horizon": h, "direction": direction, "policy": policy, **e})

    out = pd.DataFrame(rows)
    out.to_csv(args.out / "reserve_policy_comparison.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print("\n" + out.to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
