"""Robustness check: does the reserve-policy null (no funding-aware policy
beats price-only P0) hold at 2x/5x leverage, not only the paper's primary
10x? Flagged as untested in both the FMF and FRL drafts.

Reuses the leverage-independent VaR components (price-only, P1 mean-funding
freshly fit; P2/P3/P7 read from the already-computed 2022-2025 lagfix run)
and just varies the PACR threshold's leverage, per
collateral_erosion.pacr_frac_notional's own leverage parameter -- no new
methodology, same formulas as the primary-leverage analysis.

Usage:
    python -m analysis.run_reserve_policy_leverage_robustness --out analysis/output/collateral_erosion_2026_11_25_lagfix_2022_2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from . import collateral_erosion as ce
from . import config, stage1_omission
from .models import base, mean_funding
from .run_collateral_erosion_lagfix_2022_2025 import PANEL_PATH, LONG_RUN_DIR, price_only_var_backtest_short

LEVERAGES = (2, 5, 10)


def shortfall_indicator(reserve: pd.Series, pacr_frac: pd.Series) -> pd.Series:
    df = pd.DataFrame({"reserve": reserve, "pacr": pacr_frac}).dropna()
    return df["pacr"] > df["reserve"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL_PATH)
    print(f"panel: {panel.shape[0]} hours, from {PANEL_PATH}", flush=True)

    rows = []
    for h in ce.HORIZONS_HOURS:
        ds = ce.build_pacr_dataset(panel, h)

        price_long = stage1_omission.price_only_var_backtest(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        p1_long = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_long_H{h}", h,
                                             min_train_hours=args.min_train_hours)["var_forecast"]
        p2_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_historical_simulation_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p3_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_direct_quantile_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p7_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_svj_t_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]

        funding_long = {"P1_mean_funding": p1_long, "P2_historical_tail": p2_long,
                        "P3_direct_quantile": p3_long, "P4_svj_t": p7_long}

        for lev in LEVERAGES:
            pacr_frac = pd.Series(
                ce.pacr_frac_notional(ds["net_max_long"].to_numpy(), ds["mark_price_origin"].to_numpy(), lev),
                index=ds.index,
            )
            sf0 = shortfall_indicator(price_long, pacr_frac)
            rows.append({"horizon": h, "leverage": lev, "policy": "P0_price_only",
                         "n": len(sf0), "shortfalls": int(sf0.sum())})
            for name, fund in funding_long.items():
                reserve = price_long + fund
                sf_policy = shortfall_indicator(reserve, pacr_frac)
                idx = sf0.index.intersection(sf_policy.index)
                a, b = sf0.loc[idx], sf_policy.loc[idx]
                table = pd.crosstab(a, b).reindex(index=[False, True], columns=[False, True], fill_value=0)
                res = mcnemar(table.to_numpy(), exact=True)
                n_discordant = int(table.loc[True, False] + table.loc[False, True])
                rows.append({
                    "horizon": h, "leverage": lev, "policy": name,
                    "n": len(idx), "shortfalls": int(b.sum()),
                    "P0_shortfalls": int(a.sum()), "n_discordant_pairs": n_discordant,
                    "mcnemar_exact_p": float(res.pvalue),
                })
            print(f"  [H={h}, {lev}x] done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out / "reserve_policy_leverage_robustness.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print("\n" + out.to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
