"""Two follow-ups requested after the main collateral-erosion pass
(run_collateral_erosion.py):

1. P7 (SVJ-t) short-side reserve backtest -- the one policy/direction cell
   left unrun in reserve_policy_comparison.csv.
2. A formal test of whether the small shortfall-count differences among
   P0-P3/P7 (long side, both horizons) are distinguishable from chance,
   rather than leaving that as a qualitative "low power" caveat. Because
   every funding-aware reserve is price-component + a non-negative funding
   allowance, its shortfall set is a SUBSET of P0's shortfall set on the
   same origins (a funding-aware policy can only turn a P0 shortfall into a
   non-shortfall, never the reverse) -- so this is a paired, one-directional
   comparison and McNemar's exact test (conditioning on discordant pairs) is
   the right tool, not an unpaired two-sample test.

Usage:
    python -m analysis.run_collateral_erosion_extra --out analysis/output/collateral_erosion_2026_09_29
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from . import collateral_erosion as ce
from . import config, stage1_omission
from .models import base, caviar, historical_simulation, mean_funding, stochastic_vol
from .run_collateral_erosion import LONG_RUN_DIR, price_only_var_backtest_short

SVJT_KWARGS = {"n_sims": 500}


def shortfall_indicator(reserve: pd.Series, pacr_frac: pd.Series) -> pd.Series:
    df = pd.DataFrame({"reserve": reserve, "pacr": pacr_frac}).dropna()
    return (df["pacr"] > df["reserve"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet("analysis/output/full_run_fwdsum_fix/H72/panel.parquet")

    mcnemar_rows = []
    p7_short_rows = []
    for h in ce.HORIZONS_HOURS:
        ds = ce.build_pacr_dataset(panel, h)
        pacr_frac = {
            d: pd.Series(ce.pacr_frac_notional(ds[f"net_max_{d}"].to_numpy(), ds["mark_price_origin"].to_numpy(),
                                                ce.PRIMARY_LEVERAGE), index=ds.index)
            for d in ("long", "short")
        }

        price_long = stage1_omission.price_only_var_backtest(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        price_short = price_only_var_backtest_short(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]

        # --- 1. P7 short-side SVJ-t backtest (the slow, new piece) ---
        t0 = time.time()
        p7_short_bt = base.run_rolling_backtest(panel, stochastic_vol.svj_t_var, f"C_short_H{h}", h,
                                                 min_train_hours=args.min_train_hours, **SVJT_KWARGS)
        p7_short = p7_short_bt["var_forecast"]
        print(f"  [{h}h] P7 SVJ-t short: {time.time() - t0:.1f}s ({p7_short.notna().sum()} usable forecasts)", flush=True)

        reserve_p7_short = price_short + p7_short
        e_p7_short = None
        sf = shortfall_indicator(reserve_p7_short, pacr_frac["short"])
        p7_short_rows.append({"horizon": h, "direction": "short", "policy": "P7_svj_t",
                               "n": len(sf), "shortfalls": int(sf.sum()), "shortfall_rate": float(sf.mean()),
                               "mean_reserve_pct_notional": float(reserve_p7_short.reindex(sf.index).mean() * 100)})

        # --- 2. McNemar exact test, long side, P0 vs each funding-aware policy ---
        p1_long = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_long_H{h}", h,
                                             min_train_hours=args.min_train_hours)["var_forecast"]
        p2_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_historical_simulation_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p3_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_direct_quantile_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p7_long = pd.read_csv(LONG_RUN_DIR / f"H{h}" / f"backtest_svj_t_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]

        sf0 = shortfall_indicator(price_long, pacr_frac["long"])
        funding_long = {"P1_mean_funding": p1_long, "P2_historical_tail": p2_long,
                        "P3_direct_quantile": p3_long, "P7_svj_t": p7_long}
        for name, fund in funding_long.items():
            reserve = price_long + fund
            sf_policy = shortfall_indicator(reserve, pacr_frac["long"])
            idx = sf0.index.intersection(sf_policy.index)
            a, b0, s = sf0.loc[idx], sf_policy.loc[idx], idx
            table = pd.crosstab(a, b0)
            table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)
            res = mcnemar(table.to_numpy(), exact=True)
            n_discordant = int(table.loc[True, False] + table.loc[False, True])
            mcnemar_rows.append({
                "horizon": h, "policy": name, "n": len(s),
                "P0_shortfalls": int(a.sum()), "policy_shortfalls": int(b0.sum()),
                "n_discordant_pairs": n_discordant,
                "discordant_favoring_policy": int(table.loc[True, False]),
                "discordant_favoring_P0": int(table.loc[False, True]),
                "mcnemar_exact_p": float(res.pvalue),
            })
            print(f"  [{h}h] McNemar P0 vs {name}: {a.sum()} -> {b0.sum()} shortfalls, "
                  f"{n_discordant} discordant, p={res.pvalue:.4f}", flush=True)

    pd.DataFrame(p7_short_rows).to_csv(args.out / "p7_short_reserve_evaluation.csv", index=False)
    mc = pd.DataFrame(mcnemar_rows)
    mc.to_csv(args.out / "mcnemar_shortfall_tests.csv", index=False)
    with pd.option_context("display.width", 220):
        print("\n=== P7 short-side reserve ===")
        print(pd.DataFrame(p7_short_rows).to_string(index=False))
        print("\n=== McNemar exact tests, long side, P0 vs funding-aware policies ===")
        print(mc.to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
