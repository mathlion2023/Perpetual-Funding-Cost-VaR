"""Completes the H5 reserve-policy evaluation per Tina's 2026-10-07 email.

Three gaps identified by re-checking the existing H5 pipeline
(run_collateral_erosion.py's reserve_policy_comparison.csv and
run_collateral_erosion_extra.py's mcnemar_shortfall_tests.csv):

1. "Use common evaluation origins, because P7 currently has fewer
   observations, and report missing predictions separately." -- P0-P3 were
   previously evaluated on their own largest available sample (n=428/180),
   while P7 used a smaller one (n=419/178); McNemar tests intersected
   indices *pairwise* rather than across all five policies at once. This
   script restricts every policy to one common origin set per horizon/
   direction (the intersection across P0-P3 and P7) and reports, per
   policy, which origins were dropped and why.
2. "Funding-forecast performance" -- not previously evaluated at all. For
   each funding-aware policy's funding-reserve component (P1/P2/P3/P7,
   excluding P0 which has no funding forecast to score), this reports
   empirical coverage (share of origins where realised C_long_H{h}/
   C_short_H{h} exceeded the forecast -- should be near 1-alpha=5%) and
   pinball loss at alpha=0.95, on the common sample.
3. "Report uncertainty rather than assuming equivalence." -- the existing
   table reports point shortfall rates and McNemar's point p-value only.
   This adds exact (Clopper-Pearson) binomial confidence intervals on each
   policy's shortfall rate, and on the discordant-pair split underlying
   each McNemar test.

Usage:
    python -m analysis.run_collateral_erosion_h5_complete --out analysis/output/collateral_erosion_2026_10_07_h5_complete
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from . import collateral_erosion as ce
from . import config, stage1_omission
from .models import base, caviar, historical_simulation, mean_funding, stochastic_vol
from .run_collateral_erosion import LONG_RUN_DIR, price_only_var_backtest_short

SVJT_KWARGS = {"n_sims": 500}
ALPHA = config.VAR_ALPHA  # 0.95


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, alpha: float) -> float:
    e = y_true.to_numpy() - y_pred.to_numpy()
    return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))


def evaluate_reserve_policy_ci(reserve: pd.Series, pacr_frac: pd.Series) -> dict:
    df = pd.DataFrame({"reserve": reserve, "pacr": pacr_frac}).dropna()
    n = len(df)
    if n == 0:
        return {"n": 0}
    shortfall = df["pacr"] > df["reserve"]
    k = int(shortfall.sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    severity = (df["pacr"] - df["reserve"])[shortfall]
    return {
        "n": n, "shortfalls": k, "shortfall_rate": k / n,
        "shortfall_rate_ci95_lo": float(lo), "shortfall_rate_ci95_hi": float(hi),
        "mean_shortfall_severity_pct_notional": float(severity.mean() * 100) if k else 0.0,
        "max_shortfall_pct_notional": float(severity.max() * 100) if k else 0.0,
        "mean_reserve_pct_notional": float(df["reserve"].mean() * 100),
        "p50_reserve_pct_notional": float(df["reserve"].quantile(0.50) * 100),
        "p90_reserve_pct_notional": float(df["reserve"].quantile(0.90) * 100),
        "p95_reserve_pct_notional": float(df["reserve"].quantile(0.95) * 100),
        "p99_reserve_pct_notional": float(df["reserve"].quantile(0.99) * 100),
    }


def funding_forecast_performance(fund_forecast: pd.Series, realised_funding: pd.Series, alpha: float = ALPHA) -> dict:
    df = pd.DataFrame({"forecast": fund_forecast, "actual": realised_funding}).dropna()
    n = len(df)
    if n == 0:
        return {"n": 0}
    breach = df["actual"] > df["forecast"]
    k = int(breach.sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    return {
        "n": n, "breaches": k, "coverage_rate": k / n,
        "coverage_rate_ci95_lo": float(lo), "coverage_rate_ci95_hi": float(hi),
        "target_exceedance_rate": 1.0 - alpha,
        "pinball_loss": pinball_loss(df["actual"], df["forecast"], alpha),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    # Inputs as arguments (lag audit, 2026-11-06): these were hard-coded to
    # full_run_fwdsum_fix / full_run_2026_09_29_corrected, so PACR and reserve
    # results silently stayed on a panel built before the 2026-10-20 f_eff fix.
    ap.add_argument("--panel", type=Path, default=Path("analysis/output/full_run_fwdsum_fix/H72/panel.parquet"))
    ap.add_argument("--long-run-dir", type=Path, default=LONG_RUN_DIR,
                    help="run_pipeline output dir (with H72/, H168/) supplying P2/P3/P7 long-side forecasts")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    long_run_dir = args.long_run_dir
    print(f"panel: {args.panel} ({len(panel)} hours); long-side forecasts from {long_run_dir}", flush=True)

    comparison_rows = []
    missing_rows = []
    mcnemar_rows = []
    funding_perf_rows = []

    for h in ce.HORIZONS_HOURS:
        ds = ce.build_pacr_dataset(panel, h)
        pacr_frac = {
            d: pd.Series(ce.pacr_frac_notional(ds[f"net_max_{d}"].to_numpy(), ds["mark_price_origin"].to_numpy(),
                                                ce.PRIMARY_LEVERAGE), index=ds.index)
            for d in ("long", "short")
        }
        actual_funding = {"long": panel[f"C_long_H{h}"], "short": panel[f"C_short_H{h}"]}

        price_long = stage1_omission.price_only_var_backtest(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        price_short = price_only_var_backtest_short(panel, h, min_train_hours=args.min_train_hours)["var_forecast"]
        price_component = {"long": price_long, "short": price_short}

        p1_long = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_long_H{h}", h,
                                             min_train_hours=args.min_train_hours)["var_forecast"]
        p1_short = base.run_rolling_backtest(panel, mean_funding.mean_funding_reserve, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        p2_long = pd.read_csv(long_run_dir / f"H{h}" / f"backtest_historical_simulation_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p2_short = base.run_rolling_backtest(panel, historical_simulation.historical_simulation_var, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        p3_long = pd.read_csv(long_run_dir / f"H{h}" / f"backtest_direct_quantile_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        p3_short = base.run_rolling_backtest(panel, caviar.direct_quantile_var, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours)["var_forecast"]
        p7_long = pd.read_csv(long_run_dir / f"H{h}" / f"backtest_svj_t_H{h}.csv",
                               parse_dates=["origin"]).set_index("origin")["var_forecast"]
        t0 = time.time()
        p7_short = base.run_rolling_backtest(panel, stochastic_vol.svj_t_var, f"C_short_H{h}", h,
                                              min_train_hours=args.min_train_hours, **SVJT_KWARGS)["var_forecast"]
        print(f"  [{h}h] P7 SVJ-t short: {time.time() - t0:.1f}s", flush=True)

        funds = {
            "long": {"P1_mean_funding": p1_long, "P2_historical_tail": p2_long,
                     "P3_direct_quantile": p3_long, "P7_svj_t": p7_long},
            "short": {"P1_mean_funding": p1_short, "P2_historical_tail": p2_short,
                      "P3_direct_quantile": p3_short, "P7_svj_t": p7_short},
        }

        for direction in ("long", "short"):
            universe = price_component[direction].dropna().index.intersection(pacr_frac[direction].dropna().index)
            common_idx = universe
            for policy_name, series in funds[direction].items():
                avail = series.reindex(universe)
                missing_idx = avail[avail.isna()].index
                missing_rows.append({"horizon": h, "direction": direction, "policy": policy_name,
                                      "n_universe": len(universe), "n_missing": len(missing_idx),
                                      "missing_origins": ";".join(str(x) for x in missing_idx)})
                common_idx = common_idx.intersection(series.dropna().index)
            print(f"  [{h}h {direction}] universe n={len(universe)}, common (all policies) n={len(common_idx)}", flush=True)

            policies = {"P0_price_only": pd.Series(0.0, index=common_idx), **{k: v.reindex(common_idx) for k, v in funds[direction].items()}}
            pacr_common = pacr_frac[direction].reindex(common_idx)

            sf_by_policy = {}
            for policy_name, fund in policies.items():
                reserve = price_component[direction].reindex(common_idx) + fund
                e = evaluate_reserve_policy_ci(reserve, pacr_common)
                comparison_rows.append({"horizon": h, "direction": direction, "policy": policy_name, **e})
                df_sf = pd.DataFrame({"reserve": reserve, "pacr": pacr_common}).dropna()
                sf_by_policy[policy_name] = (df_sf["pacr"] > df_sf["reserve"])

                if policy_name != "P0_price_only":
                    perf = funding_forecast_performance(fund, actual_funding[direction].reindex(common_idx), ALPHA)
                    funding_perf_rows.append({"horizon": h, "direction": direction, "policy": policy_name, **perf})

            if direction == "long":
                sf0 = sf_by_policy["P0_price_only"]
                for policy_name in ("P1_mean_funding", "P2_historical_tail", "P3_direct_quantile", "P7_svj_t"):
                    sf_p = sf_by_policy[policy_name]
                    table = pd.crosstab(sf0, sf_p).reindex(index=[False, True], columns=[False, True], fill_value=0)
                    res = mcnemar(table.to_numpy(), exact=True)
                    n_discordant = int(table.loc[True, False] + table.loc[False, True])
                    fav_policy = int(table.loc[True, False])
                    if n_discordant > 0:
                        d_lo, d_hi = proportion_confint(fav_policy, n_discordant, alpha=0.05, method="beta")
                    else:
                        d_lo, d_hi = (np.nan, np.nan)
                    mcnemar_rows.append({
                        "horizon": h, "policy": policy_name, "n": len(sf0),
                        "P0_shortfalls": int(sf0.sum()), "policy_shortfalls": int(sf_p.sum()),
                        "n_discordant_pairs": n_discordant,
                        "discordant_favoring_policy": fav_policy,
                        "discordant_favoring_P0": int(table.loc[False, True]),
                        "share_discordant_favoring_policy_ci95_lo": float(d_lo), "share_discordant_favoring_policy_ci95_hi": float(d_hi),
                        "mcnemar_exact_p": float(res.pvalue),
                    })

    pd.DataFrame(comparison_rows).to_csv(args.out / "reserve_policy_comparison_common_sample.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(args.out / "missing_predictions_by_policy.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(args.out / "mcnemar_shortfall_tests_common_sample.csv", index=False)
    pd.DataFrame(funding_perf_rows).to_csv(args.out / "funding_forecast_performance.csv", index=False)

    with pd.option_context("display.width", 240, "display.float_format", "{:.4g}".format):
        print("\n=== Reserve-policy comparison, common sample ===")
        print(pd.DataFrame(comparison_rows).to_string(index=False))
        print("\n=== Missing predictions by policy ===")
        print(pd.DataFrame(missing_rows)[["horizon", "direction", "policy", "n_universe", "n_missing"]].to_string(index=False))
        print("\n=== McNemar exact tests, common sample ===")
        print(pd.DataFrame(mcnemar_rows).to_string(index=False))
        print("\n=== Funding-forecast performance ===")
        print(pd.DataFrame(funding_perf_rows).to_string(index=False))
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
