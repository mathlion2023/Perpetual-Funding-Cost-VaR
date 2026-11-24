"""Standalone re-analysis responding to Tina's 2026-09-27 email.

Reuses the already-built panel and the already-cached Section 6 model
backtests (all forecast C_long_H, the *gross* funding-cost target -- these
are unaffected by the L_net signed-funding fix in funding_cost.py, so they
do not need to be refit). Only reruns the pieces that actually depend on
L_net_H{h}: price-only VaR (unaffected by the bug but not previously cached
on its own), net-loss VaR/ES, Stage 1 (Section 7), and the omission gap
(Section 9). Also adds the new checks Tina asked for: matched breach/
prevention counts, a naive funding-adjusted benchmark, and gap economic
magnitude in bps/percent-of-collateral.

Usage: python -m analysis.rerun_2026_09_27
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, backtesting, omission_gap, stage1_omission

# Defaults reproduce the published 2026-09-27 outputs. The environment
# overrides (added 2026-09-28) let the same analysis run on a newer base run
# -- e.g. after the forward_sum fix -- without overwriting those outputs.
OLD_OUT = Path(os.environ.get("FCVAR_BASE_RUN", config.OUTPUT_DIR / "full_run_rescaled"))
NEW_OUT = Path(os.environ.get("FCVAR_REANALYSIS_OUT", config.OUTPUT_DIR / "full_run_corrected"))

FUNDING_PROXY_MODELS = ["historical_simulation", "svj_t", "rough_vol", "direct_quantile"]


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(OLD_OUT / "H72" / "panel.parquet")
    for h in config.HORIZONS_HOURS:
        r_h = p[f"R_H{h}"]
        c_signed_h = p[f"C_signed_H{h}"]
        p[f"L_net_H{h}"] = -r_h + c_signed_h
    return p


def cached_backtest(horizon: int, model_name: str) -> pd.DataFrame:
    path = OLD_OUT / f"H{horizon}" / f"backtest_{model_name}_H{horizon}.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)


def matched_breach_counts(panel: pd.DataFrame, price_res: pd.DataFrame, horizon: int) -> dict:
    idx = price_res.index
    L_price = panel.reindex(idx)[f"L_price_H{horizon}"]
    L_net = panel.reindex(idx)[f"L_net_H{horizon}"]
    VaR_price = price_res["var_forecast"]
    valid = L_price.notna() & L_net.notna() & VaR_price.notna()
    L_price, L_net, VaR_price = L_price[valid], L_net[valid], VaR_price[valid]
    breach_price = L_price > VaR_price
    breach_net = L_net > VaR_price
    return {
        "n_matched_origins": int(valid.sum()),
        "n_breach_price_only": int(breach_price.sum()),
        "n_breach_net": int(breach_net.sum()),
        "n_funding_caused_breach": int((breach_net & ~breach_price).sum()),
        "n_funding_prevented_breach": int((~breach_net & breach_price).sum()),
        "n_breach_both": int((breach_net & breach_price).sum()),
    }


def naive_funding_adjusted_backtest(panel: pd.DataFrame, price_res: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """VaR_naive_net = VaR_price + expanding causal MEAN funding cost over the
    horizon (mean, not a fitted quantile model) -- isolates the value of a
    pure mean-shift adjustment for expected funding from the value of fully
    modelling funding's own uncertainty (the direct_quantile net-VaR model)."""
    c_long_col = f"C_long_H{horizon}"
    idx_list = list(panel.index)
    pos_of = {ts: i for i, ts in enumerate(idx_list)}
    c_long_series = panel[c_long_col]
    rows = []
    for origin, row in price_res.iterrows():
        if pd.isna(row["var_forecast"]):
            rows.append({"origin": origin, "var_forecast": np.nan})
            continue
        pos = pos_of[origin]
        label_cutoff = max(pos - horizon, 0)
        hist = c_long_series.iloc[:label_cutoff].dropna()
        naive_funding = float(hist.mean()) if len(hist) >= 30 else np.nan
        vf = row["var_forecast"] + naive_funding if not np.isnan(naive_funding) else np.nan
        rows.append({"origin": origin, "var_forecast": vf})
    out = pd.DataFrame(rows).set_index("origin")
    out["realized"] = panel.reindex(out.index)[f"L_net_H{horizon}"]
    out["exception"] = (out["realized"] > out["var_forecast"]).astype("float")
    return out


def gap_economic_magnitude(gap_ds: pd.DataFrame, leverage_levels=(5, 10)) -> pd.DataFrame:
    rows = []
    for label in ["VaR_price", "VaR_net", "gap"]:
        s = gap_ds[label]
        row = {
            "measure": label,
            "mean_bps": float(s.mean() * 10_000),
            "p95_bps": float(s.quantile(0.95) * 10_000),
            "p99_bps": float(s.quantile(0.99) * 10_000),
        }
        for k in leverage_levels:
            row[f"mean_pct_collateral_{k}x"] = float(s.mean() * k * 100)
            row[f"p99_pct_collateral_{k}x"] = float(s.quantile(0.99) * k * 100)
        rows.append(row)
    return pd.DataFrame(rows).set_index("measure")


def run_horizon(h: int, panel: pd.DataFrame) -> dict:
    out_dir = NEW_OUT / f"H{h}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {"horizon": h}

    print(f"[H{h}] price-only VaR backtest (direct_quantile on L_price)...", flush=True)
    price_res = stage1_omission.price_only_var_backtest(panel, h, min_train_hours=config.MIN_TRAIN_HOURS)
    price_res.to_csv(out_dir / "price_only_var_backtest.csv")
    result["price_only_backtest_report"] = backtesting.backtest_report(price_res, config.VAR_ALPHA)

    stage1_by_proxy = {}
    for proxy in FUNDING_PROXY_MODELS:
        funding_res = cached_backtest(h, proxy)
        ds = stage1_omission.build_omission_dataset(panel, price_res, funding_res, h)
        entry = {"n_full_sample_origins": len(ds), "n_full_sample_events": int(ds["omission_event"].sum())}
        if len(ds) > 45:
            res = stage1_omission.run_stage1_out_of_sample(ds)
            entry.update(
                {
                    "n_oos_scored_origins": res["scores"]["m3_controls_funding"]["n"],
                    "n_oos_scored_events": res["scores"]["m3_controls_funding"]["n_events"],
                    "beta_funding": res["beta_funding"],
                    "p_one_sided": res["beta_funding_pvalue_one_sided_H0_beta_le_0"],
                    "lr_test_p": res["lr_test_m2_vs_m3"]["pvalue"],
                    "firth_beta_funding": res["firth_robustness"]["beta_funding"],
                    "firth_p_one_sided": res["firth_robustness"]["beta_funding_pvalue_one_sided_H0_beta_le_0"],
                    "firth_lr_test_p": res["firth_robustness"]["lr_test_m2_vs_m3"]["pvalue"],
                    "full_sample_fit_error": res["full_sample_fit_error"],
                }
            )
        stage1_by_proxy[proxy] = entry
        print(f"  [{proxy}] full-sample: n={entry['n_full_sample_origins']}, events={entry['n_full_sample_events']}"
              f" | oos-scored: n={entry.get('n_oos_scored_origins')}, events={entry.get('n_oos_scored_events')}", flush=True)
    result["stage1_by_proxy"] = stage1_by_proxy

    result["matched_breach_counts"] = matched_breach_counts(panel, price_res, h)
    print(f"  matched breach counts: {result['matched_breach_counts']}", flush=True)

    print(f"[H{h}] net-loss VaR backtest (direct_quantile on corrected L_net)...", flush=True)
    net_res = omission_gap.net_loss_var_backtest(panel, h, min_train_hours=config.MIN_TRAIN_HOURS)
    net_res.to_csv(out_dir / "net_loss_var_backtest.csv")
    result["net_var_backtest_report"] = backtesting.backtest_report(net_res, config.VAR_ALPHA)

    naive_res = naive_funding_adjusted_backtest(panel, price_res, h)
    naive_res.to_csv(out_dir / "naive_funding_adjusted_backtest.csv")
    result["naive_benchmark_backtest_report"] = backtesting.backtest_report(naive_res, config.VAR_ALPHA)
    result["dm_naive_vs_full_net_model"] = backtesting.diebold_mariano_test(naive_res, net_res, config.VAR_ALPHA)
    print(f"  naive benchmark vs full net-VaR model, DM test: {result['dm_naive_vs_full_net_model']}", flush=True)

    gap_ds = omission_gap.build_gap_dataset(panel, price_res, net_res)
    gap_ds.to_csv(out_dir / "omission_gap_dataset.csv")
    result["gap_n"] = len(gap_ds)
    if len(gap_ds) > 10:
        regime = omission_gap.fit_gap_regime_test(gap_ds)
        result["gap_regime_test"] = {"delta": regime["delta"], "p_one_sided": regime["delta_pvalue_one_sided_H0_delta_le_0"]}

        rur_summary = omission_gap.gap_and_rur_summary(gap_ds)
        rur_summary.to_csv(out_dir / "gap_rur_summary.csv")

        mc_ds = omission_gap.build_gap_market_conditions_dataset(panel, price_res, net_res)
        mc_ds.to_csv(out_dir / "omission_gap_market_conditions_dataset.csv")
        if len(mc_ds) > 15:
            mc_result = omission_gap.fit_gap_market_conditions_regression(mc_ds)
            result["market_conditions_regression"] = {
                "n": mc_result["n"],
                "r_squared": mc_result["r_squared"],
                "params": mc_result["params"],
                "pvalues": mc_result["pvalues_two_sided"],
            }

        econ = gap_economic_magnitude(gap_ds)
        econ.to_csv(out_dir / "gap_economic_magnitude.csv")

    print(f"[H{h}] ES backtests (price + net, corrected L_net)...", flush=True)
    price_es = omission_gap.price_es_backtest(panel, h, min_train_hours=config.MIN_TRAIN_HOURS)
    net_es = omission_gap.net_es_backtest(panel, h, min_train_hours=config.MIN_TRAIN_HOURS)
    price_es.to_csv(out_dir / "es_price_backtest.csv")
    net_es.to_csv(out_dir / "es_net_backtest.csv")
    es_gap_ds = omission_gap.build_es_gap_dataset(panel, price_es, net_es)
    es_gap_ds.to_csv(out_dir / "es_gap_dataset.csv")
    if len(es_gap_ds) > 10:
        es_summary = omission_gap.es_gap_summary(es_gap_ds)
        es_summary.to_csv(out_dir / "es_gap_summary.csv")

    with open(out_dir / "reanalysis_summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[H{h}] done -> {out_dir}", flush=True)
    return result


def main():
    NEW_OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()

    corrected_drain = None
    try:
        from . import funding_cost

        panel_for_drain = panel.copy()
        for h in config.HORIZONS_HOURS:
            for k in config.LEVERAGE_LEVELS:
                panel_for_drain[f"CollateralDrain_H{h}_k{k}"] = k * panel_for_drain[f"C_long_H{h}"]
        corrected_drain = funding_cost.collateral_drain_summary(panel_for_drain)
        corrected_drain.to_csv(NEW_OUT / "collateral_drain_summary_with_tail_mean.csv", index=False)
        print(corrected_drain.to_string(index=False))
    except Exception as exc:  # noqa: BLE001
        print(f"collateral drain regen failed: {exc}")

    all_results = {}
    for h in config.HORIZONS_HOURS:
        all_results[h] = run_horizon(h, panel)

    with open(NEW_OUT / "reanalysis_summary_all.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
