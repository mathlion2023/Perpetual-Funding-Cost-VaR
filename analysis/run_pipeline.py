"""End-to-end orchestration: data -> validation -> panel -> funding cost ->
forecast set -> Section 8 models -> Section 7/9 tests -> Section 10 backtests.

Usage:
    python -m analysis.run_pipeline --start 2022-01-01 --end 2025-06-30 \
        --horizon 72 --models historical_simulation direct_quantile \
        --out analysis/output/run_2022_2025

Notes:
  - Deribit BTC perpetual only (see config.py's scope note).
  - Models sv_t/svj_t/arx_garch/rough_vol are meaningfully slower (each
    origin re-fits a small numerical model); historical_simulation and
    direct_quantile are fast. caviar's recursive optimisation is moderate.
  - min_train_hours defaults to config.MIN_TRAIN_HOURS (~6 months); the
    first several months of any date range will not produce forecasts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from . import config, data_loading, data_validation, funding_cost, forecast_set, panel as panel_mod
from . import backtesting, omission_gap, stage1_omission
from .models import arx_garch, base, caviar, historical_simulation, rough_vol, stochastic_vol

MODEL_REGISTRY = {
    # name: (model_fn, model_kwargs, window_hours)
    # window_hours=None => expanding (all history to date). A fixed rolling
    # window is used where an unconstrained expanding-window fit was found
    # to be numerically unstable in practice (see arx_garch_t below).
    "historical_simulation": (historical_simulation.historical_simulation_var, {}, None),
    "direct_quantile": (caviar.direct_quantile_var, {}, None),
    "caviar": (caviar.caviar_var, {}, None),
    # arx_garch_t's joint ARX(1)+GARCH MLE occasionally fits a
    # non-stationary AR root when handed a long, regime-heterogeneous
    # expanding window (observed on real Deribit data: ~4/7 sampled origins
    # NaN'd out under an expanding window vs 1/7 under a 6-month rolling
    # window at the same points in time) -- bounding the window is a
    # disclosed stability fix, not a silent tuning choice.
    "arx_garch_t": (arx_garch.arx_garch_t_var, {"n_sims": 1000}, 24 * 180),
    "sv_t": (stochastic_vol.sv_t_var, {"n_sims": 500}, None),
    "svj_t": (stochastic_vol.svj_t_var, {"n_sims": 500}, None),
    "rough_vol": (rough_vol.direct_rough_vol_var, {"n_sims": 500}, None),
}

# Stage 3 models added to the proposal on 2026-09-28 (Sections 4.6.4-4.6.5):
# har_quantile, gas_t, ms_garch_t, stationary_bootstrap. Defined once in
# run_new_models.py and registered here too, so they can be passed to
# --models like the original seven. None is eligible as --stage1-proxy-model
# for the confirmatory Stage 1 test (proposal Section 4.6: SVJ-t stays the
# pre-specified proxy; any other proxy is sensitivity analysis only).
from .run_new_models import NEW_MODEL_REGISTRY  # noqa: E402

MODEL_REGISTRY.update(NEW_MODEL_REGISTRY)


def build_panel(start: dt.date, end: dt.date, high_funding_method: str = "expanding") -> pd.DataFrame:
    funding_df = data_loading.load_funding_range(start, end)

    report = data_validation.validate_funding_conventions(funding_df)
    print(report.summary(), flush=True)
    if not report.passed:
        warnings.warn("Section 2 validation raised warnings -- inspect before trusting funding_cost.py output.")

    # Streaming trade-flow aggregation (month-at-a-time): a multi-year trades
    # history does not fit in memory as one concatenated DataFrame (this
    # crashed with pyarrow.lib.ArrowMemoryError on the first full-history
    # attempt) -- see panel.build_hourly_panel_streaming.
    trades_files = data_loading.trades_files(start, end)
    # Auxiliary spread_binc control (Tina, 2026-09-26): bid/ask only, not a
    # funding-cost source -- see config.BINANCE_FUNDING_DIR's scope note.
    binance_df = data_loading.load_binance_quotes_range(start, end)
    p = panel_mod.build_hourly_panel_streaming(funding_df, trades_files, binance_df=binance_df)
    p = funding_cost.add_funding_cost_outcomes(p)
    p = funding_cost.add_price_and_loss_outcomes(p)
    p = funding_cost.add_collateral_drain(p)
    p = forecast_set.build_forecast_set(p, method=high_funding_method)
    return p


def build_panel_binance(start: dt.date, end: dt.date, high_funding_method: str = "expanding") -> pd.DataFrame:
    """Binance analogue of build_panel (2026-11-06, second-venue comparison).

    funding_cost.py, forecast_set.py and every Section 8 model/Section
    9-10-11 test downstream are reused completely unchanged -- only data
    ingestion and panel construction differ by venue (panel.build_binance_ticker_hourly's
    discrete-settlement f_eff is what makes that possible; see its docstring
    and config.py's note on why Binance needs a different convention than
    Deribit's continuous one).
    """
    funding_df = data_loading.load_binance_funding_range(start, end)

    report = data_validation.validate_binance_funding_conventions(funding_df)
    print(report.summary(), flush=True)
    if not report.passed:
        warnings.warn("Binance Section 2 validation raised warnings -- inspect before trusting funding_cost.py output.")

    trades_files = data_loading.binance_trades_files(start, end)
    # Cross-venue control: Deribit's own spread fills the spread_binc slot
    # here (see panel.build_binance_hourly_panel's docstring) -- only ts/bid/ask
    # are read from this, same as the Deribit panel's own Binance-quotes control.
    deribit_funding_df = data_loading.load_funding_range(start, end)
    p = panel_mod.build_binance_hourly_panel_streaming(funding_df, trades_files, deribit_funding_df)
    p = funding_cost.add_funding_cost_outcomes(p)
    p = funding_cost.add_price_and_loss_outcomes(p)
    p = funding_cost.add_collateral_drain(p)
    p = forecast_set.build_forecast_set(p, method=high_funding_method)
    return p


def run_models(panel: pd.DataFrame, horizon: int, model_names: list[str], min_train_hours: int) -> dict[str, pd.DataFrame]:
    target_col = f"C_long_H{horizon}"
    results = {}
    for name in model_names:
        model_fn, kwargs, window_hours = MODEL_REGISTRY[name]
        t0 = time.time()
        res = base.run_rolling_backtest(
            panel, model_fn, target_col, horizon, min_train_hours=min_train_hours, window_hours=window_hours, **kwargs
        )
        n_ok = res["var_forecast"].notna().sum()
        print(f"  [{name}] {len(res)} origins, {n_ok} successful forecasts, {time.time() - t0:.1f}s", flush=True)
        results[name] = res
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--end", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--horizon", type=int, default=72, choices=config.HORIZONS_HOURS)
    ap.add_argument(
        "--venue",
        choices=["deribit", "binance"],
        default="deribit",
        help="deribit (default, see config.py's scope note) or binance (second-venue comparison added "
        "2026-11-06, discrete 8-hourly funding settlement -- see build_panel_binance).",
    )
    ap.add_argument("--models", nargs="+", default=["historical_simulation", "direct_quantile"], choices=list(MODEL_REGISTRY))
    ap.add_argument(
        "--stage1-proxy-model",
        default="svj_t",
        choices=list(MODEL_REGISTRY),
        help="Which of --models supplies VaR_funding for the Section 7 (Stage 1) omission test. "
        "Defaults to svj_t, the paper's pre-specified primary confirmatory proxy (2026-09-26 peer "
        "review response) -- NOT args.models[0], which previously silently selected whichever model "
        "happened to be listed first (historical_simulation by --models' own default) regardless of "
        "which proxy the methodology text designates as primary. Must be one of --models.",
    )
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    ap.add_argument("--high-funding-method", choices=["expanding", "fixed_initial"], default="expanding")
    ap.add_argument(
        "--compute-es-gap",
        action="store_true",
        help="Also compute the item-5 expected-shortfall gap (ES_net - ES_price). Opt-in: ~6x the "
        "runtime of the VaR-only Section 9 pass, since each ES forecast averages 6 quantile-regression fits.",
    )
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "run")
    ap.add_argument(
        "--reuse-panel",
        type=Path,
        help="Load this panel.parquet (built by this same code for the same --start/--end) instead of "
        "rebuilding it from raw files -- the build is the most memory-hungry step.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.reuse_panel:
        print(f"=== Reusing panel {args.reuse_panel} ===")
        p = pd.read_parquet(args.reuse_panel)
        if str(p.index.min().date()) != str(args.start) or p.index.max().date() > args.end:
            raise ValueError(f"--reuse-panel covers {p.index.min()} -> {p.index.max()}, not {args.start} -> {args.end}")
    else:
        print(f"=== Building panel {args.start} -> {args.end} (venue={args.venue}) ===")
        p = build_panel(args.start, args.end, args.high_funding_method) if args.venue == "deribit" \
            else build_panel_binance(args.start, args.end, args.high_funding_method)
    print(f"panel: {p.shape[0]} hours, {p.shape[1]} columns")
    p.to_parquet(args.out / "panel.parquet")

    print(f"\n=== Funding-cost / collateral-drain descriptives ===")
    drain_summary = funding_cost.collateral_drain_summary(p)
    print(drain_summary.to_string(index=False))
    drain_summary.to_csv(args.out / "collateral_drain_summary.csv", index=False)

    print(f"\n=== Section 8 models: horizon={args.horizon}h, models={args.models} ===")
    results = run_models(p, args.horizon, args.models, args.min_train_hours)
    for name, res in results.items():
        res.to_csv(args.out / f"backtest_{name}_H{args.horizon}.csv")

    print("\n=== Section 10 backtest reports ===")
    reports = {}
    for name, res in results.items():
        reports[name] = backtesting.backtest_report(res, config.VAR_ALPHA)
        print(f"  [{name}] exceptions {reports[name]['realized_exceptions']}/{reports[name]['n_forecasts']} "
              f"(expected {reports[name]['expected_exceptions']:.1f}); "
              f"Kupiec p={reports[name]['kupiec_uc']['pvalue']:.3f}; "
              f"quantile loss={reports[name]['quantile_loss']:.6g}")
    with open(args.out / "backtest_reports.json", "w") as f:
        json.dump(reports, f, indent=2, default=str)

    avail = backtesting.availability_summary(results)
    print("\nForecast availability per model (scheduled origins -> usable; reasons for the rest):")
    print(avail.to_string())
    avail.to_csv(args.out / "forecast_availability.csv")

    if len(results) > 1:
        dm_matrix = backtesting.compare_models(results, config.VAR_ALPHA)
        print("\nDiebold-Mariano matrix (row vs column, +ve => row worse; pairwise samples, see availability above):")
        print(dm_matrix.to_string())
        dm_matrix.to_csv(args.out / "diebold_mariano_matrix.csv")

        # Same comparisons on the origins where every model has a usable forecast,
        # so a model that fails often (e.g. arx_garch_t) is not compared on a
        # different subsample from everyone else.
        common = backtesting.common_sample(results)
        n_common = len(next(iter(common.values())))
        print(f"\n=== Common-sample comparison: {n_common} origins usable for all {len(common)} models ===")
        common_reports = {}
        for name, res in common.items():
            r = backtesting.backtest_report(res, config.VAR_ALPHA)
            common_reports[name] = r
            print(f"  [{name}] exceptions {r['realized_exceptions']}/{r['n_forecasts']}; "
                  f"Kupiec p={r['kupiec_uc']['pvalue']:.3f}; quantile loss={r['quantile_loss']:.6g}")
        with open(args.out / "backtest_reports_common_sample.json", "w") as f:
            json.dump({"n_common_origins": n_common, "reports": common_reports}, f, indent=2, default=str)
        dm_common = backtesting.compare_models(common, config.VAR_ALPHA)
        print("Diebold-Mariano matrix, common sample:")
        print(dm_common.to_string())
        dm_common.to_csv(args.out / "diebold_mariano_matrix_common_sample.csv")

    print("\n=== Section 7: price-only VaR omission test ===")
    price_res = stage1_omission.price_only_var_backtest(p, args.horizon, min_train_hours=args.min_train_hours)
    if args.stage1_proxy_model not in results:
        raise ValueError(
            f"--stage1-proxy-model {args.stage1_proxy_model!r} was not computed -- "
            f"add it to --models (currently {args.models})."
        )
    funding_model_for_stage1 = args.stage1_proxy_model
    print(f"  (Stage 1 funding-VaR proxy: {funding_model_for_stage1})")
    ds = stage1_omission.build_omission_dataset(p, price_res, results[funding_model_for_stage1], args.horizon)
    s1_acct = stage1_omission.sample_accounting(p, price_res, results[funding_model_for_stage1], args.horizon)
    print(f"  omission dataset: {len(ds)} usable origins, {int(ds['omission_event'].sum())} omission events "
          f"(event rate {ds['omission_event'].mean():.3f})")
    print(f"  origin accounting: {s1_acct}")
    with open(args.out / "stage1_sample_accounting.json", "w") as f:
        json.dump(s1_acct | {"usable_origins": len(ds), "events": int(ds["omission_event"].sum()),
                             "first_origin": str(ds.index.min()), "last_origin": str(ds.index.max())}, f, indent=2)
    ds.to_csv(args.out / "stage1_omission_dataset.csv")
    if len(ds) > 45:
        stage1_res = stage1_omission.run_stage1_out_of_sample(ds)
        if stage1_res["full_sample_fit_error"]:
            print(f"  full-sample encompassing fit (m2/m3) failed: {stage1_res['full_sample_fit_error']} "
                  "-- beta_funding/LR test unavailable for this range; out-of-sample scores below are unaffected.")
        else:
            print(f"  beta_funding={stage1_res['beta_funding']:.4g}, "
                  f"one-sided p={stage1_res['beta_funding_pvalue_one_sided_H0_beta_le_0']:.4f}; "
                  f"LR(m2 vs m3) p={stage1_res['lr_test_m2_vs_m3']['pvalue']:.4f} (two-sided chi2(1)), "
                  f"signed-root one-sided p={stage1_res['lr_test_m2_vs_m3']['pvalue_one_sided_signed_root']:.4f}")
        fr = stage1_res["firth_robustness"]
        print(f"  [Firth robustness, peer review R2] beta_funding={fr['beta_funding']:.4g}, "
              f"one-sided p={fr['beta_funding_pvalue_one_sided_H0_beta_le_0']:.4f}; "
              f"LR(m2 vs m3) p={fr['lr_test_m2_vs_m3']['pvalue']:.4f} (two-sided), "
              f"signed-root one-sided p={fr['lr_test_m2_vs_m3']['pvalue_one_sided_signed_root']:.4f}")
        with open(args.out / "stage1_omission_test.json", "w") as f:
            json.dump({k: v for k, v in stage1_res.items() if k != "predictions" and k != "full_sample_fit_m3"}, f, indent=2, default=str)
    else:
        print(f"  skipped: only {len(ds)} origins (<45 needed for a stable out-of-sample logit).")

    print("\n=== Section 9: funding-risk omission gap ===")
    net_res = omission_gap.net_loss_var_backtest(p, args.horizon, min_train_hours=args.min_train_hours)
    gap_ds = omission_gap.build_gap_dataset(p, price_res, net_res)
    if len(gap_ds) > 10:
        gap_result = omission_gap.fit_gap_regime_test(gap_ds)
        print(f"  delta={gap_result['delta']:.4g}, one-sided p={gap_result['delta_pvalue_one_sided_H0_delta_le_0']:.4f}")
        gap_ds.to_csv(args.out / "omission_gap_dataset.csv")

        gap_rur_summary = omission_gap.gap_and_rur_summary(gap_ds)
        print("  gap/RUR summary:")
        print(gap_rur_summary.to_string())
        gap_rur_summary.to_csv(args.out / "gap_rur_summary.csv")

        mc_ds = omission_gap.build_gap_market_conditions_dataset(p, price_res, net_res)
        if len(mc_ds) > 15:
            mc_result = omission_gap.fit_gap_market_conditions_regression(mc_ds)
            print(f"  market-conditions gap regression: n={mc_result['n']}, R^2={mc_result['r_squared']:.4f}")
            for name, coef in mc_result["params"].items():
                print(f"    {name}: {coef:.4g} (p={mc_result['pvalues_two_sided'][name]:.4f})")
            mc_ds.to_csv(args.out / "omission_gap_market_conditions_dataset.csv")
            with open(args.out / "gap_market_conditions_regression.json", "w") as f:
                json.dump({k: v for k, v in mc_result.items() if k != "fit"}, f, indent=2, default=str)
        else:
            print(f"  skipped market-conditions regression: only {len(mc_ds)} origins.")
    else:
        print(f"  skipped: only {len(gap_ds)} origins.")

    if args.compute_es_gap:
        print("\n=== Section 11 item 5: expected-shortfall gap (opt-in, slow) ===")
        price_es = omission_gap.price_es_backtest(p, args.horizon, min_train_hours=args.min_train_hours)
        net_es = omission_gap.net_es_backtest(p, args.horizon, min_train_hours=args.min_train_hours)
        price_es.to_csv(args.out / "es_price_backtest.csv")
        net_es.to_csv(args.out / "es_net_backtest.csv")

        es_ds = omission_gap.build_es_gap_dataset(p, price_es, net_es)
        print(f"  es_gap dataset: {len(es_ds)} origins")
        if len(es_ds) > 10:
            es_ds.to_csv(args.out / "es_gap_dataset.csv")
            es_summary = omission_gap.es_gap_summary(es_ds)
            print(es_summary.to_string())
            es_summary.to_csv(args.out / "es_gap_summary.csv")
        else:
            print(f"  skipped ES-gap summary: only {len(es_ds)} origins.")

    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
