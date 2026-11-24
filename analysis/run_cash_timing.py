"""Proposal Section 4.12: run the path-level cash-timing stage for one horizon.

Reuses a completed run_pipeline.py panel (only mark_price, f_eff and the
forecast features are used; the path targets are rebuilt here over hours
t+1..t+h by analysis/cash_timing.py).

Outputs, for horizon h:
  h1_timing_test.csv            H1: Q_q[Peak] vs Q_q[Terminal] (non-overlapping origins, and overlapping robustness)
  h2_compounding_test.csv       H2: observed Delta K vs funding/price re-pairing permutations
  forecasts_<model>.csv         per-origin forecasts of Q_q[D^P], Q_q[D], Q_q[Peak] + realised path stats
  h3_reserve_evaluation.csv     H3: shortfall rate / coverage tests / capital tied up, per model x rule x level
  deltaK_forecast_summary.csv   forecast Delta K = Q_q[D] - Q_q[D^P] by model
  summary.json

Usage:
    python -m analysis.run_cash_timing --horizon 72 \
        --base-run analysis/output/full_run_fwdsum_fix/H72 \
        --out analysis/output/cash_timing_2026_09_28/H72
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtesting, cash_timing, config, recalibration
from .models import base, joint_path

MODELS = {
    "joint_bootstrap": (joint_path.joint_bootstrap_path, {}),
    "direct_qr": (joint_path.direct_quantile_path, {}),  # path_targets injected below
    "joint_sim": (joint_path.joint_sim_path, {}),
}
RECAL_GAMMA = 0.02


def evaluate_rule(capital: pd.Series, realised: pd.Series, q: float) -> dict:
    df = pd.DataFrame({"var_forecast": capital, "realized": realised}).dropna()
    df["exception"] = (df["realized"] > df["var_forecast"]).astype(float)
    r = backtesting.backtest_report(df, q)
    return {"n": r["n_forecasts"], "shortfalls": r["realized_exceptions"], "expected": r["expected_exceptions"],
            "shortfall_rate": r["realized_exceptions"] / max(r["n_forecasts"], 1),
            "kupiec_p": r["kupiec_uc"]["pvalue"], "cc_p": r["christoffersen_conditional_coverage"]["pvalue"],
            "qloss": r["quantile_loss"], "mean_capital": float(df["var_forecast"].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, required=True, choices=config.HORIZONS_HOURS)
    ap.add_argument("--base-run", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--min-train-hours", type=int, default=config.MIN_TRAIN_HOURS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    h, levels = args.horizon, cash_timing.LEVELS
    k_lev = cash_timing.LEVERAGE_FOR_REPORTING

    panel = pd.read_parquet(args.base_run / "panel.parquet")
    targets = cash_timing.build_path_targets(panel, h)
    lp_all, C_all, complete = cash_timing.path_arrays(panel, h)
    origins = base.non_overlapping_origins(panel.index, h, args.min_train_hours)
    pos_of = {ts: i for i, ts in enumerate(panel.index)}
    opos = np.array([pos_of[o] for o in origins])
    ok = complete[opos]
    print(f"panel {len(panel)} hours; {len(origins)} non-overlapping origins, {int(ok.sum())} with a complete {h}h path", flush=True)
    summary = {"horizon": h, "n_origins": len(origins), "n_complete": int(ok.sum())}

    # --- H1 and H2: unconditional tests on realised paths ------------------
    lp_o, C_o = lp_all[opos[ok]], C_all[opos[ok]]
    st = cash_timing.path_stats(lp_o, C_o)
    h1 = pd.DataFrame(cash_timing.timing_test(st["Peak"], st["Terminal"], st["RunMax"]))
    h1.insert(0, "sample", "non_overlapping")
    all_ok = np.where(complete)[0]
    st_all = cash_timing.path_stats(lp_all[all_ok], C_all[all_ok])
    h1b = pd.DataFrame(cash_timing.timing_test(st_all["Peak"], st_all["Terminal"], st_all["RunMax"], n_boot=300, mean_block=float(h)))
    h1b.insert(0, "sample", "overlapping_all_hours")
    h1 = pd.concat([h1, h1b], ignore_index=True)
    h1.to_csv(args.out / "h1_timing_test.csv", index=False)
    h2 = pd.DataFrame(cash_timing.compounding_test(lp_o, C_o))
    h2.to_csv(args.out / "h2_compounding_test.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print("\n=== H1 timing: Q_q[Peak] vs Q_q[Terminal] (units: fraction of notional) ===")
        print(h1.to_string(index=False))
        print("\n=== H2 compounding: Delta K observed vs funding/price re-paired ===")
        print(h2.to_string(index=False))

    # --- Out-of-sample path forecasts --------------------------------------
    # Expected-funding reserve (report Section 12's naive rule): causal
    # expanding mean of the realised terminal funding draw, floored at 0.
    term = targets[f"Terminal_H{h}"]
    forecasts = {}
    for name in args.models:
        fn, kw = MODELS[name]
        kw = dict(kw)
        if name == "direct_qr":
            kw["path_targets"] = targets
        rows, t0 = [], time.time()
        for o in origins:
            p = pos_of[o]
            tf, tl, row = panel.iloc[: p + 1], panel.iloc[: max(0, p + 1 - h)], panel.iloc[p]
            try:
                r = fn(tf, tl, row, horizon_hours=h, levels=levels, **kw)
                err = None if r is not None else "insufficient data / failed fit"
            except Exception as exc:  # noqa: BLE001 -- surfaced in output
                r, err = None, str(exc)
            rec = {"origin": o, "error": err, **(r or {})}
            rec["R_expected"] = max(float(term.loc[tl.index].mean()), 0.0) if len(tl) else np.nan
            rows.append(rec)
        fc = pd.DataFrame(rows).set_index("origin")
        for k in ("D_P", "D", "Peak", "Terminal"):
            fc[f"real_{k}"] = targets.loc[fc.index, f"{k}_H{h}"]
        fc.to_csv(args.out / f"forecasts_{name}.csv")
        forecasts[name] = fc
        n_ok = fc[f"D_q{levels[0]}"].notna().sum() if f"D_q{levels[0]}" in fc else 0
        print(f"\n  [{name}] {len(fc)} origins, {n_ok} successful forecasts, {time.time() - t0:.1f}s", flush=True)

    # --- H3: reserve rules --------------------------------------------------
    rows, dk_rows = [], []
    for name, fc in forecasts.items():
        for q in levels:
            if f"D_q{q}" not in fc:
                continue
            kp = fc[f"D_P_q{q}"]
            rules = {
                "price_only": kp,
                "price_plus_expected_funding": kp + fc["R_expected"],
                "price_plus_peak_quantile": kp + fc[f"Peak_q{q}"],
                "joint_D_quantile": fc[f"D_q{q}"],
            }
            if q == levels[0]:
                bt = pd.DataFrame({"var_forecast": rules["price_plus_peak_quantile"], "realized": fc["real_D"]})
                rec = recalibration.aci_recalibrate(bt, alpha=q, gamma=RECAL_GAMMA)
                rules["price_plus_peak_recalibrated"] = rec["var_forecast"].reindex(fc.index)
            for rule, cap in rules.items():
                e = evaluate_rule(cap, fc["real_D"], q)
                e["mean_funding_allowance"] = float((cap - kp).reindex(cap.dropna().index).mean())
                rows.append({"model": name, "q": q, "rule": rule, **e})
            # The price-risk model's own calibration against the price-only draw,
            # so price-model miscalibration is not mistaken for a funding effect.
            e = evaluate_rule(kp, fc["real_D_P"], q)
            rows.append({"model": name, "q": q, "rule": "price_only_vs_price_draw", **e, "mean_funding_allowance": 0.0})
            dk = (fc[f"D_q{q}"] - kp).dropna()
            dk_rows.append({"model": name, "q": q, "n": len(dk), "mean_deltaK": float(dk.mean()),
                            "median_deltaK": float(dk.median()), "p90_deltaK": float(dk.quantile(0.9)),
                            "share_positive": float((dk > 0).mean()),
                            f"mean_deltaK_pct_collateral_{k_lev}x": float(100 * k_lev * dk.mean())})
    h3 = pd.DataFrame(rows)
    h3.to_csv(args.out / "h3_reserve_evaluation.csv", index=False)
    dks = pd.DataFrame(dk_rows)
    dks.to_csv(args.out / "deltaK_forecast_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print("\n=== H3 reserve rules (shortfall = realised D above capital) ===")
        print(h3.to_string(index=False))
        print("\n=== Forecast Delta K = Q_q[D] - Q_q[D^P] (fraction of notional) ===")
        print(dks.to_string(index=False))

    summary["h1"] = h1.to_dict(orient="records")
    summary["h2"] = h2.to_dict(orient="records")
    with open(args.out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll output written to {args.out}")


if __name__ == "__main__":
    main()
