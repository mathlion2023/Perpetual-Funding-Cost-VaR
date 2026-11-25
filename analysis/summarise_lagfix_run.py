"""Summarise a run_pipeline rerun (both horizons) for the audit record and
the paper, and compare it with a reference run.

Conventions follow new_models_inference.py: the all-model common sample;
Diebold-Mariano with 2 unweighted autocovariance lags and Benjamini-Hochberg
FDR within each horizon (55 pairs); adaptive conformal recalibration with
gamma = 0.02, applied to each model's full series and then restricted to
the common sample.

Usage:
    python -m analysis.summarise_lagfix_run \
        --run analysis/output/full_run_2026_11_06_lagfix_2022_2025 \
        --ref analysis/output/full_run_2026_10_20_fundingrate_fix_2022_2025
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtesting, collateral_erosion as ce, config, recalibration
from .new_models_inference import bh_fdr, dm_hac

MODELS = ["historical_simulation", "direct_quantile", "caviar", "har_quantile", "arx_garch_t", "gas_t",
          "ms_garch_t", "sv_t", "svj_t", "rough_vol", "stationary_bootstrap"]
HAC_LAGS = 2
RECAL_GAMMA = 0.02
CAP_HOURLY = 0.005 / 8


def _runs(s: pd.Series) -> pd.Series:
    """Lengths of consecutive-True runs in a boolean series."""
    grp = (s != s.shift()).cumsum()
    lens = s.groupby(grp).sum()
    return lens[s.groupby(grp).first()]


def panel_stats(p: pd.DataFrame) -> dict:
    out = {"hours": len(p), "start": str(p.index.min()), "end": str(p.index.max())}
    for col in ("f_eff", "rv_1h", "mark_price"):
        na = p[col].isna()
        r = _runs(na)
        out[f"{col}_missing_hours"] = int(na.sum())
        out[f"{col}_missing_runs"] = int(len(r))
        out[f"{col}_longest_gap_hours"] = int(r.max()) if len(r) else 0
    f = p["f_eff"].dropna()
    out["share_f_zero"] = float((f == 0).mean())
    out["share_f_at_cap"] = float((f.abs() >= CAP_HOURLY * (1 - 1e-9)).mean())
    out["max_abs_f_over_cap"] = float(f.abs().max() / CAP_HOURLY)
    return out


def load_backtests(run_h: Path, H: int) -> dict[str, pd.DataFrame]:
    return {m: pd.read_csv(run_h / f"backtest_{m}_H{H}.csv", index_col="origin", parse_dates=["origin"]) for m in MODELS}


def model_table(res: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    common = backtesting.common_sample(res)
    idx = next(iter(common.values())).index
    rows = []
    for m in MODELS:
        r = backtesting.backtest_report(common[m], config.VAR_ALPHA)
        rec = recalibration.aci_recalibrate(res[m], config.VAR_ALPHA, gamma=RECAL_GAMMA)
        rr = backtesting.backtest_report(rec.reindex(idx).dropna(subset=["var_forecast"]), config.VAR_ALPHA)
        rows.append({"model": m, "n": r["n_forecasts"], "exceptions": r["realized_exceptions"],
                     "kupiec_p": r["kupiec_uc"]["pvalue"], "cc_p": r["christoffersen_conditional_coverage"]["pvalue"],
                     "qloss": r["quantile_loss"], "exceptions_recal": rr["realized_exceptions"],
                     "kupiec_p_recal": rr["kupiec_uc"]["pvalue"], "cc_p_recal": rr["christoffersen_conditional_coverage"]["pvalue"],
                     "qloss_recal": rr["quantile_loss"]})
    dm_rows = []
    for a, b in itertools.combinations(MODELS, 2):
        d = dm_hac(common[a], common[b], config.VAR_ALPHA, HAC_LAGS)
        dm_rows.append({"a": a, "b": b, "n": d["n"], "DM": d["DM"], "p": d["p"]})
    dm = pd.DataFrame(dm_rows)
    dm["p_fdr"] = bh_fdr(dm["p"].to_numpy())
    return pd.DataFrame(rows).set_index("model"), dm, len(idx)


def pacr_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for H in config.HORIZONS_HOURS:
        ds = ce.build_pacr_dataset(p, H)
        for L in (2, 5, 10, 20):
            r = pd.Series(ce.pacr_ratio(ds["net_max_long"].to_numpy(), ds["mark_price_origin"].to_numpy(), L)).dropna()
            rows.append({"H": H, "L": L, "n": len(r), "p95": r.quantile(0.95), "p99": r.quantile(0.99),
                         "share_gt0": float((r > 0).mean()), "share_gt1": float((r > 1).mean())})
    return pd.DataFrame(rows)


def drain_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for H in config.HORIZONS_HOURS:
        for k in config.LEVERAGE_LEVELS:
            s = p[f"CollateralDrain_H{H}_k{k}"].dropna()
            rows.append({"H": H, "k": k, "n": len(s), "median_pct": 100 * s.median(),
                         "p95_pct": 100 * s.quantile(0.95), "p99_pct": 100 * s.quantile(0.99), "max_pct": 100 * s.max()})
    return pd.DataFrame(rows)


def stage1(run_h: Path) -> dict:
    s = json.loads((run_h / "stage1_omission_test.json").read_text())
    out = {"beta": s["beta_funding"], "p1": s["beta_funding_pvalue_one_sided_H0_beta_le_0"],
           "lr_p_two_sided": s["lr_test_m2_vs_m3"]["pvalue"],
           "lr_p_one_sided": s["lr_test_m2_vs_m3"].get("pvalue_one_sided_signed_root"),
           "firth_beta": s["firth_robustness"]["beta_funding"],
           "firth_p1": s["firth_robustness"]["beta_funding_pvalue_one_sided_H0_beta_le_0"],
           "firth_lr_p_one_sided": s["firth_robustness"]["lr_test_m2_vs_m3"].get("pvalue_one_sided_signed_root"),
           "n_origins": s.get("n_origins"), "n_events": s.get("n_events"),
           "oos": {k: {"n": v["n"], "events": v["n_events"], "brier": v["brier"], "auc": v["auc"]} for k, v in s["scores"].items()}}
    acct = run_h / "stage1_sample_accounting.json"
    if acct.exists():
        out["accounting"] = json.loads(acct.read_text())
    return out


def panel_diff(new: pd.DataFrame, old: pd.DataFrame) -> dict:
    """How many hourly values changed between the reference and new panels."""
    idx = new.index.intersection(old.index)
    out = {"hours_new": len(new), "hours_ref": len(old), "hours_common": len(idx)}
    for c in ("f_eff", "d_oi", "rv_1h", "rv_primary", "volume_shock", "basis", "C_long_H72", "C_long_H168", "L_net_H72", "L_net_H168"):
        a, b = new.loc[idx, c], old.loc[idx, c]
        both = a.notna() & b.notna()
        changed = (~np.isclose(a[both], b[both], rtol=1e-9, atol=0)).sum()
        out[c] = {"nan_new": int(a.isna().sum()), "nan_ref": int(b.isna().sum()),
                  "nan_status_changed": int((a.isna() != b.isna()).sum()), "value_changed": int(changed)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ref", type=Path)
    args = ap.parse_args()
    out_dir = args.run / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    p = pd.read_parquet(args.run / "H72" / "panel.parquet")
    summary = {"panel": panel_stats(p)}
    if args.ref:
        summary["panel_vs_ref"] = panel_diff(p, pd.read_parquet(args.ref / "H72" / "panel.parquet"))

    for H in config.HORIZONS_HOURS:
        run_h = args.run / f"H{H}"
        res = load_backtests(run_h, H)
        tab, dm, n_common = model_table(res)
        tab.to_csv(out_dir / f"model_table_common_H{H}.csv")
        dm.to_csv(out_dir / f"dm_hac_fdr_common_H{H}.csv", index=False)
        avail = backtesting.availability_summary(res)
        avail.to_csv(out_dir / f"availability_H{H}.csv")
        hs = {"scheduled_origins": int(avail["scheduled_origins"].iloc[0]), "common_origins": n_common,
              "availability": avail.to_dict(orient="index"),
              "kupiec_reject_raw": int((tab["kupiec_p"] < 0.05).sum()),
              "kupiec_reject_recal": int((tab["kupiec_p_recal"] < 0.05).sum()),
              "cc_reject_raw": int((tab["cc_p"] < 0.05).sum()),
              "cc_reject_recal": int((tab["cc_p_recal"] < 0.05).sum()),
              "dm_pairs": len(dm), "dm_sig_fdr": int((dm["p_fdr"] < 0.05).sum()),
              "stage1": stage1(run_h)}
        if args.ref:
            ref_res = load_backtests(args.ref / f"H{H}", H)
            hs["vs_ref"] = {m: {"usable_ref": int(ref_res[m].dropna(subset=["var_forecast", "realized"]).shape[0]),
                                "usable_new": int(res[m].dropna(subset=["var_forecast", "realized"]).shape[0]),
                                "exc_ref": int(ref_res[m].dropna(subset=["var_forecast", "realized"])["exception"].sum()),
                                "exc_new": int(res[m].dropna(subset=["var_forecast", "realized"])["exception"].sum()),
                                "qloss_ref": backtesting.quantile_loss(ref_res[m], config.VAR_ALPHA),
                                "qloss_new": backtesting.quantile_loss(res[m], config.VAR_ALPHA)} for m in MODELS}
            ref_s1 = args.ref / f"H{H}" / "stage1_omission_test.json"
            if ref_s1.exists():
                hs["stage1_ref"] = stage1(args.ref / f"H{H}")
        summary[f"H{H}"] = hs
        with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
            print(f"\n=== H={H}: common origins {n_common} / scheduled {hs['scheduled_origins']} ===")
            print(tab.to_string())
            print(f"Kupiec rejections raw {hs['kupiec_reject_raw']}/11, recal {hs['kupiec_reject_recal']}/11; "
                  f"DM significant after FDR {hs['dm_sig_fdr']}/{len(dm)}")

    pacr = pacr_table(p)
    pacr.to_csv(out_dir / "pacr_long_table.csv", index=False)
    drain = drain_table(p)
    drain.to_csv(out_dir / "drain_table.csv", index=False)
    summary["pacr_long"] = pacr.to_dict(orient="records")
    summary["drain"] = drain.to_dict(orient="records")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4g}".format):
        print("\nPACR (long):\n" + pacr.to_string(index=False))
        print("\nDrain:\n" + drain.to_string(index=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwritten {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
