"""Independent reconciliation of funding units, signs, leverage scaling,
collateral drain, and PACRRatio (lag audit, 2026-11-06).

Recomputes a handful of origins *from the raw Deribit ticker files* with
plain Python (gzip + csv, no pandas, no panel.py/funding_cost.py/
collateral_erosion.py code), straight from the written definitions:

  hourly funding   f_j        = mean(predicted_funding_rate over hour j) / 8      (8h rate -> hourly)
  long cost        C_long     = sum_{j=t+1}^{t+H} max(f_j, 0)
  signed cost      C_signed   = sum_{j=t+1}^{t+H} f_j                            (long pays when > 0)
  net loss (long)  L_net      = -(M_{t+H} - M_t)/M_t + C_signed                  (M = last mark in the hour)
  drain            Drain(k)   = k * C_long                                       (fraction of collateral at leverage k)
  PACRRatio (proposal Sec. 4.13, inverse contract, USD notional N, direction d = +1 long / -1 short):
     S_h      = N / M_{t+h}                                         position size, BTC
     PricePnL = d * N * (1/M_t - 1/M_{t+h})                         BTC
     FundPnL  = sum_{j<=h} -d * f_{t+j} * N / M_{t+j}               BTC, each hour on its own size
     MM_h     = (base(t+h) + 0.00005 * S_h) * S_h                 BTC, base = 0.55% before / 1% from 2022-01-11
     Equity_h = C0 + PricePnL + FundPnL,  C0 = (N/L) / M_t
     PACRRatio = max_h [MM_h - Equity_h]^+ / C0

and compares them with the pipeline's own values (panel.parquet targets,
and PACRRatio from collateral_erosion.py when that module is importable).

Usage:
    python -m analysis.reconcile_units --panel <run>/H72/panel.parquet --n 8 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import math
import random
from collections import defaultdict
from pathlib import Path

from . import config

MM_RATE_CHANGE = dt.datetime(2022, 1, 11, tzinfo=dt.timezone.utc)


def mm_base(hour: dt.datetime) -> float:
    """Deribit MM base rate at `hour`: 0.55% representative pre-change, 1% from 2022-01-11."""
    return 0.01 if hour >= MM_RATE_CHANGE else 0.0055


MM_PER_BTC = 0.00005    # 0.005% per BTC of position size
NOTIONAL = 100_000.0


def _raw_hours(day: dt.date) -> dict[dt.datetime, dict]:
    """Per-hour mean predicted_funding_rate and last mark_price for one raw daily file."""
    month = day.strftime("%Y-%m")
    f = config.FUNDING_DIR / month / f"derivatives_drbt_perpetual-future_btcperpetual_{day.isoformat()}.csv.gz"
    acc: dict[dt.datetime, dict] = defaultdict(lambda: {"pf": [], "mark": None, "mark_ts": -1})
    if not f.exists():
        return {}
    with gzip.open(f, "rt") as fh:
        for row in csv.DictReader(fh):
            ts = int(float(row["timestamp"]))
            hour = dt.datetime.fromtimestamp(ts - ts % 3600, tz=dt.timezone.utc)
            a = acc[hour]
            pf = row.get("predicted_funding_rate", "")
            if pf not in ("", None):
                a["pf"].append(float(pf))
            if row.get("mark_price") not in ("", None) and ts >= a["mark_ts"]:
                a["mark"], a["mark_ts"] = float(row["mark_price"]), ts
    return {h: {"f": (sum(a["pf"]) / len(a["pf"]) / 8.0) if a["pf"] else None, "mark": a["mark"]} for h, a in acc.items()}


def independent_origin(origin: dt.datetime, H: int) -> dict | None:
    hours = {}
    for d in range((H // 24) + 2):
        hours.update(_raw_hours((origin + dt.timedelta(days=d)).date()))
    seq = [origin + dt.timedelta(hours=j) for j in range(H + 1)]
    if any(h not in hours or hours[h]["mark"] is None for h in seq) or any(hours[h]["f"] is None for h in seq[1:]):
        return None  # missing hour inside the window -> pipeline target must be NaN too
    f = [hours[h]["f"] for h in seq[1:]]
    m = [hours[h]["mark"] for h in seq]
    out = {
        "C_long": sum(max(x, 0.0) for x in f),
        "C_signed": sum(f),
        "L_net": -(m[-1] - m[0]) / m[0] + sum(f),
    }
    for k in config.LEVERAGE_LEVELS:
        out[f"Drain_k{k}"] = k * out["C_long"]
    for d, dname in ((1, "long"), (-1, "short")):
        for L in (2, 10, 20):
            c0 = (NOTIONAL / L) / m[0]
            fund, worst = 0.0, 0.0
            for h in range(1, H + 1):
                fund += -d * f[h - 1] * NOTIONAL / m[h]
                size = NOTIONAL / m[h]
                mm = (mm_base(seq[h]) + MM_PER_BTC * size) * size
                equity = c0 + d * NOTIONAL * (1 / m[0] - 1 / m[h]) + fund
                worst = max(worst, mm - equity)
            out[f"PACRRatio_{dname}_L{L}"] = worst / c0
    return out


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    p = pd.read_parquet(args.panel)
    rng = random.Random(args.seed)
    try:
        from . import collateral_erosion as ce  # noqa: F401
    except Exception as exc:  # cloud-only / unavailable: report, do not fake the comparison
        ce = None
        print(f"collateral_erosion.py not importable ({exc}); PACRRatio is computed independently but not compared.")

    worst_rel = defaultdict(float)
    for H in config.HORIZONS_HOURS:
        cand = p.index[p[f"C_long_H{H}"].notna()]
        picks = sorted(rng.sample(list(cand), min(args.n, len(cand))))
        # Also force one origin whose window touches a missing hour, if any.
        gap_cand = p.index[p[f"C_long_H{H}"].isna() & p["f_eff"].notna()][: -H] if len(p) > H else []
        if len(gap_cand):
            picks.append(gap_cand[len(gap_cand) // 2])
        print(f"\n=== H={H}: {len(picks)} origins ===")
        for o in picks:
            ind = independent_origin(o.to_pydatetime(), H)
            row = p.loc[o]
            if ind is None:
                ok = pd.isna(row[f"C_long_H{H}"])
                print(f"  {o}  window has a missing hour -> independent=NaN; pipeline C_long={'NaN' if ok else row[f'C_long_H{H}']}  {'OK' if ok else 'MISMATCH'}")
                worst_rel["nan_consistency"] = max(worst_rel["nan_consistency"], 0.0 if ok else 1.0)
                continue
            pairs = {"C_long": f"C_long_H{H}", "C_signed": f"C_signed_H{H}", "L_net": f"L_net_H{H}"}
            pairs |= {f"Drain_k{k}": f"CollateralDrain_H{H}_k{k}" for k in config.LEVERAGE_LEVELS}
            parts = []
            for a, b in pairs.items():
                rel = abs(ind[a] - row[b]) / max(abs(ind[a]), 1e-12)
                worst_rel[a] = max(worst_rel[a], rel)
                parts.append(f"{a}={ind[a]:.6g}/{row[b]:.6g}")
            print(f"  {o}  independent/pipeline: " + "; ".join(parts))
            if ce is not None:
                pos = p.index.get_loc(o)
                window = p.iloc[pos : pos + H + 1]
                ds = ce.add_leverage_columns(ce.build_pacr_dataset(window, H), leverage_grid=(2, 10, 20))
                pp = []
                for k, v in ind.items():
                    if not k.startswith("PACR"):
                        continue
                    got = float(ds[k].iloc[0])
                    worst_rel["PACRRatio"] = max(worst_rel["PACRRatio"], abs(v - got) / max(abs(v), 1e-12) if v else abs(got))
                    pp.append(f"{k.split('_', 1)[1]}={v:.4g}/{got:.4g}")
                print("      PACRRatio independent/pipeline: " + ", ".join(pp))
    print("\nmax relative difference, independent vs pipeline:")
    for k, v in worst_rel.items():
        print(f"  {k:16s} {v:.2e}")


if __name__ == "__main__":
    main()
