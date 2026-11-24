"""Proposal Section 4.12: intra-horizon cash-timing risk (path-level stage).

Research question: how much extra capital does funding require to keep a
leveraged long position open, beyond the capital needed for price risk, and
can that requirement be anticipated?

Path targets, for a long position of notional 1 opened at the end of hour t,
over the h hours t+1..t+h (the window funding_cost.forward_sum's docstring
documents; NOTE 2026-09-28: forward_sum's implementation actually sums
t..t+h, h+1 hours including the origin hour -- see the report, so Terminal
here equals C_signed_H{h} minus f_eff[t], not C_signed_H{h} itself):

    L^P_s   = -(M_{t+s} - M_t) / M_t          price loss after s hours
    C_s     = sum_{j=1..s} f_{t+j}             signed cumulative funding
    D^P     = max(0, max_s L^P_s)              worst price-only collateral draw
    D       = max(0, max_s (L^P_s + C_s))      worst draw with funding
    Peak    = max(0, max_s C_s)                worst funding-only draw
    Terminal= C_h                              (= C_signed_H{h})

The capital baseline for price risk is the *within-horizon* price drawdown
D^P (margin-period-of-risk convention: margin covers peak, not terminal,
exposure), so Delta K_q = Q_q[D] - Q_q[D^P] isolates funding's contribution.

This module builds the targets and runs the two unconditional tests:
  H1 timing      -- Q_q[Peak] vs Q_q[Terminal], stationary-bootstrap CI;
  H2 compounding -- observed Delta K vs Delta K with funding paths randomly
                    re-paired to other origins' price paths (permutation test).
H3 (reserve adequacy) is out-of-sample and lives in run_cash_timing.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

LEVELS = (0.95, 0.99)
LEVERAGE_FOR_REPORTING = 10


def path_arrays(panel: pd.DataFrame, h: int):
    """(n-h, h) arrays of L^P_s and C_s for every start hour t, plus a mask of
    windows with no missing mark_price / f_eff hour (origin t itself included
    for the price base)."""
    M = panel["mark_price"].to_numpy(dtype=float)
    f = panel["f_eff"].to_numpy(dtype=float)
    n = len(M)
    m = n - h
    WM = sliding_window_view(M[1:], h)[:m]
    base = M[:m, None]
    lp = -(WM - base) / base

    # F[k] = sum f[0..k-1], so sum f[t+1..t+s] = F[t+s+1] - F[t+1].
    F = np.concatenate(([0.0], np.cumsum(np.nan_to_num(f, nan=0.0))))
    C = sliding_window_view(F[2:], h)[:m] - F[1 : m + 1, None]

    fnan = np.concatenate(([0], np.cumsum(np.isnan(f))))
    mnan = np.concatenate(([0], np.cumsum(np.isnan(M))))
    f_missing = (fnan[h + 1 : h + 1 + m] - fnan[1 : 1 + m]) > 0
    m_missing = ((mnan[h + 1 : h + 1 + m] - mnan[1 : 1 + m]) > 0) | np.isnan(M[:m])
    complete = ~(f_missing | m_missing)
    return lp, C, complete


def path_stats(lp: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "D_P": np.maximum(0.0, lp.max(axis=-1)),
        "D": np.maximum(0.0, (lp + C).max(axis=-1)),
        "Peak": np.maximum(0.0, C.max(axis=-1)),
        "Terminal": C[..., -1],
        "RunMax": C.max(axis=-1),  # unfloored running max: the Section 12 "PeakDraw" used for Excess
    }


def build_path_targets(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per-hour forward-looking path targets (NaN where the window is incomplete
    or runs past the sample end). Like every target in funding_cost.py, these
    must never be used as predictors dated t."""
    lp, C, complete = path_arrays(panel, h)
    stats = path_stats(lp, C)
    out = pd.DataFrame(index=panel.index)
    for k, v in stats.items():
        col = np.full(len(panel), np.nan)
        col[: len(v)] = np.where(complete, v, np.nan)
        out[f"{k}_H{h}"] = col
    return out


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        idx[i] = rng.integers(n) if rng.random() < 1.0 / mean_block else (idx[i - 1] + 1) % n
    return idx


def timing_test(peak: np.ndarray, terminal: np.ndarray, runmax: np.ndarray | None = None, levels=LEVELS,
                n_boot: int = 2000, mean_block: float = 5.0, seed: int = 0) -> list[dict]:
    """H1: is the q-quantile of the peak funding draw materially above that of
    the terminal draw? Stationary bootstrap over the (time-ordered) origins
    for a CI on the difference."""
    rng = np.random.default_rng(seed)
    n = len(peak)
    boots = np.array([stationary_bootstrap_indices(n, mean_block, rng) for _ in range(n_boot)])
    # Two different estimands, both reported:
    #   diff     = Q_q[Peak] - Q_q[Terminal]: does timing raise the *tail* reserve?
    #   Q_excess = Q_q[RunMax - Terminal]: the within-window temporary cash need,
    #              with RunMax the unfloored running max of cumulative signed
    #              funding -- report Section 12's "Excess" (reproduces its
    #              published 1.76% / 3.72% of collateral at 10x exactly).
    excess = (peak if runmax is None else runmax) - terminal
    rows = []
    for q in levels:
        diff = np.quantile(peak, q) - np.quantile(terminal, q)
        bd = np.quantile(peak[boots], q, axis=1) - np.quantile(terminal[boots], q, axis=1)
        be = np.quantile(excess[boots], q, axis=1)
        rows.append({
            "q": q, "n": n,
            "Q_peak": float(np.quantile(peak, q)), "Q_terminal": float(np.quantile(terminal, q)),
            "diff": float(diff), "ci_lo": float(np.quantile(bd, 0.025)), "ci_hi": float(np.quantile(bd, 0.975)),
            "Q_excess": float(np.quantile(excess, q)),
            "excess_ci_lo": float(np.quantile(be, 0.025)), "excess_ci_hi": float(np.quantile(be, 0.975)),
            "share_peak_gt_terminal": float(np.mean(peak > terminal + 1e-15)),
        })
    return rows


def compounding_test(lp: np.ndarray, C: np.ndarray, levels=LEVELS, n_perm: int = 2000, seed: int = 0) -> list[dict]:
    """H2: does funding tend to be paid when price losses have already drained
    collateral? Compare observed Delta K with Delta K after randomly re-pairing
    each origin's funding path with another origin's price path: the marginal
    distributions of both are preserved, only their timing link is broken."""
    rng = np.random.default_rng(seed)
    obs = path_stats(lp, C)
    d_p = obs["D_P"]
    rows = []
    perm_D = np.empty((n_perm, len(d_p)))
    for b in range(n_perm):
        pi = rng.permutation(len(d_p))
        perm_D[b] = np.maximum(0.0, (lp + C[pi]).max(axis=-1))
    for q in levels:
        dk_obs = np.quantile(obs["D"], q) - np.quantile(d_p, q)
        dk_perm = np.quantile(perm_D, q, axis=1) - np.quantile(d_p, q)
        rows.append({
            "q": q, "n": len(d_p),
            "deltaK_obs": float(dk_obs), "deltaK_perm_mean": float(dk_perm.mean()),
            "deltaK_perm_p95": float(np.quantile(dk_perm, 0.95)),
            "p_one_sided": float((1 + np.sum(dk_perm >= dk_obs)) / (1 + n_perm)),
        })
    mean_obs = float(np.mean(obs["D"] - d_p))
    mean_perm = (perm_D - d_p).mean(axis=1)
    rows.append({"q": "mean", "n": len(d_p), "deltaK_obs": mean_obs, "deltaK_perm_mean": float(mean_perm.mean()),
                 "deltaK_perm_p95": float(np.quantile(mean_perm, 0.95)),
                 "p_one_sided": float((1 + np.sum(mean_perm >= mean_obs)) / (1 + n_perm))})
    return rows
