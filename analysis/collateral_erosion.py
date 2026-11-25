"""Revised design (Tina, 2026-09-29): path-dependent collateral erosion.

Core object is the Peak Additional Collateral Requirement (PACR): the extra
collateral that would have been needed at the forecast origin to keep the
realised account-equity path above the Deribit Standard Margin maintenance
requirement throughout the holding horizon, for a single segregated BTC
inverse-perpetual position (see maintenance_margin_calculation_deribit_btc_
perpetual.docx). Baseline position-size convention: fixed USD notional
N_USD = 100,000 (the worked example in that document); BTC-equivalent size
is recomputed each hour as N_USD / mark_price, matching an inverse contract.

Derivation note (why PACR collapses to a closed form in leverage L):
funding and price P&L, and the maintenance-margin requirement, do not
depend on leverage at all -- leverage only sets initial collateral
C0_BTC(L) = (N_USD/L)/M_t. Writing everything as a ratio of N_USD:

    net_{t,h}(d) = mm_ratio_{t,h} - price_pnl_ratio_{t,h}(d) - cum_funding_ratio_{t,h}(d)

is leverage-free, and (see module docstring in the accompanying analysis
notes) PACRRatio_t,H(d,L) = max(L * M_t * NetMax_t(d) - 1, 0), where
NetMax_t(d) = max_{h=1..H} net_{t,h}(d). So the expensive path computation
(sliding windows over mark_price/f_eff) is done once per (origin, horizon,
direction); every leverage level is then a free closed-form transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

N_USD = 100_000.0

# VALIDATED (2026-09-29) against archived Deribit documentation (web.archive.org
# snapshots of deribit.com/pages/docs/{futures,perpetual}, plus Deribit's own
# "New SM Margin Model" blog post): the additive structure
# MM = base_rate + 0.005%*S_BTC was in continuous effect for BTC perpetual/
# futures from at least 2020-01 through the tiered-leverage replacement, which
# Deribit's own announcement states took effect "some time in August 2026" --
# i.e. after this project's entire 2020-2025 sample. The 0.005%/BTC size
# coefficient is confirmed stable across every snapshot checked (2020-01,
# 2021-09, and the 2022-01-11 change below). The base rate drifted within the
# sample: 0.575% (BTC perpetual, 2020-01-29 snapshot) down to 0.525%
# (2021-09-23 snapshot, both futures and perpetual by then).
#
# UPDATE (2026-10-07): the "11 January 2022" Deribit knowledge-base article
# (support.deribit.com/hc/en-us/articles/25944826427293-11-January-2022),
# previously unrecoverable from the archive, was retrieved directly and
# confirms the base rate was raised again, from 0.525% to 1%, effective
# 2022-01-11 -- bundled in the same announcement with an unrelated increase
# to the BTC liquidation fee (0.5% to 0.75%). No further base-rate change was
# found between 2022-01-11 and the tiered-leverage replacement (~2026-08), so
# 1% is the applicable rate for the remainder of the sample. This means the
# single representative rate used below (0.55%, treated as constant for the
# full 2020-2025 sample) understates the true rate for the ~70% of the
# sample's hourly origins from 2022-01-11 onward, where the true rate was 1%.
# A sensitivity check (holding price/funding paths fixed, re-deriving PACR at
# each candidate rate) found this immaterial to any headline H1/H2
# conclusion: moving across the two pre-2022 rates changes the 20x PACRRatio
# >100% share by at most 0.07 percentage points at either horizon; using the
# confirmed 1% rate for the full sample instead (an upper-bound stress check)
# shifts the same shares by only 0.7-1.2 percentage points, consistent with
# MM's structurally small contribution to NetMax relative to the price and
# funding terms. See Topic1_Collateral_Erosion_Pilot_Results.tex's "Note on
# the maintenance-margin rate assumption" for the full comparison. A
# time-varying rate schedule (0.55% pre-2022-01-11, 1% after) would be a more
# faithful refinement than the constant mid-sample value used here, and is
# not yet implemented in net_max_arrays below.
#
# maintenance_margin_calculation_deribit_btc_perpetual.docx's stated base
# rate (1%) is corrected here: 1% is Deribit's documented INITIAL-margin base
# rate pre-2022 (confirmed in the same archived snapshots, "Initial margin =
# 1% + Position Size*0.005%"), not the maintenance-margin rate -- the source
# document conflated the two. (Post-2022-01-11, INITIAL margin's own base
# rate was itself raised again, to 2%, while MAINTENANCE margin's base rate
# became 1% -- the two no longer coincide numerically the way the source
# document implied even by accident.) This correction has negligible effect
# on any previously-reported PACR figure: MM's own contribution to NetMax is
# orders of magnitude smaller than the price/funding terms (MM ~ 1/M_t *
# ~0.5-1%, a fraction of a percent of notional per BTC of size), so this was
# already true whether the base rate read 1%, 0.55%, or the regime-correct
# time-varying value. See Topic1_Full_History_Results_Report.md §21.1 for
# sourcing detail.
MM_BASE_RATE = 0.0055
MM_SIZE_COEF = 0.00005
# Time-varying schedule (lag/units audit, 2026-11-06): the note above
# documents that the base rate was 1% from 2022-01-11, but the constant 0.55%
# was still applied to every hour, understating MM for the whole 2022-2025
# sample. MM is now evaluated at each hour's own base rate: MM_BASE_RATE
# (representative pre-2022 value) before the change, 1% from it.
# Set USE_MM_SCHEDULE = False to reproduce the old constant-rate convention.
MM_BASE_RATE_FROM_2022_01_11 = 0.01
MM_RATE_CHANGE = pd.Timestamp("2022-01-11", tz="UTC")
USE_MM_SCHEDULE = True


def mm_base_rate_series(index: pd.DatetimeIndex) -> np.ndarray:
    """Maintenance-margin base rate applying at each hour of `index`."""
    if not USE_MM_SCHEDULE:
        return np.full(len(index), MM_BASE_RATE)
    return np.where(index >= MM_RATE_CHANGE, MM_BASE_RATE_FROM_2022_01_11, MM_BASE_RATE)
LEVERAGE_GRID = (2, 5, 10, 20)
HORIZONS_HOURS = (72, 168)
DIRECTIONS = {"long": 1, "short": -1}


def mm_rate(s_btc: np.ndarray | float, base_rate: np.ndarray | float = MM_BASE_RATE) -> np.ndarray | float:
    """Deribit Standard Margin BTC-perpetual maintenance-margin rate."""
    return base_rate + MM_SIZE_COEF * s_btc


def maintenance_margin_btc(n_usd: float, mark_price: np.ndarray | float, as_of: pd.Timestamp | None = None):
    """Unlike `net_max_arrays` (which applies `mm_base_rate_series`'s
    time-varying schedule at each hour), this had no time argument at all and
    silently used the stale pre-2022-01-11 MM_BASE_RATE for every call (dead
    code, zero call sites as of the 2026-11-25 code audit, but a trap for any
    future caller). Pass `as_of` to get the correct rate for that hour;
    omitting it keeps the old (pre-2022) constant-rate behaviour."""
    s = n_usd / mark_price
    base_rate = MM_BASE_RATE if as_of is None else mm_base_rate_series(pd.DatetimeIndex([as_of]))[0]
    return s * mm_rate(s, base_rate)


def _cumsum_window(series: np.ndarray, h: int, m: int) -> np.ndarray:
    """sum_{k=i+1..i+j} series[k] for every origin i (0-indexed) and offset
    j=1..h, vectorised via the same F[k]=cumsum trick used in
    cash_timing.path_arrays. `series` must be the full-length (n) array,
    not pre-sliced -- the F[2:]/F[1:m+1] offsets already encode the "start
    at k=i+1" shift."""
    F = np.concatenate(([0.0], np.cumsum(np.nan_to_num(series, nan=0.0))))
    return sliding_window_view(F[2:], h)[:m] - F[1 : m + 1, None]


def net_max_arrays(panel: pd.DataFrame, h: int, include_funding: bool = True) -> dict[str, np.ndarray]:
    """NetMax_t(d) for d in {long, short}, plus the completeness mask, for
    every origin t with a full h-hour forward window.

    include_funding=False zeroes the cum_funding_ratio term, isolating the
    price-only deficit on the identical realised mark-price path (Tina,
    2026-10-07 email, "PACR comparison with and without funding"). The
    completeness mask still requires f_eff to be non-missing in both cases,
    so the with/without comparison is always on the same (paired) set of
    origins, not each version's own largest available sample."""
    from .models.base import assert_hourly_grid

    assert_hourly_grid(panel.index, "net_max_arrays panel")  # windows below are positional
    M = panel["mark_price"].to_numpy(dtype=float)
    f = panel["f_eff"].to_numpy(dtype=float)
    n = len(M)
    m = n - h
    invM = 1.0 / M

    base_invM = invM[:m]
    W_invM = sliding_window_view(invM[1:], h)[:m]  # 1/M_{t+1..t+h}, shape (m, h)

    price_pnl_ratio_long = base_invM[:, None] - W_invM
    price_pnl_ratio_short = -price_pnl_ratio_long

    g = f * invM  # per-hour funding-cost ratio contribution (elementwise)
    cum_g = _cumsum_window(g, h, m)
    if include_funding:
        cum_funding_ratio_long = -cum_g
        cum_funding_ratio_short = cum_g
    else:
        cum_funding_ratio_long = np.zeros_like(cum_g)
        cum_funding_ratio_short = np.zeros_like(cum_g)

    s_future = N_USD * W_invM
    W_base = sliding_window_view(mm_base_rate_series(panel.index)[1:], h)[:m]  # rate at hour t+h
    mm_ratio = W_invM * mm_rate(s_future, W_base)

    net_long = mm_ratio - price_pnl_ratio_long - cum_funding_ratio_long
    net_short = mm_ratio - price_pnl_ratio_short - cum_funding_ratio_short

    fnan = np.concatenate(([0], np.cumsum(np.isnan(f))))
    mnan = np.concatenate(([0], np.cumsum(np.isnan(M))))
    f_missing = (fnan[h + 1 : h + 1 + m] - fnan[1 : 1 + m]) > 0
    m_missing = ((mnan[h + 1 : h + 1 + m] - mnan[1 : 1 + m]) > 0) | np.isnan(M[:m])
    complete = ~(f_missing | m_missing)

    return {
        "net_max_long": np.where(complete, net_long.max(axis=-1), np.nan),
        "net_max_short": np.where(complete, net_short.max(axis=-1), np.nan),
        "mark_price_origin": M[:m],
        "complete": complete,
    }


def pacr_ratio(net_max: np.ndarray, mark_price_origin: np.ndarray, leverage: int) -> np.ndarray:
    """PACRRatio = PACR_BTC / C0_BTC(L), dimensionless (both numerator and
    denominator in BTC, so the M_t/USD units of net_max cancel)."""
    return np.maximum(leverage * mark_price_origin * net_max - 1.0, 0.0)


def pacr_frac_notional(net_max: np.ndarray, mark_price_origin: np.ndarray, leverage: int) -> np.ndarray:
    """PACR expressed as a fraction of USD notional (PACR_USD / N_usd), the
    same units as VaR_price/VaR_funding elsewhere in this project, so it can
    be compared directly to a reserve = PriceReserve + FundingReserve.

    net_max carries units of BTC/USD (it is built from 1/M_t terms), so
    converting to a plain fraction of notional requires multiplying by the
    origin mark price: PACR_frac = M_t * max(NetMax - 1/(L*M_t), 0)
                                  = max(M_t*NetMax - 1/L, 0) = PACRRatio / L.
    (A units bug in an earlier pass compared the un-multiplied BTC/USD
    quantity directly against Reserve and silently produced ~0 shortfalls
    everywhere -- this function is the fix.)"""
    return np.maximum(mark_price_origin * net_max - 1.0 / leverage, 0.0)


def build_pacr_dataset(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """One row per origin with NetMax_long/short and mark_price_origin --
    everything needed to derive PACR/PACRRatio at any leverage without
    recomputing the path arrays."""
    arr = net_max_arrays(panel, h)
    idx = panel.index[: len(arr["net_max_long"])]
    return pd.DataFrame(
        {
            "net_max_long": arr["net_max_long"],
            "net_max_short": arr["net_max_short"],
            "mark_price_origin": arr["mark_price_origin"],
        },
        index=idx,
    )


def add_leverage_columns(dataset: pd.DataFrame, leverage_grid=LEVERAGE_GRID) -> pd.DataFrame:
    out = dataset.copy()
    for direction in ("long", "short"):
        nm = out[f"net_max_{direction}"]
        for lev in leverage_grid:
            out[f"PACRRatio_{direction}_L{lev}"] = pacr_ratio(nm.to_numpy(), out["mark_price_origin"].to_numpy(), lev)
    return out


# --- H1: descriptive funding erosion (linear fixed-USD-notional approximation,
# explicitly permitted for this descriptive measure by Section 4.2 of the
# revised design) -- reuses c_long/c_short/C_long_H/C_short_H already built
# by funding_cost.py, scaled by leverage. -----------------------------------

SUMMARY_PERCENTILES = (0.50, 0.75, 0.90, 0.95, 0.99)


def _tail_summary(s: pd.Series) -> dict:
    s = s.dropna()
    row = {"n": int(len(s)), "mean": float(s.mean()), "median": float(s.median())}
    for q in SUMMARY_PERCENTILES:
        row[f"p{int(q * 100)}"] = float(s.quantile(q))
    for q in (0.95, 0.99):
        tail = s[s >= s.quantile(q)]
        row[f"es{int(q * 100)}"] = float(tail.mean()) if len(tail) else np.nan
    return row


def h1_funding_erosion_summary(panel: pd.DataFrame, horizons=HORIZONS_HOURS, leverage_grid=LEVERAGE_GRID) -> pd.DataFrame:
    rows = []
    for h in horizons:
        c_signed = panel[f"C_signed_H{h}"] if f"C_signed_H{h}" in panel else panel[f"C_long_H{h}"] - panel[f"C_short_H{h}"]
        for direction, d in DIRECTIONS.items():
            gross = panel[f"C_long_H{h}"] if d == 1 else panel[f"C_short_H{h}"]
            net_effect_unlev = -d * c_signed  # FundingPnL(d) per unit notional, before leverage scaling
            for lev in leverage_grid:
                net_row = _tail_summary(net_effect_unlev * lev)
                gross_row = _tail_summary(gross * lev)
                thresholds = {}
                g = (gross * lev).dropna()
                for thr in (0.01, 0.05, 0.10, 0.25):
                    thresholds[f"share_gt_{int(thr * 100)}pct"] = float((g > thr).mean()) if len(g) else np.nan
                rows.append({"horizon": h, "direction": direction, "leverage": lev,
                             **{f"net_{k}": v for k, v in net_row.items()},
                             **{f"gross_{k}": v for k, v in gross_row.items()},
                             **thresholds})
    return pd.DataFrame(rows)


# --- H2: PACR / PACRRatio across the leverage x horizon x direction grid ---

def h2_pacr_grid_summary(panel: pd.DataFrame, horizons=HORIZONS_HOURS, leverage_grid=LEVERAGE_GRID) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ds = build_pacr_dataset(panel, h)
        for direction in ("long", "short"):
            for lev in leverage_grid:
                pr = pd.Series(pacr_ratio(ds[f"net_max_{direction}"].to_numpy(), ds["mark_price_origin"].to_numpy(), lev),
                                index=ds.index)
                row = _tail_summary(pr)
                pr_d = pr.dropna()
                for thr in (0.05, 0.10, 0.20, 0.50, 1.00):
                    row[f"share_pacr_ratio_gt_{thr}"] = float((pr_d > thr).mean()) if len(pr_d) else np.nan
                rows.append({"horizon": h, "direction": direction, "leverage": lev, **row})
    return pd.DataFrame(rows)


# --- Funding attribution: paired PACR with vs without funding, same origins, ---
# same realised price path (Tina, 2026-10-07 email). Isolates funding's own
# contribution to PACR from price risk. --------------------------------------

def build_pacr_funding_attribution(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """NetMax_t(d) computed twice on identical origins and the identical
    realised mark-price path: once with signed funding included (matches
    net_max_arrays' default), once with funding zeroed out. Both share the
    same completeness mask (requires f_eff non-missing even in the
    without-funding leg), so every row is a genuine paired comparison, not
    each version's own largest available sample."""
    with_f = net_max_arrays(panel, h, include_funding=True)
    without_f = net_max_arrays(panel, h, include_funding=False)
    idx = panel.index[: len(with_f["net_max_long"])]
    return pd.DataFrame(
        {
            "net_max_long_with": with_f["net_max_long"],
            "net_max_long_without": without_f["net_max_long"],
            "net_max_short_with": with_f["net_max_short"],
            "net_max_short_without": without_f["net_max_short"],
            "mark_price_origin": with_f["mark_price_origin"],
        },
        index=idx,
    )


def h2b_funding_attribution_summary(panel: pd.DataFrame, horizons=HORIZONS_HOURS, leverage_grid=LEVERAGE_GRID) -> pd.DataFrame:
    """For each origin, direction, leverage, horizon: PACRRatio with funding
    vs. without, on the identical realised path. Reports the delta
    distribution and the four path categories Tina's email asks for --
    funding creates / removes / increases / decreases a collateral deficit
    -- plus the residual "no deficit either way" share for completeness."""
    rows = []
    for h in horizons:
        ds = build_pacr_funding_attribution(panel, h)
        M0 = ds["mark_price_origin"].to_numpy()
        for direction in ("long", "short"):
            nm_with = ds[f"net_max_{direction}_with"].to_numpy()
            nm_without = ds[f"net_max_{direction}_without"].to_numpy()
            for lev in leverage_grid:
                w = pacr_ratio(nm_with, M0, lev)
                wo = pacr_ratio(nm_without, M0, lev)
                valid = ~(np.isnan(w) | np.isnan(wo))
                w, wo = w[valid], wo[valid]
                delta = w - wo
                no_deficit_either = (wo == 0) & (w == 0)
                creates = (wo == 0) & (w > 0)
                removes = (wo > 0) & (w == 0)
                increases = (wo > 0) & (w > wo)
                decreases = (wo > 0) & (w < wo) & (w > 0)
                unchanged_positive = (wo > 0) & (w == wo)
                either_positive = (wo > 0) | (w > 0)
                rows.append({
                    "horizon": h, "direction": direction, "leverage": lev,
                    "n": int(valid.sum()),
                    "mean_pacr_with": float(w.mean()), "mean_pacr_without": float(wo.mean()),
                    "mean_delta": float(delta.mean()),
                    "mean_delta_given_any_deficit": float(delta[either_positive].mean()) if either_positive.any() else np.nan,
                    "share_no_deficit_either_way": float(no_deficit_either.mean()),
                    "share_creates_deficit": float(creates.mean()),
                    "share_removes_deficit": float(removes.mean()),
                    "share_increases_deficit": float(increases.mean()),
                    "share_decreases_deficit": float(decreases.mean()),
                    "share_unchanged_positive": float(unchanged_positive.mean()),
                })
    return pd.DataFrame(rows)


# --- H3 / H4: quantile regressions of PACRRatio (long side, primary leverage) --

PRIMARY_LEVERAGE = 10
QUANTILE_TAUS = (0.50, 0.90, 0.95)
MARKET_CONDITION_PREDICTORS = ["rv_primary", "basis", "d_oi", "spread", "ofi", "volume_shock"]


def build_h3_h4_dataset(panel: pd.DataFrame, h: int, direction: str = "long",
                         leverage: int = PRIMARY_LEVERAGE) -> pd.DataFrame:
    ds = build_pacr_dataset(panel, h)
    d = DIRECTIONS[direction]
    c_signed = panel.loc[ds.index, f"C_signed_H{h}"] if f"C_signed_H{h}" in panel else (
        panel.loc[ds.index, f"C_long_H{h}"] - panel.loc[ds.index, f"C_short_H{h}"])
    r_h = panel.loc[ds.index, f"R_H{h}"]
    df = pd.DataFrame(index=ds.index)
    df["PACRRatio"] = pacr_ratio(ds[f"net_max_{direction}"].to_numpy(), ds["mark_price_origin"].to_numpy(), leverage)
    df["paying_side"] = ((d * c_signed) > 0).astype(float)
    df["adverse_move"] = ((-d * r_h) > 0).astype(float)
    df["paying_x_adverse"] = df["paying_side"] * df["adverse_move"]
    for c in MARKET_CONDITION_PREDICTORS:
        df[c] = panel.loc[ds.index, c]
    return df.dropna().sort_index()


def _hurdle_fit(df: pd.DataFrame, predictor_cols: list[str], taus, standardise: bool) -> tuple[dict, pd.DataFrame]:
    """Two-part / hurdle model for a corner-solution response (PACRRatio has a
    genuine point mass at 0, not a censoring artifact -- there really is no
    shortfall on most origins). Plain quantile regression is degenerate at any
    tau below the zero-mass share (fitted slopes ~0, sparsity/standard-error
    estimates blow up, hence the exact 0.0/1.0 p-values in the first pass).

    Part 1 (participation): logit for Pr(PACRRatio > 0 | X).
    Part 2 (severity): quantile regression of PACRRatio on X, conditional on
    PACRRatio > 0, at the same pre-specified taus -- interpreted as "the
    tau-quantile of shortfall size, given a shortfall occurs" (Duan, Manning,
    Morris & Newhouse, 1983, the standard two-part-model treatment of
    corner-solution/zero-inflated outcomes; here applied part-2 with a
    quantile, rather than mean, regression).
    """
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg

    X = df[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = sm.add_constant(X)
    y = df["PACRRatio"]
    participate = (y > 0).astype(float)

    logit = sm.Logit(participate, X).fit(disp=0)
    participation = {"n": int(logit.nobs), "share_positive": float(participate.mean()),
                      "pseudo_r2": float(logit.prsquared)}
    for c in predictor_cols:
        participation[c] = float(logit.params[c])
        participation[f"{c}_p"] = float(logit.pvalues[c])

    pos = df[y > 0]
    Xp = pos[predictor_cols].copy()
    if standardise:
        for c in predictor_cols:
            Xp[c] = (Xp[c] - Xp[c].mean()) / Xp[c].std()
    Xp = sm.add_constant(Xp)
    yp = pos["PACRRatio"]

    rows = []
    for tau in taus:
        res = QuantReg(yp, Xp).fit(q=tau, max_iter=5000)
        row = {"tau": tau, "n": int(res.nobs), "const": float(res.params["const"])}
        for c in predictor_cols:
            row[c] = float(res.params[c])
            row[f"{c}_p"] = float(res.pvalues[c])
        rows.append(row)
    return participation, pd.DataFrame(rows)


def fit_h3_interaction(df: pd.DataFrame, taus=QUANTILE_TAUS) -> tuple[dict, pd.DataFrame]:
    """Returns (participation_logit_dict, severity_quantile_df). See
    _hurdle_fit's docstring for why this replaced plain QuantReg."""
    return _hurdle_fit(df, ["paying_side", "adverse_move", "paying_x_adverse"], taus, standardise=False)


def fit_h4_market_conditions(df: pd.DataFrame, taus=QUANTILE_TAUS) -> tuple[dict, pd.DataFrame]:
    """Returns (participation_logit_dict, severity_quantile_df); market-
    condition predictors are standardised as before."""
    return _hurdle_fit(df, MARKET_CONDITION_PREDICTORS, taus, standardise=True)
