"""Section 2 validation: Deribit funding-convention checks.

The methodology requires four checks against Deribit's *official* documentation
before any funding-cost outcome is constructed: funding frequency, funding
units, funding sign, and notional base. This module hard-codes the answers as
found in Deribit's own materials (cited below) and adds empirical sanity
checks against the actual ticker data, so a future re-run flags it if the
data ever stops looking consistent with these documented conventions.

Sources (fetched 2026-09-23):
  - Deribit Insights, "Perpetual Swap Funding":
    https://insights.deribit.com/education/perpetual-swap-funding/
  - Deribit Support, "Linear Perpetual" / "Inverse Perpetual" articles
    (support.deribit.com/hc/en-us/articles/31424969384605,
     support.deribit.com/hc/en-us/articles/31424954847133)

Key documented facts (BTC perpetual):
  1. FREQUENCY: funding is calculated and transferred continuously ("every
     few seconds" per Deribit); it is not a discrete 8-hourly settlement
     event the way e.g. Binance's classic perpetuals are quoted. The rate is
     nonetheless *quoted* as an 8-hour interest rate for comparability.
  2. UNITS: the reported rate is an 8-hour rate. The actual funding payment
     over an elapsed interval is
         FundingPayment = FundingRate_8h * PositionSize_BTC * (ElapsedHours / 8).
     So the *hourly* effective rate is FundingRate_8h / 8, not the raw field
     value -- this is the conversion Section 2's "funding units" check exists
     to catch.
  3. SIGN: positive funding rate => longs pay shorts. Matches the
     methodology's c_t = max(f_eff_t, 0) long-cost convention directly.
  4. NOTIONAL BASE: position size is BTC-denominated (Deribit's BTC
     perpetual is an inverse contract), but because
     FundingPayment_USD = FundingPayment_BTC * MarkPrice
                          = FundingRate_8h * (Notional_USD / MarkPrice) * MarkPrice * (h/8)
                          = FundingRate_8h * Notional_USD * (h/8),
     the fractional rate applies identically whether notional is expressed
     in BTC or USD -- no additional notional-base adjustment is needed beyond
     the /8 unit conversion in (2).
  5. CAP/CLAMP: premium rate is dampened to zero within +/-0.025% of the
     index, then the funding rate is capped at +/-0.5% (8h rate) for BTC.

These are load-bearing assumptions for funding_cost.py. If Deribit changes
its funding mechanism, or if these documented facts are ever contradicted by
the empirical checks below, funding_cost.py must not be trusted until
re-validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

FUNDING_RATE_8H_CAP_BTC = 0.005  # +/-0.5%, 8-hour rate
DAMPENER_BAND = 0.00025  # +/-0.025%


@dataclass
class ValidationReport:
    n_obs: int
    frac_nonzero_change_within_hour: float
    max_abs_funding_rate: float
    frac_within_documented_cap: float
    frac_negative: float
    frac_positive: float
    frac_zero: float
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        lines = [
            "Section 2 funding-convention validation",
            "----------------------------------------",
            f"  observations checked            : {self.n_obs:,}",
            f"  fraction changing within the hour: {self.frac_nonzero_change_within_hour:.3f}"
            "  (near 1.0 => confirms continuous, not discrete-settlement, computation)",
            f"  max |funding_rate| observed      : {self.max_abs_funding_rate:.6f}",
            f"  fraction within documented +/-{FUNDING_RATE_8H_CAP_BTC:.3%} cap: {self.frac_within_documented_cap:.4f}",
            f"  sign split (neg / zero / pos)    : {self.frac_negative:.3f} / {self.frac_zero:.3f} / {self.frac_positive:.3f}",
        ]
        if self.warnings:
            lines.append("  WARNINGS:")
            lines.extend(f"    - {w}" for w in self.warnings)
        else:
            lines.append("  No inconsistencies with the documented convention were found.")
        return "\n".join(lines)


def validate_funding_conventions(funding_df: pd.DataFrame) -> ValidationReport:
    """Empirically check the raw funding_rate field against Deribit's documented convention.

    Parameters
    ----------
    funding_df : DataFrame with at least columns ['ts', 'funding_rate'].

    This does not re-derive the documented facts from first principles (that
    requires Deribit's live API, out of scope for an offline data pipeline);
    it checks that the *data on disk* is consistent with them, so a silent
    change in feed format or exchange convention gets caught rather than
    quietly propagated into the funding-cost outcome.
    """
    df = funding_df.dropna(subset=["funding_rate"]).copy()
    n_obs = len(df)
    warnings: list[str] = []

    if n_obs == 0:
        warnings.append("No non-null funding_rate observations in range.")
        return ValidationReport(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), warnings)

    df["hour"] = df["ts"].dt.floor("h")
    within_hour_changes = df.groupby("hour")["funding_rate"].nunique()
    multi_obs_hours = within_hour_changes[within_hour_changes.index.isin(df["hour"].value_counts()[df["hour"].value_counts() > 1].index)]
    frac_changing = float((multi_obs_hours > 1).mean()) if len(multi_obs_hours) else float("nan")

    max_abs = float(df["funding_rate"].abs().max())
    frac_within_cap = float((df["funding_rate"].abs() <= FUNDING_RATE_8H_CAP_BTC + 1e-9).mean())
    frac_negative = float((df["funding_rate"] < 0).mean())
    frac_positive = float((df["funding_rate"] > 0).mean())
    frac_zero = float((df["funding_rate"] == 0).mean())

    if not np.isnan(frac_changing) and frac_changing < 0.5:
        warnings.append(
            "Funding rate rarely changes within an hour "
            f"(only {frac_changing:.1%} of multi-observation hours show variation) -- "
            "this looks more like a discrete-settlement feed than Deribit's documented "
            "continuous computation. Re-check the feed / venue before trusting the "
            "hourly-accrual (/8) conversion in funding_cost.py."
        )
    if frac_within_cap < 0.999:
        warnings.append(
            f"{(1 - frac_within_cap):.4%} of observations exceed the documented "
            f"+/-{FUNDING_RATE_8H_CAP_BTC:.3%} 8-hour cap -- either the cap has changed, "
            "the field is not the 8-hour rate assumed here, or there is a data-quality issue."
        )

    return ValidationReport(
        n_obs=n_obs,
        frac_nonzero_change_within_hour=frac_changing,
        max_abs_funding_rate=max_abs,
        frac_within_documented_cap=frac_within_cap,
        frac_negative=frac_negative,
        frac_positive=frac_positive,
        frac_zero=frac_zero,
        warnings=warnings,
    )


# --- Binance (second venue, 2026-11-06) -----------------------------------
#
# Binance's documented funding rule (binance.com/en/support/faq, "Introduction
# to Binance Futures Funding Rates"): F = avg(PremiumIndex) + clamp(InterestRate
# - PremiumIndex, -0.05%, +0.05%), settled every 8 hours by default at 00:00,
# 08:00, 16:00 UTC (can shorten to 4h/1h in extreme volatility, not observed in
# this sample -- see below). This module does not re-derive F from the premium
# index; it checks, the same way validate_funding_conventions does for Deribit,
# that the recorded `funding_rate` field on disk behaves the way that
# documented settlement rule implies: flat between settlements, and changing
# only at settlement hours.
BINANCE_FUNDING_RATE_CAP = 0.0075  # +/-0.75%, loosest documented per-settlement cap (0.75 x MMR tier); see module note


@dataclass
class BinanceValidationReport:
    n_obs: int
    n_settlement_events: int
    frac_settlement_events_at_documented_hours: float
    median_gap_hours_between_settlements: float
    frac_nonzero_change_within_hour: float
    max_abs_funding_rate: float
    frac_within_documented_cap: float
    frac_negative: float
    frac_positive: float
    frac_zero: float
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        lines = [
            "Section 2 funding-convention validation (Binance, discrete settlement)",
            "------------------------------------------------------------------------",
            f"  observations checked                  : {self.n_obs:,}",
            f"  distinct settlement events             : {self.n_settlement_events:,}",
            f"  fraction of settlements at hour in {config.BINANCE_SETTLEMENT_HOURS_UTC} UTC: "
            f"{self.frac_settlement_events_at_documented_hours:.3f}  (near 1.0 => confirms the documented 8h schedule)",
            f"  median gap between settlements (hours) : {self.median_gap_hours_between_settlements:.2f}  (expect ~8.0)",
            f"  fraction changing within the hour      : {self.frac_nonzero_change_within_hour:.3f}"
            "  (near 0.0 => confirms discrete, not continuous, computation -- opposite expectation from Deribit)",
            f"  max |funding_rate| observed            : {self.max_abs_funding_rate:.6f}",
            f"  fraction within +/-{BINANCE_FUNDING_RATE_CAP:.2%} cap        : {self.frac_within_documented_cap:.4f}",
            f"  sign split (neg / zero / pos)          : {self.frac_negative:.3f} / {self.frac_zero:.3f} / {self.frac_positive:.3f}",
        ]
        if self.warnings:
            lines.append("  WARNINGS:")
            lines.extend(f"    - {w}" for w in self.warnings)
        else:
            lines.append("  No inconsistencies with the documented Binance convention were found.")
        return "\n".join(lines)


def validate_binance_funding_conventions(funding_df: pd.DataFrame) -> BinanceValidationReport:
    """Empirically check Binance's raw funding_rate field against its documented
    discrete 8-hourly settlement convention (see module note above).

    Unlike Deribit, `predicted_funding_rate` is entirely null for Binance in
    this data and the realized `funding_rate` field is flat between
    settlements -- so the two checks that matter here are (a) settlements land
    on the documented hours, and (b) the value genuinely does not move between
    settlements (confirming this is not actually a continuous feed that merely
    looks sparse). This is the input panel.build_binance_ticker_hourly's
    discrete f_eff construction depends on being correct.
    """
    df = funding_df.dropna(subset=["funding_rate"]).sort_values("ts").copy()
    n_obs = len(df)
    warnings_: list[str] = []

    if n_obs == 0:
        warnings_.append("No non-null funding_rate observations in range.")
        return BinanceValidationReport(0, 0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), warnings_)

    # Settlement events: rows where the value differs from the immediately
    # preceding row (first row excluded -- no prior value to compare against).
    changed = df["funding_rate"].ne(df["funding_rate"].shift(1))
    changed.iloc[0] = False
    events = df.loc[changed]
    n_events = len(events)
    frac_at_documented_hours = float(events["ts"].dt.hour.isin(config.BINANCE_SETTLEMENT_HOURS_UTC).mean()) if n_events else float("nan")

    event_gaps_hours = events["ts"].diff().dropna().dt.total_seconds() / 3600.0
    median_gap = float(event_gaps_hours.median()) if len(event_gaps_hours) else float("nan")

    df["hour"] = df["ts"].dt.floor("h")
    within_hour_changes = df.groupby("hour")["funding_rate"].nunique()
    multi_obs_hours = within_hour_changes[within_hour_changes.index.isin(df["hour"].value_counts()[df["hour"].value_counts() > 1].index)]
    frac_changing = float((multi_obs_hours > 1).mean()) if len(multi_obs_hours) else float("nan")

    max_abs = float(df["funding_rate"].abs().max())
    frac_within_cap = float((df["funding_rate"].abs() <= BINANCE_FUNDING_RATE_CAP + 1e-9).mean())
    frac_negative = float((df["funding_rate"] < 0).mean())
    frac_positive = float((df["funding_rate"] > 0).mean())
    frac_zero = float((df["funding_rate"] == 0).mean())

    if not np.isnan(frac_at_documented_hours) and frac_at_documented_hours < 0.95:
        warnings_.append(
            f"Only {frac_at_documented_hours:.1%} of settlement events land on the documented "
            f"{config.BINANCE_SETTLEMENT_HOURS_UTC} UTC hours -- funding frequency may have shortened to "
            "4h/1h in this sample (Binance does this in extreme volatility) or the feed/venue has changed; "
            "re-check before trusting the discrete-settlement f_eff construction in panel.py."
        )
    if not np.isnan(frac_changing) and frac_changing > 0.05:
        warnings_.append(
            f"funding_rate changes within an hour in {frac_changing:.1%} of multi-observation hours -- "
            "this looks more like a continuous feed than Binance's documented discrete settlement. "
            "Re-check the feed / venue before trusting the discrete-settlement (/8-free) conversion."
        )
    if frac_within_cap < 0.999:
        warnings_.append(
            f"{(1 - frac_within_cap):.4%} of observations exceed the +/-{BINANCE_FUNDING_RATE_CAP:.2%} cap used here "
            "-- either a tighter/looser per-symbol cap applies, or there is a data-quality issue."
        )

    return BinanceValidationReport(
        n_obs=n_obs,
        n_settlement_events=n_events,
        frac_settlement_events_at_documented_hours=frac_at_documented_hours,
        median_gap_hours_between_settlements=median_gap,
        frac_nonzero_change_within_hour=frac_changing,
        max_abs_funding_rate=max_abs,
        frac_within_documented_cap=frac_within_cap,
        frac_negative=frac_negative,
        frac_positive=frac_positive,
        frac_zero=frac_zero,
        warnings=warnings_,
    )
