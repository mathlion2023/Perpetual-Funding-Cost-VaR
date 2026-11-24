"""Section 3: construct the hourly information panel.

One row per hour t; every field in row t is built only from observations
timestamped in [t, t+1h), so nothing in row t uses information from after
the end of hour t (no look-ahead). Downstream horizon variables (Sections
4-6) instead look *forward* from t and must not be confused with this panel.

Deribit-specific convention baked in here (see data_validation.py for the
sourcing): the raw `funding_rate` field is an 8-hour rate that accrues
continuously, so the hourly effective funding rate is the intra-hour mean of
that field divided by 8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _hour_floor(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("h")


def _complete_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto every hour between the first and last observed hour.

    Lag-audit fix (2026-11-06): any shift()/rolling() on a frame built by
    groupby('hour') counts *rows*, so after a missing hour the "previous
    hour" silently became the last observed hour (e.g. d_oi across a
    multi-day gap). Every lag/rolling feature in this module is now built
    only after this reindex, so shift(1) is exactly one elapsed hour and a
    rolling(n) window spans exactly n elapsed hours.
    """
    if df.empty:
        return df
    grid = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    return df.reindex(grid)


def bps_spread(ask: pd.Series, bid: pd.Series) -> pd.Series:
    """Quoted spread in bps: 10,000 * (ask - bid) / mid. Changed 2026-09-26 from
    a raw ratio (~1e-5), which was ~2000x smaller than spread_binc and not
    comparable across the two regressors -- any output fit against the old
    ratio scale is stale. Shared by build_ticker_hourly, build_binance_spread_hourly,
    and construct_spread_bps.py so this scale convention can't drift out of
    sync between them again."""
    mid = (ask + bid) / 2.0
    return 10_000 * (ask - bid) / mid


def _end_of_hour_snapshot(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Last observation within each hour, for level variables (mark/index/OI)."""
    g = df.assign(hour=_hour_floor(df["ts"])).groupby("hour", sort=True)
    return g[cols].last()


def build_ticker_hourly(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Sections 3.1-3.3 and the raw funding-rate aggregation, from ticker snapshots."""
    df = funding_df.copy()
    df["hour"] = _hour_floor(df["ts"])

    df["spread"] = bps_spread(df["ask"], df["bid"])
    # NOTE: ask_amount/bid_amount are present in the Binance feed but are
    # empirically all-null in this Deribit feed across the full 2020-2025
    # sample (checked at multiple points in the series). depth_imbalance is
    # therefore computed but will be entirely NaN with the current data
    # source; it is not part of the Section 6 predictor vector X_t, so this
    # does not block any modeling stage, but re-sourcing L2 depth data is a
    # prerequisite if a future revision wants to use it.
    df["depth_imbalance"] = (df["bid_amount"] - df["ask_amount"]) / (df["bid_amount"] + df["ask_amount"])

    g = df.groupby("hour", sort=True)

    hourly = pd.DataFrame(index=g.size().index)
    hourly["mark_price"] = g["mark_price"].last()
    hourly["index_price"] = g["index_price"].last()
    hourly["open_interest"] = g["open_interest"].last()
    hourly["spread"] = g["spread"].median()
    hourly["depth_imbalance"] = g["depth_imbalance"].median()

    # CORRECTED 2026-10-20 (Tina, via Kaiko support): `funding_rate` is the
    # TRAILING 8-hour accrued rate (confirmed empirically -- it tracks a
    # rolling 8h mean of `predicted_funding_rate` at correlation 0.99+ across
    # every period checked), not a point-in-time current rate. `predicted_funding_rate`
    # is the live, continuously-updating current rate (despite its name, not a
    # forecast of future funding) -- dividing ITS intra-hour mean by 8 is what
    # actually gives the rate accrued during that specific hour. Every
    # downstream target (c_long/c_signed/c_short, forward_sum'd into
    # C_long_H/C_signed_H/L_net_H) is built from f_eff, so this correction
    # propagates through the whole pipeline automatically.
    #
    # `funding_rate`'s own intra-hour mean is retained below as `f_trail8h`,
    # a genuinely separate (slower, smoothed) feature -- it is what the old,
    # incorrect `f_eff` used to be, and under the corrected understanding it
    # is a legitimate trailing-average predictor in its own right, not ground
    # truth. It replaces the old `f_pred` feature in forecast_set.py's X_t:
    # `f_pred` was built from this same `predicted_funding_rate` field and so
    # would now be numerically identical to the corrected f_eff_t, which would
    # make it a redundant duplicate column rather than a second, genuinely
    # distinct funding-rate timescale in the predictor set.
    if "predicted_funding_rate" in df.columns:
        hourly["predicted_funding_rate_8h_mean"] = g["predicted_funding_rate"].mean()
        hourly["f_eff"] = hourly["predicted_funding_rate_8h_mean"] / 8.0
    else:
        hourly["f_eff"] = np.nan

    hourly["funding_rate_8h_mean"] = g["funding_rate"].mean()
    hourly["f_trail8h"] = hourly["funding_rate_8h_mean"] / 8.0

    hourly = _complete_hourly_grid(hourly)

    # Section 3.1: basis from end-of-hour mark/index.
    hourly["basis"] = (hourly["mark_price"] - hourly["index_price"]) / hourly["index_price"]

    # Section 3.3: log change in open interest.
    hourly["d_oi"] = np.log(hourly["open_interest"]) - np.log(hourly["open_interest"].shift(1))

    return hourly


def build_binance_spread_hourly(binc_df: pd.DataFrame) -> pd.DataFrame:
    """Auxiliary Spread_binc control (Tina, 2026-09-26 email): Binance
    BTCUSD_PERP (COIN-M, inverse/coin-margined perpetual -- confirmed against
    the raw file path/naming convention during the 2026-09-26 peer review;
    NOT the linear USDT-margined BTCUSDT contract, despite the "BTC-USDT"
    label this project used elsewhere before that correction) quoted spread
    in bps, aggregated the same way as the Deribit `spread` column (median
    per hour). Kept as a separate column, not a replacement -- used here
    purely as a second, independent liquidity proxy alongside the Deribit
    spread, never as a funding-cost source (see config.py's scope note).
    Coin-margined BTCUSD_PERP is, if anything, a closer structural analog to
    Deribit's own inverse BTC-PERPETUAL than the linear contract would have
    been.

    `binc_df` has only `ts`, `bid`, `ask` (data_loading.load_binance_quotes_range);
    an empty frame (no Binance coverage in range) returns an empty hourly frame,
    which reindexes to all-NaN spread_binc downstream rather than raising.
    """
    return _cross_venue_spread_hourly(binc_df, "spread_binc")


def _cross_venue_spread_hourly(df: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """Hourly median quoted spread (bps) from any ts/bid/ask ticker frame.

    Shared by build_binance_spread_hourly (Deribit panel's spread_binc
    control, from Binance quotes) and build_binance_ticker_hourly (Binance
    panel's spread_binc slot, from *Deribit* quotes -- see that function's
    docstring for why the column keeps the same name in both directions)."""
    if df.empty:
        return pd.DataFrame(columns=[out_col])
    d = df.copy()
    d["hour"] = _hour_floor(d["ts"])
    d["spread_bps"] = bps_spread(d["ask"], d["bid"])
    return d.groupby("hour", sort=True)["spread_bps"].median().rename(out_col).to_frame()


def build_binance_ticker_hourly(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Binance analogue of build_ticker_hourly -- NOT a drop-in reuse, because
    Binance's funding settles discretely three times a day rather than
    continuously (see config.py's note; confirmed empirically 2026-11-06 via
    data_validation.validate_binance_funding_conventions). `predicted_funding_rate`
    is entirely null for Binance in this data, so there is no live/trailing
    split the way Deribit has one; instead:

      - `funding_rate_hour_value` is simply the (constant-within-hour) raw
        8h-rate reading for that hour, from end-of-hour .last() like every
        other level variable here.
      - `f_trail8h` = funding_rate_hour_value / 8, held across all hours of
        each 8h settlement block -- a smooth, always-observable predictor,
        playing the same X_t role as Deribit's f_trail8h (its own slower,
        smoothed funding signal).
      - `f_eff` is a settlement *hurdle*, not an hourly accrual: zero at every
        hour, except at hours whose UTC hour-of-day is in
        config.BINANCE_SETTLEMENT_HOURS_UTC, where it equals the just-settled
        funding_rate_hour_value (full 8h rate, no /8). This makes
        funding_cost.py's existing forward_sum(c_long/c_signed, h) machinery
        correct unchanged: summing f_eff.clip(lower=0) over (t, t+H] picks up
        exactly the settlement payments that fall in that window, each
        counted once in full -- not Deribit's smooth hourly trickle. A missing
        settlement-hour reading stays NaN (not silently zeroed), consistent
        with every other missing-data convention in this module.
    """
    df = funding_df.copy()
    df["hour"] = _hour_floor(df["ts"])
    df["spread"] = bps_spread(df["ask"], df["bid"])

    g = df.groupby("hour", sort=True)

    hourly = pd.DataFrame(index=g.size().index)
    hourly["mark_price"] = g["mark_price"].last()
    hourly["index_price"] = g["index_price"].last()
    hourly["open_interest"] = g["open_interest"].last()
    # Named "spread" (not "spread_binance_own") so forecast_set.PREDICTOR_COLUMNS
    # needs no Binance-specific variant: in the Binance panel this is Binance's
    # own quoted spread, the same role Deribit's "spread" plays in its panel.
    hourly["spread"] = g["spread"].median()
    hourly["funding_rate_hour_value"] = g["funding_rate"].last()

    hourly = _complete_hourly_grid(hourly)

    is_settlement_hour = pd.Series(hourly.index.hour, index=hourly.index).isin(config.BINANCE_SETTLEMENT_HOURS_UTC)
    hourly["f_eff"] = hourly["funding_rate_hour_value"].where(is_settlement_hour, 0.0)
    hourly["f_trail8h"] = hourly["funding_rate_hour_value"] / 8.0

    hourly["basis"] = (hourly["mark_price"] - hourly["index_price"]) / hourly["index_price"]
    hourly["d_oi"] = np.log(hourly["open_interest"]) - np.log(hourly["open_interest"].shift(1))

    return hourly


def build_binance_hourly_panel(
    binance_funding_df: pd.DataFrame, binance_trades_df: pd.DataFrame, deribit_funding_df: pd.DataFrame
) -> pd.DataFrame:
    """Binance analogue of build_hourly_panel: Binance ticker + Binance trade
    flow, joined with Deribit's own quoted spread filling the `spread_binc`
    slot (see build_binance_ticker_hourly's and forecast_set.PREDICTOR_COLUMNS'
    cross-venue-control design -- the column keeps the name spread_binc so
    forecast_set.build_forecast_set needs no Binance-specific variant, even
    though here it holds *Deribit's* spread, not Binance's own).

    `deribit_funding_df` only needs ts/bid/ask (data_loading.load_funding_range's
    output has these); nothing else is read from it here.
    """
    ticker_hourly = build_binance_ticker_hourly(binance_funding_df)
    flow_hourly = build_trade_flow_hourly(binance_trades_df)
    panel = _join_ticker_and_flow(ticker_hourly, flow_hourly)
    deribit_spread_hourly = _cross_venue_spread_hourly(deribit_funding_df, "spread_binc")
    return panel.join(deribit_spread_hourly.reindex(panel.index))


def build_binance_hourly_panel_streaming(
    binance_funding_df: pd.DataFrame, binance_trades_files: list, deribit_funding_df: pd.DataFrame
) -> pd.DataFrame:
    """Memory-safe variant of build_binance_hourly_panel for multi-year ranges
    (see build_hourly_panel_streaming)."""
    ticker_hourly = build_binance_ticker_hourly(binance_funding_df)
    flow_hourly = build_trade_flow_hourly_streaming(binance_trades_files)
    panel = _join_ticker_and_flow(ticker_hourly, flow_hourly)
    deribit_spread_hourly = _cross_venue_spread_hourly(deribit_funding_df, "spread_binc")
    return panel.join(deribit_spread_hourly.reindex(panel.index))


def _realised_vol_1bar(prices: pd.Series, ts: pd.Series, bar: str) -> pd.DataFrame:
    """Resample trade prices to `bar` last-price bars, return per-hour RV^2 and RV."""
    px = pd.Series(prices.values, index=ts.values).sort_index()
    # Forward-fill no further than one hour (lag-audit fix, 2026-11-06): an
    # unlimited ffill bridged multi-hour trade gaps with flat bars and then
    # booked the whole gap's price move as one bar return in the first hour
    # after the gap, inflating that hour's RV. Minutes without a trade inside
    # an active hour are still carried forward as before.
    bars_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(bar))
    bars = px.resample(bar).last().ffill(limit=bars_per_hour)
    logret = np.log(bars).diff()
    hour = pd.Series(bars.index).dt.floor("h").values
    rv2 = pd.Series(logret.values**2, index=bars.index).groupby(hour).sum()
    rv2.index = pd.to_datetime(rv2.index, utc=True)
    return pd.DataFrame({"rv2": rv2, "rv": np.sqrt(rv2)})


def _trade_flow_hourly_core(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk aggregation: total/signed volume, OFI, realised variance -- no
    cross-chunk rolling windows here (those need the full, concatenated
    hourly series; see _apply_rolling_flow_features)."""
    df = df.copy()
    df["hour"] = _hour_floor(df["ts"])
    df["signed_amount"] = df["sign"] * df["amount"]
    g = df.groupby("hour", sort=True)

    hourly = pd.DataFrame(index=g.size().index)
    hourly["total_volume"] = g["amount"].sum()
    hourly["signed_volume"] = g["signed_amount"].sum()
    hourly["ofi"] = hourly["signed_volume"] / hourly["total_volume"]

    rv_1min = _realised_vol_1bar(df["price"], df["ts"], "1min")
    rv_5min = _realised_vol_1bar(df["price"], df["ts"], "5min")
    hourly["rv2_1h"] = rv_1min["rv2"]
    hourly["rv_1h"] = rv_1min["rv"]
    hourly["rv2_1h_5min"] = rv_5min["rv2"]
    hourly["rv_1h_5min"] = rv_5min["rv"]
    return hourly


def _apply_rolling_flow_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Cross-chunk rolling RV windows and volume shock -- must run on the full,
    time-ordered hourly series (not per-chunk), or windows reset at chunk
    boundaries. Reindexed to the complete hourly grid first, so windows and
    the volume-shock lag are measured in elapsed hours, not observed rows."""
    hourly = _complete_hourly_grid(hourly.sort_index())
    for label, n in config.RV_WINDOWS_HOURS.items():
        hourly[f"rv_{label}"] = np.sqrt(hourly["rv2_1h"].rolling(n, min_periods=n).sum())
        hourly[f"rv_{label}_5min"] = np.sqrt(hourly["rv2_1h_5min"].rolling(n, min_periods=n).sum())

    log_tv = np.log(hourly["total_volume"].replace(0, np.nan))
    lookback = config.VOLUME_SHOCK_LOOKBACK_HOURS
    hourly["volume_shock"] = log_tv - log_tv.shift(1).rolling(lookback, min_periods=lookback).mean()
    return hourly


def build_trade_flow_hourly(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Section 3.4: total/signed volume, order-flow imbalance, realised volatility, volume shock.

    Loads the whole `trades_df` into memory at once -- fine for short date
    ranges, but see build_trade_flow_hourly_streaming for multi-year ranges
    where the raw tick data does not fit in memory.
    """
    hourly = _trade_flow_hourly_core(trades_df)
    return _apply_rolling_flow_features(hourly)


def build_trade_flow_hourly_streaming(files: list) -> pd.DataFrame:
    """Same output as build_trade_flow_hourly, but processes trade files one
    calendar month at a time and discards each month's raw ticks before
    moving to the next -- avoids holding a multi-year, multi-hundred-million-row
    trades DataFrame in memory at once (this is what caused an
    ArrowMemoryError when the pipeline first attempted a full 2020-2025 run).

    Each month's first 1-minute and 5-minute realised-vol bar has a NaN
    return (no prior bar to diff against within that chunk) instead of
    linking to the previous month's last price -- a deliberate, negligible
    approximation (roughly 1 lost return per ~43,800 one-minute bars per
    month) traded off against not re-introducing a large in-memory carry
    structure.
    """
    from collections import defaultdict

    from . import data_loading

    by_month: dict[str, list] = defaultdict(list)
    for f in files:
        by_month[f.parent.name].append(f)

    monthly_hourly = []
    for month in sorted(by_month):
        month_files = sorted(by_month[month])
        frames = [pd.read_csv(f, usecols=data_loading.TRADES_COLUMNS) for f in month_files]
        df = pd.concat(frames, ignore_index=True)
        del frames

        df["ts"] = pd.to_datetime(df["date"], unit="ms", utc=True)
        sell = df["sell"]
        if sell.dtype == object:
            sell = sell.astype(str).str.strip().str.lower().map({"true": True, "false": False})
        df["sign"] = sell.astype(bool).map({False: 1, True: -1}).astype("int8")
        # (ts, id) stable sort -- see the identical fix and rationale in
        # data_loading.load_trades_range; must match exactly so streaming and
        # in-memory panels agree on which trade .resample().last() picks.
        df = df.sort_values(["ts", "id"], kind="stable")

        monthly_hourly.append(_trade_flow_hourly_core(df))
        del df

    flow_hourly = pd.concat(monthly_hourly)
    return _apply_rolling_flow_features(flow_hourly)


def _join_ticker_and_flow(ticker_hourly: pd.DataFrame, flow_hourly: pd.DataFrame) -> pd.DataFrame:
    start = min(ticker_hourly.index.min(), flow_hourly.index.min())
    end = max(ticker_hourly.index.max(), flow_hourly.index.max())
    full_index = pd.date_range(start, end, freq="h", tz="UTC")

    panel = ticker_hourly.reindex(full_index).join(flow_hourly.reindex(full_index))
    panel.index.name = "hour"
    return panel


def _join_binance_spread(panel: pd.DataFrame, binance_df: pd.DataFrame | None) -> pd.DataFrame:
    """Always leaves a `spread_binc` column on the panel -- all-NaN if
    `binance_df` is None/empty -- so forecast_set.PREDICTOR_COLUMNS' presence
    check does not depend on every caller remembering to pass binance_df."""
    if binance_df is None or binance_df.empty:
        panel = panel.copy()
        panel["spread_binc"] = np.nan
        return panel
    binance_hourly = build_binance_spread_hourly(binance_df)
    return panel.join(binance_hourly.reindex(panel.index))


def build_hourly_panel(funding_df: pd.DataFrame, trades_df: pd.DataFrame, binance_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join ticker- and trade-derived hourly features into the Section 3 information panel.

    Loads all of `trades_df` into memory at once -- fine for date ranges up
    to a couple of years; for multi-year ranges use
    build_hourly_panel_streaming, which never holds the full raw tick
    history in memory simultaneously.

    The output is reindexed onto a complete hourly grid spanning the union of
    both inputs' ranges; hours with no ticker or no trade activity are left
    as NaN rather than silently dropped or filled, so gaps are visible to
    downstream code.

    `binance_df` (optional, from data_loading.load_binance_quotes_range) adds
    the auxiliary `spread_binc` column alongside the Deribit `spread` column;
    omitted entirely (default None) leaves the panel exactly as before this
    control existed.
    """
    ticker_hourly = build_ticker_hourly(funding_df)
    flow_hourly = build_trade_flow_hourly(trades_df)
    panel = _join_ticker_and_flow(ticker_hourly, flow_hourly)
    return _join_binance_spread(panel, binance_df)


def build_hourly_panel_streaming(funding_df: pd.DataFrame, trades_files: list, binance_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Memory-safe variant of build_hourly_panel for multi-year ranges: trade
    files are aggregated one month at a time (see
    build_trade_flow_hourly_streaming) rather than concatenated into one
    giant DataFrame first.

    `binance_df`: see build_hourly_panel.
    """
    ticker_hourly = build_ticker_hourly(funding_df)
    flow_hourly = build_trade_flow_hourly_streaming(trades_files)
    panel = _join_ticker_and_flow(ticker_hourly, flow_hourly)
    return _join_binance_spread(panel, binance_df)
