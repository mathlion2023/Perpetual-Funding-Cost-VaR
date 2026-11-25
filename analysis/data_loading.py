"""Raw data ingestion for Deribit BTC perpetual ticker snapshots and trade ticks
(and, since 2026-11-06, the parallel Binance BTCUSD_PERP comparison -- see
config.py's note on its different funding-settlement convention).

Source files are Kaiko-format daily gzipped CSVs:

  Funding/ticker file columns:
    timestamp (unix seconds), 24h_volume, ask, bid, ask_amount, bid_amount,
    funding_rate, mark_price, open_interest, predicted_funding_rate, price,
    index_price

  Trades file columns:
    id, exchange, symbol, date (unix milliseconds), price, amount, sell (bool)

Both are minute-or-finer granularity snapshots/ticks, not settlement-level
funding records -- see Section 2 of the methodology for why every field here
must be validated against Deribit's official documentation before being
treated as a cash funding-cost outcome.
"""

from __future__ import annotations

import datetime as dt
import gzip
import warnings
from pathlib import Path

import pandas as pd

from . import config

FUNDING_COLUMNS = [
    "timestamp",
    "24h_volume",
    "ask",
    "bid",
    "ask_amount",
    "bid_amount",
    "funding_rate",
    "mark_price",
    "open_interest",
    "predicted_funding_rate",
    "price",
    "index_price",
]

TRADES_COLUMNS = ["id", "exchange", "symbol", "date", "price", "amount", "sell"]

BINANCE_QUOTE_COLUMNS = ["timestamp", "bid", "ask"]


def _daily_files(root: Path, prefix: str, start: dt.date, end: dt.date) -> list[Path]:
    """Return sorted daily files under root whose embedded date lies in [start, end]."""
    files = []
    for month_dir in sorted(root.glob("*")):
        if not month_dir.is_dir():
            continue
        for f in sorted(month_dir.glob(f"{prefix}_*.csv.gz")):
            date_str = f.name.rsplit("_", 1)[-1].replace(".csv.gz", "")
            try:
                file_date = dt.date.fromisoformat(date_str)
            except ValueError:
                continue
            if start <= file_date <= end:
                files.append(f)
    return files


def funding_files(start: dt.date, end: dt.date) -> list[Path]:
    return _daily_files(config.FUNDING_DIR, "derivatives_drbt_perpetual-future_btcperpetual", start, end)


def trades_files(start: dt.date, end: dt.date) -> list[Path]:
    return _daily_files(config.TRADES_DIR, "trades_drbt_perpetual-future_btcperpetual", start, end)


def binance_funding_files(start: dt.date, end: dt.date) -> list[Path]:
    return _daily_files(config.BINANCE_FUNDING_DIR, "derivatives_binc_perpetual-future_btcusd_perp", start, end)


def binance_trades_files(start: dt.date, end: dt.date) -> list[Path]:
    return _daily_files(config.BINANCE_TRADES_DIR, "trades_binc_perpetual-future_btcusd_perp", start, end)


def load_funding_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load raw ticker snapshots for [start, end] into one sorted DataFrame.

    Returns columns as in FUNDING_COLUMNS plus a `ts` UTC-timestamp column.
    No resampling or validation is applied here -- see panel.py and
    data_validation.py.
    """
    files = funding_files(start, end)
    if not files:
        raise FileNotFoundError(f"No funding files found for [{start}, {end}] under {config.FUNDING_DIR}")

    frames = []
    files_missing_columns: dict[str, list[str]] = {}
    for f in files:
        with gzip.open(f, "rt") as gf:
            header = gf.readline().strip().split(",")
        available = [c for c in FUNDING_COLUMNS if c in header]
        missing = [c for c in FUNDING_COLUMNS if c not in header]
        if missing:
            files_missing_columns[f.name] = missing
        frame = pd.read_csv(f, usecols=available)
        frame = frame.reindex(columns=FUNDING_COLUMNS)  # missing columns become all-NaN
        frames.append(frame)

    if files_missing_columns:
        n_bad = len(files_missing_columns)
        example = next(iter(files_missing_columns.items()))
        warnings.warn(
            f"{n_bad}/{len(files)} funding files in [{start}, {end}] are missing expected columns "
            f"(e.g. {example[0]} missing {example[1]}) -- likely a Kaiko export/schema gap, observed "
            f"empirically to cluster in 2021. Those hours will have NaN funding_rate and propagate as "
            f"NaN through funding_cost.py rather than being silently dropped or interpolated.",
            stacklevel=2,
        )

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # kind="stable" (code audit, 2026-11-25): pandas' default sort is not
    # stable, so a same-timestamp tie would otherwise break inconsistently
    # depending on how much surrounding data is being sorted -- same failure
    # mode documented and fixed for load_trades_range below. No separate
    # tiebreak column exists for this snapshot feed, so a stable sort (tie
    # order = original file/row concatenation order) is the deterministic fix.
    df = df.sort_values("ts", kind="stable").drop_duplicates(subset="ts").reset_index(drop=True)
    return df


def load_trades_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Load raw trade ticks for [start, end] into one sorted DataFrame.

    Adds:
      ts        -- UTC timestamp parsed from the millisecond `date` field
      sign      -- +1 for buyer-initiated (sell == False), -1 for seller-initiated
                   (sell == True), per the methodology's Section 3.4 convention.
                   This is the exchange-reported side, not a tick-test inference.
    """
    files = trades_files(start, end)
    if not files:
        raise FileNotFoundError(f"No trades files found for [{start}, {end}] under {config.TRADES_DIR}")
    frames = [pd.read_csv(f, usecols=TRADES_COLUMNS) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["date"], unit="ms", utc=True)

    sell = df["sell"]
    if sell.dtype == object:
        sell = sell.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    df["sell"] = sell.astype(bool)
    df["sign"] = df["sell"].map({False: 1, True: -1}).astype("int8")

    # Sort by (ts, id): many trades share the same millisecond timestamp at
    # this venue's volume, and pandas' default sort is not stable, so ties
    # break inconsistently depending on how much surrounding data is being
    # sorted (e.g. a full multi-year sort vs. a single month) -- this
    # silently changed which trade .resample().last() picked as "last" in a
    # given minute bar between the in-memory and chunked code paths. `id` is
    # the exchange's own monotonic trade id, a deterministic tiebreak that
    # does not depend on load order or chunk size.
    df = df.sort_values(["ts", "id"], kind="stable").reset_index(drop=True)
    return df


def load_binance_quotes_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Bid/ask only, for the auxiliary Spread_binc liquidity control (see
    panel.build_binance_spread_hourly) -- not a funding-cost source, so
    funding_rate/mark_price/basis/OI are never read from this feed.

    Binance's snapshot history starts later than Deribit's and this feed is
    not validated for full-range coverage the way load_funding_range's inputs
    are, so a missing/partial file set here warns and returns whatever is
    available (possibly empty) rather than raising -- hours with no Binance
    coverage simply get a NaN spread_binc downstream, same NaN-not-dropped
    convention as the rest of panel.py.
    """
    files = binance_funding_files(start, end)
    if not files:
        warnings.warn(
            f"No Binance funding files found for [{start}, {end}] under {config.BINANCE_FUNDING_DIR} -- "
            "spread_binc will be entirely NaN over this range.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["ts", "bid", "ask"])

    frames = []
    files_missing_columns: dict[str, list[str]] = {}
    for f in files:
        with gzip.open(f, "rt") as gf:
            header = gf.readline().strip().split(",")
        available = [c for c in BINANCE_QUOTE_COLUMNS if c in header]
        missing = [c for c in BINANCE_QUOTE_COLUMNS if c not in header]
        if missing:
            files_missing_columns[f.name] = missing
        frame = pd.read_csv(f, usecols=available)
        frame = frame.reindex(columns=BINANCE_QUOTE_COLUMNS)  # missing columns become all-NaN
        frames.append(frame)

    if files_missing_columns:
        n_bad = len(files_missing_columns)
        example = next(iter(files_missing_columns.items()))
        warnings.warn(
            f"{n_bad}/{len(files)} Binance quote files in [{start}, {end}] are missing expected columns "
            f"(e.g. {example[0]} missing {example[1]}) -- same Kaiko export/schema gap documented for "
            f"load_funding_range. Those rows will have NaN bid/ask and be dropped below, same as any "
            f"other hour with no Binance coverage.",
            stacklevel=2,
        )

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # kind="stable" (code audit, 2026-11-25): see load_funding_range's comment.
    df = df.dropna(subset=["bid", "ask"]).sort_values("ts", kind="stable").drop_duplicates(subset="ts").reset_index(drop=True)
    return df[["ts", "bid", "ask"]]


def load_binance_funding_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Full-column Binance ticker snapshots for [start, end] (2026-11-06,
    second-venue comparison) -- same Kaiko schema as load_funding_range
    (confirmed empirically), unlike load_binance_quotes_range which only
    reads bid/ask for the auxiliary spread_binc control. Used by
    panel.build_binance_ticker_hourly; see config.py's note on why Binance's
    `funding_rate` is a discretely-settled value (not Deribit's continuous
    one) and `predicted_funding_rate` is entirely null here.
    """
    files = binance_funding_files(start, end)
    if not files:
        raise FileNotFoundError(f"No Binance funding files found for [{start}, {end}] under {config.BINANCE_FUNDING_DIR}")

    frames = []
    files_missing_columns: dict[str, list[str]] = {}
    for f in files:
        with gzip.open(f, "rt") as gf:
            header = gf.readline().strip().split(",")
        available = [c for c in FUNDING_COLUMNS if c in header]
        missing = [c for c in FUNDING_COLUMNS if c not in header]
        if missing:
            files_missing_columns[f.name] = missing
        frame = pd.read_csv(f, usecols=available)
        frame = frame.reindex(columns=FUNDING_COLUMNS)
        frames.append(frame)

    if files_missing_columns:
        n_bad = len(files_missing_columns)
        example = next(iter(files_missing_columns.items()))
        warnings.warn(
            f"{n_bad}/{len(files)} Binance funding files in [{start}, {end}] are missing expected columns "
            f"(e.g. {example[0]} missing {example[1]}) -- same Kaiko export/schema gap documented for "
            f"load_funding_range.",
            stacklevel=2,
        )

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # kind="stable" (code audit, 2026-11-25): see load_funding_range's comment.
    df = df.sort_values("ts", kind="stable").drop_duplicates(subset="ts").reset_index(drop=True)
    return df


def load_binance_trades_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Binance trade ticks for [start, end] -- same schema and sign/stable-sort
    convention as load_trades_range (see that function's docstring); used by
    panel.build_binance_hourly_panel_streaming for the realised-vol/OFI/volume
    features, which are venue-agnostic.
    """
    files = binance_trades_files(start, end)
    if not files:
        raise FileNotFoundError(f"No Binance trades files found for [{start}, {end}] under {config.BINANCE_TRADES_DIR}")
    frames = [pd.read_csv(f, usecols=TRADES_COLUMNS) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["date"], unit="ms", utc=True)

    sell = df["sell"]
    if sell.dtype == object:
        sell = sell.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    df["sell"] = sell.astype(bool)
    df["sign"] = df["sell"].map({False: 1, True: -1}).astype("int8")
    df = df.sort_values(["ts", "id"], kind="stable").reset_index(drop=True)
    return df
