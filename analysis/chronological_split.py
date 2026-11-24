"""Chronological train/validation/test split utility.

Tina, 2026-10-07 email: Q4 needs volatility-regime thresholds "estimated
from earlier training data" and applied forward; Q5 needs policies
"calibrate[d] before evaluating the test sample" on "common origins." No
equivalent utility existed previously -- every other split in this codebase
is a per-origin walk-forward train_full/train_labeled distinction
(models/base.py), not a single held-out period held fixed across an entire
analysis. This is shared scaffolding for both.
"""
from __future__ import annotations

import pandas as pd


def chronological_split(index: pd.DatetimeIndex, train_frac: float = 0.6, val_frac: float = 0.2) -> dict:
    """Split index into train/val/test by elapsed TIME (not row count), so
    the boundaries are the same regardless of any later origin-subsampling
    (e.g. non-overlapping-origin spacing)."""
    t0, t1 = index.min(), index.max()
    total = t1 - t0
    train_end = t0 + total * train_frac
    val_end = t0 + total * (train_frac + val_frac)
    return {
        "train": index[index <= train_end],
        "val": index[(index > train_end) & (index <= val_end)],
        "test": index[index > val_end],
        "train_end": train_end,
        "val_end": val_end,
    }
