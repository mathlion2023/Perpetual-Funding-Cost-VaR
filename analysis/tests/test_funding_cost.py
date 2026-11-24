"""Regression tests for the forward-looking target builders in funding_cost.py.

Run with either `python -m pytest analysis/tests` or
`python -m analysis.tests.test_funding_cost` (no pytest needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import funding_cost


def _hourly(values) -> pd.Series:
    # forward_sum/forward_value now refuse anything but a complete hourly
    # DatetimeIndex (lag audit, 2026-11-06), so test series carry one too.
    values = np.asarray(values, dtype=float)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC"))


def test_forward_sum_excludes_origin_hour():
    # Regression test for the 2026-09-28 off-by-one: out[t] must be
    # x[t+1] + ... + x[t+h], never including x[t] itself.
    s = _hourly([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = funding_cost.forward_sum(s, 2)
    assert out.tolist()[:4] == [5.0, 7.0, 9.0, 11.0]
    assert out.iloc[4:].isna().all()  # windows running past the end


def test_forward_sum_nan_only_inside_window():
    # A NaN at the origin hour must NOT void the window (it is not in it);
    # a NaN inside the window must.
    s = _hourly([np.nan, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    out = funding_cost.forward_sum(s, 2)
    assert out.iloc[0] == 2.0          # window x[1], x[2]
    assert np.isnan(out.iloc[1])       # window x[2], x[3] contains NaN
    assert np.isnan(out.iloc[2])       # window x[3], x[4] contains NaN
    assert out.iloc[3] == 2.0          # window x[4], x[5]


def test_forward_sum_matches_rolling_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    x[rng.choice(500, 25, replace=False)] = np.nan
    s = _hourly(x)
    for h in (1, 3, 72):
        ref = s.shift(-1).rolling(h).sum().shift(-(h - 1))  # NaN if any value in x[t+1..t+h] is NaN
        got = funding_cost.forward_sum(s, h)
        both = ref.notna() | got.notna()
        assert (ref.notna() == got.notna()).all()
        assert np.allclose(got[both], ref[both])


def test_price_and_funding_windows_align():
    # L_net_H{h} = -R_H{h} + C_signed_H{h}: the price move (mark at t -> t+h)
    # and the funding sum must cover the same hours t+1..t+h. With
    # f_eff[k] = k and mark[k] = 100 + k, both are pinned down exactly.
    s = _hourly(np.arange(10, dtype=float))
    mark = _hourly(100.0 + np.arange(10))
    h = 3
    assert funding_cost.forward_sum(s, h).iloc[0] == 1.0 + 2.0 + 3.0   # hours 1..3, not 0..3
    assert funding_cost.forward_value(mark, h).iloc[0] == 103.0        # mark at hour 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
