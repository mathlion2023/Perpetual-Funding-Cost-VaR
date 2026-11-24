"""Lag / missing-hour / alignment regression tests (lag audit, 2026-11-06).

Every test builds a small synthetic series with deliberately missing hours
(single hours and multi-hour gaps) and checks that a "previous hour" is the
previous elapsed hour, rolling windows span elapsed hours, forecasts start
from the origin hour, and no model sees a target that is not yet realised.

Run with either `python -m pytest analysis/tests` or
`python -m analysis.tests.test_time_alignment` (no pytest needed).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis import config, funding_cost, panel
from analysis.models import arx_garch, arx_mean, base, block_bootstrap, caviar, gas, ms_garch, rough_vol, stochastic_vol

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def _grid(n: int) -> pd.DatetimeIndex:
    return pd.date_range(T0, periods=n, freq="h")


def _ticks_for_hours(hours: list[int], per_hour: int = 4, oi=None) -> pd.DataFrame:
    """Ticker snapshots inside the given hour offsets only (others missing)."""
    rows = []
    for h in hours:
        for k in range(per_hour):
            ts = T0 + pd.Timedelta(hours=h, minutes=15 * k)
            rows.append({
                "ts": ts, "mark_price": 100.0 + h, "index_price": 100.0, "open_interest": (oi or (lambda x: 1000.0 + x))(h),
                "ask": 100.5, "bid": 99.5, "ask_amount": np.nan, "bid_amount": np.nan,
                "funding_rate": 1e-4, "predicted_funding_rate": 8e-4,
            })
    return pd.DataFrame(rows)


# --- panel.py ------------------------------------------------------------------

def test_d_oi_is_nan_across_missing_hours():
    hours = [0, 1, 2, 5, 6]  # hours 3 and 4 missing
    hourly = panel.build_ticker_hourly(_ticks_for_hours(hours, oi=lambda h: 1000.0 * math.exp(0.01 * h)))
    assert list(hourly.index) == list(_grid(7))  # complete grid, gaps kept as rows
    assert np.isclose(hourly["d_oi"].iloc[1], 0.01)
    assert hourly["d_oi"].iloc[3:6].isna().all()  # 3, 4 missing; 5's previous hour (4) is missing
    assert np.isclose(hourly["d_oi"].iloc[6], 0.01)  # 6 vs 5 is one elapsed hour
    # Old behaviour would have given d_oi[5] = log(OI_5 / OI_2) = 0.03.


def test_rolling_flow_features_use_elapsed_hours():
    idx = _grid(60).delete(list(range(30, 33)))  # 3-hour gap at 30-32
    hourly = pd.DataFrame({"rv2_1h": 1.0, "rv2_1h_5min": 1.0, "total_volume": 10.0}, index=idx)
    out = panel._apply_rolling_flow_features(hourly)
    assert len(out) == 60
    n = config.RV_WINDOWS_HOURS["primary"]
    rv = out["rv_primary"]
    # Any 24h window touching hours 30-32 is incomplete -> NaN, through hour 32+24.
    assert rv.iloc[30:32 + n].isna().all()
    assert np.isclose(rv.iloc[32 + n], math.sqrt(n))
    assert np.isclose(rv.iloc[29], math.sqrt(n))
    vs = out["volume_shock"]
    assert vs.iloc[30:33 + config.VOLUME_SHOCK_LOOKBACK_HOURS].isna().all()


def test_realised_vol_does_not_bridge_a_multi_hour_trade_gap():
    # Trades every minute in hour 0 at 100, none in hours 1-2, then hour 3 at 110.
    ts = [T0 + pd.Timedelta(minutes=m) for m in range(60)] + [T0 + pd.Timedelta(hours=3, minutes=m) for m in range(60)]
    px = [100.0] * 60 + [110.0] * 60
    rv = panel._realised_vol_1bar(pd.Series(px), pd.Series(ts), "1min")
    # Hour 3 has a flat price inside the hour; the 10% gap move must not be booked there.
    assert np.isclose(rv.loc[T0 + pd.Timedelta(hours=3), "rv2"], 0.0)


# --- funding_cost.py / base.py --------------------------------------------------

def test_forward_ops_refuse_non_grid_index():
    s = pd.Series(1.0, index=_grid(10).delete(4))
    for fn in (lambda: funding_cost.forward_sum(s, 2), lambda: funding_cost.forward_value(s, 2)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError on a non-hourly-grid index")


def test_rolling_backtest_never_sees_unrealised_targets():
    n, H = 400, 24
    p = pd.DataFrame({"f_eff": np.linspace(0, 1, n)}, index=_grid(n))
    p.loc[p.index[100:130], "f_eff"] = np.nan  # a gap inside the training history
    p["C"] = funding_cost.forward_sum(p["f_eff"], H)
    seen = []

    def spy(train_full, train_labeled, origin_row, *, target_col, alpha, horizon_hours):
        origin = origin_row.name
        assert train_full.index[-1] == origin
        assert train_labeled.index[-1] + pd.Timedelta(hours=horizon_hours) <= origin  # target realised by origin
        assert (train_labeled.index == train_full.index[: len(train_labeled)]).all()  # prefix of train_full
        seen.append(origin)
        return 0.0

    out = base.run_rolling_backtest(p, spy, "C", H, min_train_hours=48)
    assert ((out.index[1:] - out.index[:-1]) == pd.Timedelta(hours=H)).all()  # origins exactly H elapsed hours apart
    assert out.index[-1] + pd.Timedelta(hours=H) <= p.index[-1]  # last target fully inside the sample


# --- ARX mean (gas_t, ms_garch_t, joint_path) ----------------------------------

def _arx_frame(n: int, gaps: list[tuple[int, int]], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.5 * f[t - 1] + rng.normal(scale=1e-5)
    df = pd.DataFrame({"f_eff": f}, index=_grid(n))
    for c in arx_mean.DEFAULT_EXOG:
        df[c] = rng.normal(size=n)
    for a, b in gaps:
        df.iloc[a:b] = np.nan
    return df


def test_arx_mean_lag_is_previous_elapsed_hour():
    df = _arx_frame(3000, gaps=[(1000, 1001), (2000, 2030)])
    fit = arx_mean.fit_arx_mean(df, df.iloc[-1], min_obs=500)
    assert len(fit.resid) == len(fit.resid_index)
    assert ((fit.resid_index[1:] - fit.resid_index[:-1]) == pd.Timedelta("1h")).all()  # grid, not spliced rows
    assert fit.resid_index[-1] == df.index[-1]
    r = pd.Series(fit.resid, index=fit.resid_index)
    # Hour 1000 is missing, so hour 1001 has no observed f_{t-1}: no residual.
    assert np.isnan(r.iloc[1000 - 1]) and np.isnan(r.iloc[1001 - 1])
    assert r.iloc[2030 - 1:2031 - 1].isna().all()
    # Manual OLS with correct lags reproduces the slope.
    y = df["f_eff"]; X = pd.concat([y.shift(1)] + [df[c].shift(1) for c in arx_mean.DEFAULT_EXOG], axis=1)
    d = pd.concat([y, X], axis=1).dropna()
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(d)), d.iloc[:, 1:].to_numpy()]), d.iloc[:, 0].to_numpy(), rcond=None)
    assert np.isclose(fit.rho, beta[1], atol=1e-8)


def test_gas_filter_steps_through_missing_hours():
    u = np.array([0.5, np.nan, np.nan, -0.3])
    omega, a_s, b, nu, lam0 = -0.1, 0.05, 0.9, 6.0, 0.0
    _, lam_next = gas._gas_filter(u, omega, a_s, b, nu, lam0)
    lam = lam0
    for x in u:
        if np.isnan(x):
            lam = omega + b * lam
        else:
            z2 = x * x / math.exp(lam)
            lam = omega + a_s * ((nu + 1) / (nu - 2 + z2) * z2 - 1.0) + b * lam
    assert np.isclose(lam_next, lam)


def test_ms_filter_missing_hours_add_no_likelihood():
    # Constant, identical variances in both regimes => each observed hour's
    # likelihood is the same t density whatever the gaps; NaNs add nothing.
    rng = np.random.default_rng(0)
    u_obs = rng.standard_normal(200)
    u = u_obs.copy(); u[[20, 50, 51, 52]] = np.nan
    par = dict(w1=1.0, a1=0.0, b1=0.0, w2=1.0, a2=0.0, b2=0.0, p11=0.9, p22=0.8, nu=7.0)
    nll_gap, *_ = ms_garch._ms_filter(u, *par.values())
    nll_ref, *_ = ms_garch._ms_filter(u_obs[np.isfinite(u)], *par.values())
    assert np.isclose(nll_gap, nll_ref)


# --- SV-t / SVJ-t --------------------------------------------------------------

def test_kalman_missing_hour_is_predict_only():
    x = np.array([-20.0, np.nan, -21.0])
    omega, phi, s2 = -20.0, 0.8, 0.1
    lv = stochastic_vol._kalman_filter_levels(x, omega, phi, s2, stochastic_vol.GAUSSIAN_LOG_CHISQ1_MEAN, stochastic_vol.GAUSSIAN_LOG_CHISQ1_VAR)
    assert np.isclose(lv[1], omega + phi * (lv[0] - omega))
    nll_gap, _ = stochastic_vol._kalman_filter_core(x, omega, phi, s2, stochastic_vol.GAUSSIAN_LOG_CHISQ1_MEAN, stochastic_vol.GAUSSIAN_LOG_CHISQ1_VAR)
    assert np.isfinite(nll_gap)


def test_sv_requires_origin_hour_observed():
    df = _arx_frame(4000, gaps=[(3999, 4000)])  # origin hour itself missing
    assert stochastic_vol._fit_sv_core(df, stochastic_vol.DEFAULT_EXOG, min_obs=500) is None
    df2 = _arx_frame(4000, gaps=[(3000, 3010)])
    fit = stochastic_vol._fit_sv_core(df2, stochastic_vol.DEFAULT_EXOG, min_obs=500)
    assert fit is not None and fit.last_f == df2["f_eff"].iloc[-1]


# --- rough_vol -----------------------------------------------------------------

def test_hurst_increments_use_elapsed_lags():
    # log_rv = 0.01 * hour: every true lag-L increment is exactly 0.01*L, so
    # E|increment| scales with slope exactly 1 in log-log -- with or without
    # gaps. Splicing observed rows (old code) mixes in longer spans.
    n = 2000
    s = pd.Series(0.01 * np.arange(n, dtype=float), index=_grid(n))
    s.iloc[np.r_[100:140, 700:705, 1500]] = np.nan
    H, diag = rough_vol.estimate_hurst(s)
    assert np.isclose(H, 1.0)
    assert np.allclose(diag["m1"], 0.01 * diag["lag"])


def test_rough_state_next_uses_lags_through_origin():
    n, L = 400, 168
    x = pd.Series(np.random.default_rng(3).normal(size=n), index=_grid(n))
    w = np.arange(1, L + 1) ** (0.1 - 0.5); w /= w.sum()
    expected = float(np.dot(w[::-1], x.to_numpy()[n - L:]))  # lags 1..L relative to hour n, i.e. through the origin n-1
    assert np.isclose(rough_vol.rough_vol_state_next(x, 0.1, max_lag=L), expected)


# --- CAViaR / bootstrap ---------------------------------------------------------

def test_caviar_recursion_restarts_after_missing_hour():
    xb = np.array([1.0, 1.0, np.nan, 1.0])
    q = caviar._caviar_recurse(xb, 0.0, 0.5, 2.0)
    assert np.allclose(q[:2], [2.0, 2.0]) and np.isnan(q[2])
    assert np.isclose(q[3], 0.5 * 2.0 + 1.0)  # restarted from q0, not chained to q[1] across the gap


def test_caviar_forecast_is_recursion_at_origin():
    n, H = 1500, 24
    rng = np.random.default_rng(5)
    df = pd.DataFrame({c: rng.normal(size=n) for c in caviar.DEFAULT_FEATURES}, index=_grid(n))
    df["f_eff"] = np.abs(rng.normal(scale=1e-4, size=n))
    df["C"] = funding_cost.forward_sum(df["f_eff"], H)
    pos = n - H - 1
    full, lab = df.iloc[: pos + 1], df.iloc[: pos - H + 1]
    kw = dict(target_col="C", horizon_hours=H, n_restarts=1, clip_nonnegative=False)
    f1 = caviar.caviar_var(full, lab, full.iloc[-1], **kw)
    # Change X only at origin-1, an unlabeled hour between the label cutoff and
    # the origin. The fit (labeled rows only) is unchanged, but q_{origin-1}
    # and hence the forecast must move. The old code jumped from q at the
    # label cutoff straight to the origin, so this hour had no effect.
    full2 = full.copy(); full2.iloc[-2, full2.columns.get_loc("basis")] += 5.0
    f2 = caviar.caviar_var(full2, lab, full2.iloc[-1], **kw)
    assert np.isfinite(f1) and np.isfinite(f2) and not np.isclose(f1, f2, rtol=0, atol=1e-12)
    full3 = full.copy(); full3.iloc[-1, full3.columns.get_loc("basis")] = np.nan
    assert np.isnan(caviar.caviar_var(full3, lab, full3.iloc[-1], **kw))


def test_bootstrap_blocks_never_cross_a_gap():
    x = np.arange(100, dtype=float)
    x[[10, 40, 41, 42, 99]] = np.nan
    paths = block_bootstrap.stationary_bootstrap_paths(x, 2000, 50, 24.0, np.random.default_rng(0))
    assert np.isfinite(paths).all()
    steps = np.diff(paths, axis=1)
    contiguous = steps == 1.0
    # Wherever a path continued its block, the two hours really are adjacent and
    # observed; a block can only "continue" from 9 to 11 etc. by restarting.
    assert not ((paths[:, :-1] == 9.0) & contiguous & (paths[:, 1:] == 11.0)).any()


# --- ARX-GARCH-t ----------------------------------------------------------------

def test_arx_garch_matches_arch_on_gap_free_data():
    from arch import arch_model

    rng = np.random.default_rng(11)
    n = 3000
    e = np.zeros(n); h = np.zeros(n); y = np.zeros(n)
    h[0] = 1.0
    for t in range(1, n):
        h[t] = 0.05 + 0.08 * e[t - 1] ** 2 + 0.9 * h[t - 1]
        e[t] = math.sqrt(h[t]) * rng.standard_t(6) * math.sqrt(4 / 6)
        y[t] = 0.1 + 0.6 * y[t - 1] + e[t]
    X = np.column_stack([np.ones(n - 1), y[:-1]])
    ours = arx_garch._fit_arx_garch_t(y[1:], X)
    ref = arch_model(y[1:], x=y[:-1].reshape(-1, 1), mean="LS", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    assert abs(ours["beta"][1] - ref.params.iloc[1]) < 0.02
    assert abs(ours["a"] + ours["b"] - (ref.params["alpha[1]"] + ref.params["beta[1]"])) < 0.02
    assert abs(ours["nll"] - (-ref.loglikelihood)) / abs(ref.loglikelihood) < 1e-3


def test_garch_filter_missing_hour_advances_variance_by_expectation():
    e = np.array([1.0, np.nan, 0.5])
    nll, eT, hT, obs = arx_garch._garch_t_filter(e, 0.1, 0.1, 0.8, 6.0, 1.0)
    h1 = 0.1 + 0.1 * 1.0 + 0.8 * 1.0
    h2 = 0.1 + 0.9 * h1
    assert np.isclose(hT, h2) and obs and eT == 0.5


# --- PACR (collateral_erosion.py) vs a direct equity-path calculation ------------

def test_pacr_closed_form_matches_direct_equity_path():
    from analysis import collateral_erosion as ce

    rng = np.random.default_rng(7)
    n, H = 300, 24
    idx = pd.date_range("2022-01-09", periods=n, freq="h", tz="UTC")  # spans the 2022-01-11 MM change
    M = 40000 * np.exp(np.cumsum(rng.normal(scale=0.01, size=n)))
    f = rng.normal(2e-5, 4e-5, size=n)
    f[150] = np.nan  # a missing hour: windows containing it must be NaN
    panel = pd.DataFrame({"mark_price": M, "f_eff": f}, index=idx)
    ds = ce.add_leverage_columns(ce.build_pacr_dataset(panel, H), leverage_grid=(10,))
    N = ce.N_USD
    for t in (0, 40, 100, 130, 200):
        for d, name in ((1, "long"), (-1, "short")):
            window = f[t + 1 : t + H + 1]
            got = ds[f"PACRRatio_{name}_L10"].iloc[t]
            if np.isnan(window).any():
                assert np.isnan(got)
                continue
            c0 = (N / 10) / M[t]
            fund, worst = 0.0, 0.0
            for h in range(1, H + 1):
                fund += -d * f[t + h] * N / M[t + h]
                size = N / M[t + h]
                base_rate = 0.01 if idx[t + h] >= pd.Timestamp("2022-01-11", tz="UTC") else 0.0055
                mm = (base_rate + 0.00005 * size) * size
                equity = c0 + d * N * (1 / M[t] - 1 / M[t + h]) + fund
                worst = max(worst, mm - equity)
            assert np.isclose(got, worst / c0, rtol=1e-9, atol=1e-12), (t, name, got, worst / c0)


# --- Binance discrete settlement (2026-11-06, second-venue comparison) ----------

def _binance_ticks_for_hours(hours: list[int], rate_for_hour) -> pd.DataFrame:
    """Binance-style ticker snapshots: `funding_rate` is whatever rate_for_hour(h)
    returns for hour offset h, meant to be held piecewise-constant across each
    8h settlement block (mirroring the real discretely-settled feed -- see
    config.py's note). No predicted_funding_rate column, matching the real
    Binance feed where that field is entirely null."""
    rows = []
    for h in hours:
        for k in range(4):
            ts = T0 + pd.Timedelta(hours=h, minutes=15 * k)
            rows.append({
                "ts": ts, "mark_price": 40000.0 + h, "index_price": 40000.0, "open_interest": 1000.0 + h,
                "ask": 100.5, "bid": 99.5, "funding_rate": rate_for_hour(h),
            })
    return pd.DataFrame(rows)


def test_binance_f_eff_only_at_settlement_hours():
    rates = {0: 1e-4, 8: 2e-4, 16: -3e-4}
    rate_for_hour = lambda h: rates[(h // 8) * 8]

    hourly = panel.build_binance_ticker_hourly(_binance_ticks_for_hours(list(range(24)), rate_for_hour))
    assert list(hourly.index) == list(_grid(24))
    for h in range(24):
        if h in config.BINANCE_SETTLEMENT_HOURS_UTC:
            assert np.isclose(hourly["f_eff"].iloc[h], rates[h]), h
        else:
            assert hourly["f_eff"].iloc[h] == 0.0, h  # exactly zero, not NaN -- no cash flow that hour


def test_binance_f_eff_nan_on_missing_settlement_hour():
    rates = {0: 1e-4, 8: 2e-4, 16: -3e-4}
    rate_for_hour = lambda h: rates[(h // 8) * 8]

    hours = [h for h in range(24) if h != 8]  # hour 8's ticks entirely missing
    hourly = panel.build_binance_ticker_hourly(_binance_ticks_for_hours(hours, rate_for_hour))
    assert np.isnan(hourly["f_eff"].iloc[8])  # missing settlement stays NaN, not silently zeroed
    assert np.isclose(hourly["f_eff"].iloc[0], rates[0])
    assert np.isclose(hourly["f_eff"].iloc[16], rates[16])
    assert hourly["f_eff"].iloc[7] == 0.0  # non-settlement hours around the gap are unaffected
    assert hourly["f_eff"].iloc[9] == 0.0


def test_binance_discrete_settlement_sum_matches_manual():
    """funding_cost.py is reused completely unchanged for Binance (only panel.py
    differs) -- this checks forward_sum(c_long, 72) over the discrete f_eff
    hurdle reproduces a manual sum of max(rate, 0) over exactly the settlements
    falling in (origin, origin+72], each counted once in full (no /8)."""
    rng = np.random.default_rng(11)
    n_blocks = 12
    block_rates = rng.normal(0, 2e-4, size=n_blocks)
    total_hours = n_blocks * 8
    rate_for_hour = lambda h: block_rates[h // 8]

    hourly = panel.build_binance_ticker_hourly(_binance_ticks_for_hours(list(range(total_hours)), rate_for_hour))
    p = funding_cost.add_funding_cost_outcomes(hourly)

    h_horizon = 72  # 9 settlements per full window
    for origin in (0, 10, 17):
        window_hours = range(origin + 1, origin + h_horizon + 1)
        settlement_hours_in_window = [h for h in window_hours if h % 8 == 0 and h < total_hours]
        manual = sum(max(block_rates[h // 8], 0.0) for h in settlement_hours_in_window)
        got = p[f"C_long_H{h_horizon}"].iloc[origin]
        assert origin + h_horizon < total_hours  # sanity: this origin's window is fully in range
        assert np.isclose(got, manual, atol=1e-12), (origin, got, manual)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
