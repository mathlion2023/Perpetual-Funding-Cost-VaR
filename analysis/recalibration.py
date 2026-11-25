"""Section 4.6.6 (added 2026-09-28): adaptive conformal recalibration layer.

A model-agnostic post-processing step applied to any model's backtest output
(the DataFrame returned by models.base.run_rolling_backtest). It answers a
separate question from model choice: how much of a model's miscalibration
can a simple online correction remove, and at what cost in quantile loss?

Method -- adaptive conformal inference (Gibbs & Candes, 2021) on
forecast-error scores:

    score_i      = realized_i - VaR_i                  (past, fully realised origins only)
    VaR_adj_t    = VaR_t + Quantile_{1 - a_t}(score_1, ..., score_{t-1})
    err_t        = 1[realized_t > VaR_adj_t]
    a_{t+1}      = a_t + gamma * (a_target - err_t),    a_target = 1 - alpha (= 0.05)

The adjustment at origin t uses only scores from earlier origins. Because
origins are non-overlapping and spaced exactly H hours apart, the previous
origin's H-hour outcome is fully realised by origin t, so the layer
preserves the out-of-sample discipline of Section 4.8. Recalibrated forecasts
are always reported alongside -- never instead of -- the raw forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GAMMAS = (0.005, 0.02, 0.05)


def aci_recalibrate(bt: pd.DataFrame, alpha: float = 0.95, gamma: float = 0.02, min_history: int = 20) -> pd.DataFrame:
    df = bt[["var_forecast", "realized"]].dropna().sort_index()
    a_target = 1.0 - alpha
    a_t = a_target
    scores: list[float] = []
    adj, levels = [], []
    for var_t, real_t in zip(df["var_forecast"].to_numpy(), df["realized"].to_numpy()):
        if len(scores) >= min_history:
            level = float(np.clip(1.0 - a_t, 0.0, 1.0))
            v = var_t + float(np.quantile(scores, level))
        else:
            level = np.nan
            v = var_t  # warm-up: raw forecast until enough scores exist
        adj.append(v)
        levels.append(level)
        err = float(real_t > v)
        if len(scores) >= min_history:
            # clip a_t itself, not just the derived `level` (code audit,
            # 2026-11-25): without this, a long low-exception stretch can
            # drift a_t outside [0,1], and the derived level then saturates
            # at 0/1 for several subsequent origins before a_t re-enters
            # range, even though a_t is reported as though still meaningful.
            a_t = float(np.clip(a_t + gamma * (a_target - err), 0.0, 1.0))
        scores.append(real_t - var_t)
    out = pd.DataFrame({"var_forecast": adj, "realized": df["realized"].to_numpy(), "aci_level": levels}, index=df.index)
    out["exception"] = (out["realized"] > out["var_forecast"]).astype(float)
    return out


def rolling_quantile_recalibrate(bt: pd.DataFrame, alpha: float = 0.95, min_history: int = 20) -> pd.DataFrame:
    """Referee-requested simpler baseline (Topic1_FRL_Referee_Report.pdf,
    item 3): static split-conformal correction, i.e. aci_recalibrate with the
    online a_t update removed. The adjustment at each origin is var_t plus
    the alpha-quantile of all earlier (fully realised) forecast errors --
    the level never moves away from alpha, so this isolates how much of
    ACI's gain comes from the adaptive step itself versus from simply
    adding an expanding-window empirical quantile correction."""
    df = bt[["var_forecast", "realized"]].dropna().sort_index()
    scores: list[float] = []
    adj = []
    for var_t, real_t in zip(df["var_forecast"].to_numpy(), df["realized"].to_numpy()):
        if len(scores) >= min_history:
            v = var_t + float(np.quantile(scores, alpha))
        else:
            v = var_t  # same warm-up convention as aci_recalibrate
        adj.append(v)
        scores.append(real_t - var_t)
    out = pd.DataFrame({"var_forecast": adj, "realized": df["realized"].to_numpy()}, index=df.index)
    out["exception"] = (out["realized"] > out["var_forecast"]).astype(float)
    return out
