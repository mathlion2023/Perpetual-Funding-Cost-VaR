"""Referee-requested ablation (Topic1_FRL_Referee_Report.pdf, item 1): a
contract-aware forecast that simulates the premium $p_s$ forward and applies
the dampener-and-cap rule (Equation~1 in the letter) to each simulated path,
compared against the existing historical-simulation baseline (same
information and estimation window, Section 5), with the dampener and the
cap switched off in turn to isolate each feature's contribution to the
undercoverage result in Table 3.

Premium paths are simulated by an hourly IID bootstrap of the realised
`basis` column (= (mark_price - index_price) / index_price, panel.py) over
the training window -- the same non-parametric, no-fitted-parameters spirit
as `historical_simulation.py`, so this is a genuinely matched baseline and
not a new parametric estimator competing on different terms. This is a
documented simplification (an IID bootstrap does not reproduce basis's own
serial correlation) left for a longer-form treatment; the ablation's
purpose is to isolate the rule's two features, not to win the horse race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DELTA = 0.00025  # dampener band, premium units
KAPPA_BAR = 0.005  # 8-hour rate cap


def _apply_rule(p: np.ndarray, use_dampener: bool, use_cap: bool) -> np.ndarray:
    """phi_s = Pi_[-kappa,kappa]( (p-delta)^+ - (-p-delta)^+ ), Equation 1.
    use_dampener=False skips the inner zero-band clamp (phi_s = p directly,
    still subject to the cap if use_cap). use_cap=False skips the outer
    projection onto [-kappa,kappa]."""
    if use_dampener:
        phi = np.clip(p - DELTA, 0.0, None) - np.clip(-p - DELTA, 0.0, None)
    else:
        phi = p
    if use_cap:
        phi = np.clip(phi, -KAPPA_BAR, KAPPA_BAR)
    return phi


def contract_aware_var(
    train_full: pd.DataFrame,
    train_labeled: pd.DataFrame,
    origin_row: pd.Series,
    *,
    target_col: str,
    alpha: float = 0.95,
    horizon_hours: int,
    n_sims: int = 2000,
    seed: int = 0,
    use_dampener: bool = True,
    use_cap: bool = True,
    min_obs: int = 24 * 180,
) -> float:
    """Simulate `n_sims` forward premium paths by an IID bootstrap of
    historical hourly `basis`, apply the rule (with the requested features
    switched on/off) to get simulated f_t, sum the positive part over the
    horizon, and return the alpha-quantile across simulated paths. This is
    the gross-funding (C^g) target by construction; it is not adapted for
    net/signed targets."""
    hist = train_full["basis"].dropna().to_numpy()
    if len(hist) < min_obs:
        return np.nan
    rng = np.random.default_rng(seed)
    p_sim = rng.choice(hist, size=(n_sims, horizon_hours), replace=True)
    phi_sim = _apply_rule(p_sim, use_dampener, use_cap)
    f_sim = phi_sim / 8.0
    c_g_sim = np.clip(f_sim, 0.0, None).sum(axis=1)
    return float(np.quantile(c_g_sim, alpha))
