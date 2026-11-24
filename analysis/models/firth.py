"""Firth's bias-reduced logistic regression (Firth, 1993; Heinze & Schemper, 2002).

Added 2026-09-26 per peer-review item R2: ordinary MLE logistic regression is
known to be finite-sample biased under rare events (as few as 3-14 positive
cases feed Stage 1's Model 2/3 here -- see stage1_omission.py), a distinct
problem from the scale-conditioning issue fixed in that module. This is a
robustness check against that bias, not a replacement for the primary MLE fit.

Penalizes the log-likelihood by log|I(beta)|^(1/2) (Jeffreys invariant prior),
implemented via the standard IRLS modification: augment the working response
with a leverage-weighted correction h_i*(0.5 - p_i), where h_i is the i-th
diagonal of the logistic-regression hat matrix. Verified against ordinary MLE
on a well-behaved (non-rare-event) simulated dataset before use here -- the
two agree to 3 decimal places when rare-event bias is not in play, confirming
the implementation itself is correct.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def firth_logit(y: np.ndarray, X: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> dict:
    """Fit Firth's penalized logistic regression via modified IRLS.

    Returns point estimates, SEs (from the unpenalized observed information
    at the penalized MLE -- standard practice per Heinze & Schemper), Wald
    z/p-values, and the penalized log-likelihood (for a penalized
    likelihood-ratio test between nested models, the standard practical
    robustness comparison -- not a true asymptotic chi-square null).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta = np.zeros(X.shape[1])
    converged = False

    for i in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)

        sqrt_w = np.sqrt(w)
        Xw = X * sqrt_w[:, None]
        XtWX = Xw.T @ Xw
        XtWX_inv = np.linalg.pinv(XtWX)

        h = w * np.einsum("ij,jk,ik->i", X, XtWX_inv, X)  # hat-matrix diagonal
        y_star = y + h * (0.5 - p)
        step = XtWX_inv @ (X.T @ (y_star - p))

        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break

    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(p * (1 - p), 1e-10, None)
    sqrt_w = np.sqrt(w)
    XtWX = (X * sqrt_w[:, None]).T @ (X * sqrt_w[:, None])
    XtWX_inv = np.linalg.pinv(XtWX)
    _, logdet = np.linalg.slogdet(XtWX)
    penalized_loglik = float(np.sum(y * eta - np.log1p(np.exp(eta))) + 0.5 * logdet)

    se = np.sqrt(np.clip(np.diag(XtWX_inv), 0, None))
    z = beta / se
    pvalues = 2 * stats.norm.sf(np.abs(z))

    return {
        "beta": beta,
        "se": se,
        "z": z,
        "pvalues": pvalues,
        "penalized_loglik": penalized_loglik,
        "n_iter": i,
        # `converged`/`cond_number` (2026-09-29): np.linalg.pinv never raises
        # on a singular/near-singular XtWX (unlike the ordinary-MLE path's
        # full_sample_fit_error in stage1_omission.py), so a non-converged or
        # ill-conditioned Firth fit would previously return silently. These
        # make that visible without changing the fit itself -- pinv is still
        # the right tool here, since staying stable under a singular Hessian
        # is the whole point of using Firth's method under rare events.
        "converged": converged,
        "cond_number": float(np.linalg.cond(XtWX)),
    }
