from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict
from sklearn.ensemble import GradientBoostingRegressor

# Optional dependencies at import time
try:
    import cvxpy as cp
    _HAS_CVX = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVX = False

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def quantile_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    X_future: pd.DataFrame,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
    random_state: int = 7,
) -> Dict[float, np.ndarray]:
    """
    Train simple quantile regressors and generate probabilistic forecasts.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Target series.
    X_future : pd.DataFrame
        Feature matrix for the forecast horizon.
    quantiles : iterable of float
        Quantiles to predict. Values in (0, 1).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Mapping quantile -> np.ndarray of predictions.
    """
    preds: Dict[float, np.ndarray] = {}
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            random_state=random_state,
        )
        model.fit(X, y)
        preds[q] = model.predict(X_future)
    return preds


def cvar_cash_buffer(
    scenarios: np.ndarray,
    alpha: float = 0.95,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """
    Minimum buffer sizing with a CVaR constraint.

    Variables
    ---------
    b >= 0   (buffer to add)
    t  free  (VaR-like threshold)
    z_i >= 0 (hinge auxiliaries)

    Problem
    -------
    minimize      b + eps*(t + avg(z))
    subject to    z_i >= -(s_i + b) - t      for all i
                  t + (1/((1-alpha)*N)) * sum_i z_i <= 0
                  b >= 0, z_i >= 0

    Intuition
    ---------
    We choose the smallest buffer b such that the CVaR_α of the shortfall

        loss_i = -(s_i + b)

    is non-positive. This guarantees that, on average, the worst (1–α)
    scenarios are covered. The ε-term in the objective avoids the
    b–t cancellation degeneracy that can return b=0 even when losses
    are deeply negative.

    Parameters
    ----------
    scenarios : np.ndarray
        Array of cashflow scenarios.
    alpha : float
        CVaR confidence level in (0,1). Typical = 0.95.
    eps : float
        Small regularization weight to stabilize optimization.

    Returns
    -------
    (b_opt, t_opt) : tuple of float
        Optimal buffer b and associated VaR-like threshold t.
    """
    s = np.asarray(scenarios, dtype=float).ravel()
    n = s.shape[0]
    if n == 0:
        return 0.0, 0.0

    # ---- cvxpy version ----
    if _HAS_CVX:
        b = cp.Variable(nonneg=True)
        t = cp.Variable()
        z = cp.Variable(n, nonneg=True)

        constraints = [
            z >= -(s + b) - t,
            t + (1.0 / ((1.0 - alpha) * n)) * cp.sum(z) <= 0.0,
        ]
        avg_z = (1.0 / n) * cp.sum(z)

        # Regularized objective to break degeneracy
        prob = cp.Problem(cp.Minimize(b + eps * (t + avg_z)), constraints)

        for solver in ("CLARABEL", "ECOS", "SCS"):
            try:
                prob.solve(solver=getattr(cp, solver), verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate"):
                    return float(b.value), float(t.value)
            except Exception:
                continue
        raise RuntimeError(f"CVaR optimization failed: {prob.status}")

    # ---- SciPy LP fallback ----
    if not _HAS_SCIPY:
        raise ImportError("Neither cvxpy nor SciPy is available.")

    c = np.zeros(2 + n, dtype=float)
    c[0] = 1.0

    A_ub = np.zeros((n + 1, 2 + n), dtype=float)
    b_ub = np.zeros(n + 1, dtype=float)
    for i in range(n):
        A_ub[i, 0] = -1.0   # -b
        A_ub[i, 1] = -1.0   # -t
        A_ub[i, 2 + i] = -1.0  # -z_i
        b_ub[i] = s[i]

    A_ub[n, 1] = 1.0
    A_ub[n, 2:] = 1.0 / ((1.0 - alpha) * n)
    b_ub[n] = 0.0

    bounds = [(0, None), (-1e12, 1e12)] + [(0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"presolve": True})
    if not res.success:
        raise RuntimeError(f"SciPy LP solver failed: {res.message}")

    return float(res.x[0]), float(res.x[1])