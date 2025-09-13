from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict
from sklearn.ensemble import GradientBoostingRegressor

# cvxpy is optional at import time so the rest of the library still loads.
try:
    import cvxpy as cp
    _HAS_CVX = True
except Exception:  # pragma: no cover
    cp = None
    _HAS_CVX = False


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
) -> tuple[float, float]:
    """
    Compute an optimal cash buffer using a CVaR-style convex program.

    Prefers Clarabel; falls back to ECOS, then SCS if needed.
    """
    if not _HAS_CVX:
        raise ImportError(
            "cvxpy is required for cvar_cash_buffer. "
            "Install cvxpy (and clarabel or ecos) to enable this feature."
        )

    s = np.asarray(scenarios, dtype=float).ravel()
    n = s.shape[0]
    if n == 0:
        return 0.0, 0.0

    b = cp.Variable(nonneg=True)   # buffer
    t = cp.Variable()              # VaR-like threshold
    z = cp.Variable(n, nonneg=True)

    # z_i >= -(s_i + b) - t
    constraints = [z >= -(s + b) - t]

    objective = cp.Minimize(b + (1.0 / ((1.0 - alpha) * n)) * cp.sum(z))
    prob = cp.Problem(objective, constraints)

    # Prefer Clarabel; graceful fallbacks
    solved = False
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            prob.solve(solver=getattr(cp, solver), verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except Exception:
            continue

    if not solved or b.value is None or t.value is None:
        raise RuntimeError(f"CVaR optimization failed with status: {prob.status}")

    return float(b.value), float(t.value)