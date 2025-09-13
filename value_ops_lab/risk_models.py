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

    This solves a common risk-aware sizing problem: choose buffer 'b'
    that minimizes a convex surrogate of tail shortfall while keeping
    the formulation simple for demos.

    Parameters
    ----------
    scenarios : np.ndarray
        Array of simulated end-of-period cash balances without buffer.
        Negative values represent shortfalls.
    alpha : float
        Confidence level for CVaR. Typical values are 0.90 to 0.99.

    Returns
    -------
    (buffer, aux_threshold) : tuple of float
        buffer : optimal nonnegative cash buffer.
        aux_threshold : auxiliary threshold variable value.

    Notes
    -----
    - Requires cvxpy to be installed. If unavailable, raises ImportError.
    - This is a demonstration model. In production, you may want to add
      explicit cost of capital, liquidity covenants, and multi-period dynamics.
    """
    if not _HAS_CVX:
        raise ImportError(
            "cvxpy is required for cvar_cash_buffer. "
            "Install cvxpy or use a fallback optimizer."
        )

    scenarios = np.asarray(scenarios, dtype=float).ravel()
    n = scenarios.shape[0]

    b = cp.Variable(nonneg=True)          # buffer to size
    t = cp.Variable()                     # threshold (VaR-like)
    z = cp.Variable(n, nonneg=True)       # auxiliary variables for CVaR hinge

    # z_i >= -(s_i + b) - t
    constraints = [z >= -(scenarios + b) - t]

    # Minimize b + (1/((1 - alpha) * n)) * sum(z)
    objective = cp.Minimize(b + (1.0 / ((1.0 - alpha) * n)) * cp.sum(z))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    return float(b.value), float(t.value)