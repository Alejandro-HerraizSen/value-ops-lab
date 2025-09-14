from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, Tuple

from sklearn.ensemble import GradientBoostingRegressor

# --- Optional deps (kept optional so the package imports cleanly) ---
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


# ============================================================================
# Probabilistic forecasting (simple, lightweight)
# ============================================================================

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


# ============================================================================
# Empirical tail metrics (used both for reporting and degeneracy detection)
# ============================================================================

def empirical_left_var(x: np.ndarray, alpha: float) -> float:
    """
    Left-tail VaR on a series x (downside). VaR_{alpha} (left) = quantile at (1 - alpha).
    """
    x = np.asarray(x, float).ravel()
    return float(np.quantile(x, 1.0 - alpha))


def empirical_left_cvar(x: np.ndarray, alpha: float) -> float:
    """
    Left-tail CVaR on a series x (empirical average of worst (1-alpha) tail).
    If the tail set is empty (alpha=1), returns min(x).
    """
    x = np.asarray(x, float).ravel()
    q = empirical_left_var(x, alpha)
    tail = x[x <= q]
    if tail.size == 0:
        tail = np.array([x.min()])
    return float(tail.mean())


# ============================================================================
# CVaR buffer sizing under downside risk
# ============================================================================

def _solve_cvar_cvxpy(s: np.ndarray, alpha: float, eps: float = 1e-6) -> Tuple[float, float, str, str]:
    """
    Internal: CVXPY solve with ε-regularization.
    Returns (b, t, status, solver_name) or raises.
    """
    if not _HAS_CVX:
        raise ImportError("cvxpy not available")

    n = s.size
    b = cp.Variable(nonneg=True)
    t = cp.Variable()
    z = cp.Variable(n, nonneg=True)

    # Constraints:
    # z_i >= -(s_i + b) - t
    # t + (1/((1-alpha)*n)) * sum z_i <= 0
    constraints = [
        z >= -(s + b) - t,
        t + (1.0 / ((1.0 - alpha) * n)) * cp.sum(z) <= 0.0,
    ]

    # ε-regularized objective to avoid b–t degeneracy
    avg_z = (1.0 / n) * cp.sum(z)
    prob = cp.Problem(cp.Minimize(b + eps * (t + avg_z)), constraints)

    # Prefer Clarabel -> ECOS -> SCS
    for solver in ("CLARABEL", "ECOS", "SCS"):
        try:
            kwargs = {}
            if solver == "ECOS":
                kwargs = dict(abstol=1e-8, reltol=1e-8, feastol=1e-8, verbose=False)
            if solver == "SCS":
                kwargs = dict(eps=1e-5, max_iters=20000, verbose=False)
            prob.solve(solver=getattr(cp, solver), **kwargs)
            if prob.status in ("optimal", "optimal_inaccurate") and b.value is not None:
                return float(b.value), float(t.value), prob.status, solver
        except Exception:
            continue

    raise RuntimeError(f"CVXPY failed: status={prob.status}")


def _solve_cvar_scipy(s: np.ndarray, alpha: float) -> Tuple[float, float, str]:
    """
    Internal: SciPy HiGHS LP solve (no ε needed).
    Returns (b, t, solver_name) or raises.
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy not available")

    n = s.size

    # Decision vector x = [b, t, z_1..z_n]
    # Objective: minimize b  -> c = [1, 0, 0..0]
    c = np.zeros(2 + n, dtype=float)
    c[0] = 1.0

    # Constraints:
    # 1) z_i >= -(s_i + b) - t  ->  -b - t - z_i <= s_i
    A_ub = np.zeros((n + 1, 2 + n), dtype=float)
    b_ub = np.zeros(n + 1, dtype=float)
    for i in range(n):
        A_ub[i, 0] = -1.0   # -b
        A_ub[i, 1] = -1.0   # -t
        A_ub[i, 2 + i] = -1.0  # -z_i
        b_ub[i] = s[i]

    # 2) t + (1/((1-alpha)*n)) * sum z_i <= 0
    A_ub[n, 1] = 1.0
    A_ub[n, 2:] = 1.0 / ((1.0 - alpha) * n)
    b_ub[n] = 0.0

    # Bounds: b >= 0, t free, z_i >= 0
    bounds = [(0, None), (-1e12, 1e12)] + [(0, None)] * n

    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs", options={"presolve": True}
    )
    if not res.success:
        raise RuntimeError(f"HiGHS failed: {res.message}")

    return float(res.x[0]), float(res.x[1]), "HiGHS"


def cvar_cash_buffer(
    scenarios: np.ndarray,
    alpha: float = 0.95,
    eps: float = 1e-6,
    prefer: str = "auto",
    return_solver: bool = False,
) -> Tuple[float, float] | Tuple[float, float, str]:
    r"""
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
    is non-positive. The ε-term avoids b–t cancellation that can return b≈0
    even when tail losses are negative.

    Robustness
    ----------
    Some CVXPY/solver combinations can still exhibit degeneracy on small N.
    We detect this by comparing against the empirical CVaR buffer and
    **fallback to SciPy HiGHS** if the CVXPY result is inconsistent.

    Parameters
    ----------
    scenarios : np.ndarray
        Cashflow scenarios (negative = shortfall).
    alpha : float
        CVaR confidence (0<alpha<1), e.g., 0.95.
    eps : float
        Small regularization weight (default 1e-6).
    prefer : {"auto","cvxpy","scipy"}
        Solver preference:
          - "auto"  : try CVXPY first, fallback to SciPy on degeneracy/failure
          - "cvxpy" : force CVXPY
          - "scipy" : force SciPy HiGHS
    return_solver : bool
        If True, return (b, t, solver_label).

    Returns
    -------
    (b_opt, t_opt[, solver_label])
    """
    s = np.asarray(scenarios, float).ravel()
    n = s.size
    if n == 0:
        out = (0.0, 0.0, "empty") if return_solver else (0.0, 0.0)
        return out

    # Empirical CVaR buffer used to detect degeneracy
    emp_cvar = empirical_left_cvar(s, alpha)
    emp_buf = max(0.0, -emp_cvar)

    # Forced solver paths
    p = prefer.lower()
    if p == "scipy":
        b, t, label = _solve_cvar_scipy(s, alpha)
        return (b, t, label) if return_solver else (b, t)
    if p == "cvxpy":
        b, t, _, label = _solve_cvar_cvxpy(s, alpha, eps)
        return (b, t, label) if return_solver else (b, t)

    # Auto: try CVXPY, check degeneracy; fallback to SciPy
    try:
        b, t, status, label = _solve_cvar_cvxpy(s, alpha, eps)
        # Degeneracy heuristic: empirical says buffer>0, but cvxpy returns ~0
        if (b <= 1e-6) and (emp_buf > 1e-6):
            b2, t2, label2 = _solve_cvar_scipy(s, alpha)
            return (b2, t2, f"{label2} (cvxpy_degenerate)") if return_solver else (b2, t2)
        return (b, t, label) if return_solver else (b, t)
    except Exception:
        b2, t2, label2 = _solve_cvar_scipy(s, alpha)
        return (b2, t2, label2) if return_solver else (b2, t2)