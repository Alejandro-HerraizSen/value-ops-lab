from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict


def dp_pay_terms(
    cash_today: float,
    obligations: List[Tuple[float, int]],
    discount: float,
    penalty: float,
    horizon: int = 3,
    grid_size: int = 51,
) -> Dict[Tuple[int, float], str]:
    """
    Toy dynamic program for pay-now vs delay decisions on obligations.

    This is illustrative and meant for discussion with stakeholders.
    It shows how a DP structure can encode cash constraints and timing.

    Parameters
    ----------
    cash_today : float
        Current cash on hand used to define the discretization grid.
    obligations : list of (amount, due_in_periods)
        Each tuple contains the obligation amount and when it is due.
        Example: [(10000, 0), (7000, 1), (5000, 2)]
    discount : float
        Early payment discount as a fraction. Example: 0.02 for 2 percent.
    penalty : float
        Late payment penalty per period as a fraction. Example: 0.015.
    horizon : int
        Number of decision periods. Often equals len(obligations) for the toy case.
    grid_size : int
        Number of discrete cash states for the approximation grid.

    Returns
    -------
    dict
        Mapping from (period, cash_state) to action label "pay_now" or "delay".

    Notes
    -----
    - This is not a full working capital optimization engine.
    - It is intentionally simple to keep code readable and suitable for demos.
    - For production, model multi-obligation states, cash transitions, and
      policy constraints. Consider linear or stochastic programming.
    """
    total_obl = sum(a for a, _ in obligations)
    grid_max = max(cash_today, total_obl) * 1.5
    grid = np.linspace(0.0, grid_max, grid_size)

    # Value function V[t, i] for period t and cash grid index i
    V = np.zeros((horizon + 1, grid_size))
    policy: Dict[Tuple[int, float], str] = {}

    # Work backwards
    for t in range(horizon - 1, -1, -1):
        # Pick an obligation for the period in a simple sequence
        idx = min(t, len(obligations) - 1)
        amt, due = obligations[idx]

        for i, c in enumerate(grid):
            # If we pay now, cost is amount discounted, limited by cash on hand
            pay_now_cost = max(amt * (1.0 - discount) - c, 0.0)

            # If we delay and the obligation is due now or earlier, incur penalty
            delay_cost = amt * (1.0 + penalty) if due <= t else 0.0

            # Continuing value is V[t+1, i] since we do not model cash transition here
            a_pay = pay_now_cost + V[t + 1, i]
            a_delay = delay_cost + V[t + 1, i]

            if a_pay <= a_delay:
                V[t, i] = a_pay
                policy[(t, float(grid[i]))] = "pay_now"
            else:
                V[t, i] = a_delay
                policy[(t, float(grid[i]))] = "delay"

    return policy