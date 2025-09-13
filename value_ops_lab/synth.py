from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n_months: int = 24, seed: int = 7) -> pd.DataFrame:
    """
    Create a small synthetic finance dataset with monthly granularity.

    Columns
    -------
    month : Timestamp at month start
    sales : revenue per month
    cogs : cost of goods sold per month
    ar_balance : average AR balance for the month
    ap_balance : average AP balance for the month
    inventory : average inventory balance for the month

    Parameters
    ----------
    n_months : int
        Number of months to generate.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", periods=n_months, freq="MS")

    # Build a simple random walk revenue with noise and a floor
    sales = np.maximum(1e5 + rng.normal(0, 2e4, n_months).cumsum(), 5e4)
    cogs = 0.6 * sales + rng.normal(0, 5e3, n_months)

    # Working capital balances proportional to flows plus small noise
    ar = 0.25 * sales + rng.normal(0, 3e3, n_months)
    ap = 0.15 * cogs + rng.normal(0, 2e3, n_months)
    inv = 0.20 * cogs + rng.normal(0, 2e3, n_months)

    df = pd.DataFrame(
        {
            "month": months,
            "sales": sales,
            "cogs": cogs,
            "ar_balance": ar,
            "ap_balance": ap,
            "inventory": inv,
        }
    )
    return df