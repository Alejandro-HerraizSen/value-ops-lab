from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    """
    Validate required columns exist in a DataFrame.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def dso(
    ar_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    on: Literal["month", "date"] = "month",
    ar_col: str = "ar_balance",
    sales_col: str = "sales",
) -> pd.DataFrame:
    """
    Compute Days Sales Outstanding by period.

    DSO = 365 * (Average AR) / Sales

    Parameters
    ----------
    ar_df : pd.DataFrame with columns [on, ar_col]
    sales_df : pd.DataFrame with columns [on, sales_col]
    on : str
        Grouping key. Typically "month".
    ar_col : str
        Column name for AR balance.
    sales_col : str
        Column name for sales.

    Returns
    -------
    pd.DataFrame with [on, "DSO"]
    """
    _validate_columns(ar_df, [on, ar_col], "ar_df")
    _validate_columns(sales_df, [on, sales_col], "sales_df")

    ar = ar_df.groupby(on)[ar_col].mean()
    sales = sales_df.groupby(on)[sales_col].sum()

    # Protect against divide by zero
    dso_series = 365.0 * (ar / sales.replace(0.0, np.nan)).fillna(0.0)
    return dso_series.reset_index(name="DSO")


def dpo(
    ap_df: pd.DataFrame,
    cogs_df: pd.DataFrame,
    on: Literal["month", "date"] = "month",
    ap_col: str = "ap_balance",
    cogs_col: str = "cogs",
) -> pd.DataFrame:
    """
    Compute Days Payables Outstanding by period.

    DPO = 365 * (Average AP) / COGS
    """
    _validate_columns(ap_df, [on, ap_col], "ap_df")
    _validate_columns(cogs_df, [on, cogs_col], "cogs_df")

    ap = ap_df.groupby(on)[ap_col].mean()
    cogs = cogs_df.groupby(on)[cogs_col].sum()
    dpo_series = 365.0 * (ap / cogs.replace(0.0, np.nan)).fillna(0.0)
    return dpo_series.reset_index(name="DPO")


def dio(
    inv_df: pd.DataFrame,
    cogs_df: pd.DataFrame,
    on: Literal["month", "date"] = "month",
    inv_col: str = "inventory",
    cogs_col: str = "cogs",
) -> pd.DataFrame:
    """
    Compute Days Inventory Outstanding by period.

    DIO = 365 * (Average Inventory) / COGS
    """
    _validate_columns(inv_df, [on, inv_col], "inv_df")
    _validate_columns(cogs_df, [on, cogs_col], "cogs_df")

    inv = inv_df.groupby(on)[inv_col].mean()
    cogs = cogs_df.groupby(on)[cogs_col].sum()
    dio_series = 365.0 * (inv / cogs.replace(0.0, np.nan)).fillna(0.0)
    return dio_series.reset_index(name="DIO")


def ccc(
    dso_df: pd.DataFrame,
    dpo_df: pd.DataFrame,
    dio_df: pd.DataFrame,
    on: Literal["month", "date"] = "month",
) -> pd.DataFrame:
    """
    Compute CCC per period combining DSO, DPO, DIO.

    CCC = DSO + DIO - DPO

    Parameters
    ----------
    dso_df, dpo_df, dio_df : pd.DataFrame
        Each must have columns [on, "DSO"/"DPO"/"DIO"].

    Returns
    -------
    pd.DataFrame with [on, DSO, DPO, DIO, CCC]
    """
    _validate_columns(dso_df, [on, "DSO"], "dso_df")
    _validate_columns(dpo_df, [on, "DPO"], "dpo_df")
    _validate_columns(dio_df, [on, "DIO"], "dio_df")

    df = dso_df.merge(dpo_df, on=on, how="outer").merge(dio_df, on=on, how="outer")
    df = df.sort_values(by=on).reset_index(drop=True)
    df["CCC"] = df["DSO"].fillna(0.0) + df["DIO"].fillna(0.0) - df["DPO"].fillna(0.0)
    return df


def cash_unlocked(current_ccc: float, target_ccc: float, daily_sales: float) -> float:
    """
    Estimate cash unlocked by reducing CCC.

    Parameters
    ----------
    current_ccc : float
        Current CCC in days.
    target_ccc : float
        Target CCC in days. Must be less than or equal to current_ccc.
    daily_sales : float
        Average daily sales in currency units.

    Returns
    -------
    float
        Estimated cash unlocked.

    Notes
    -----
    This is a simple lever illustration. In client work you should
    confirm driver causality and implementation feasibility.
    """
    delta_days = max(current_ccc - target_ccc, 0.0)
    return float(delta_days) * float(daily_sales)