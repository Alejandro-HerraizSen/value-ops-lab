from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Union


def read_csv(path: Union[str, Path], **read_kwargs) -> pd.DataFrame:
    """
    Read a CSV file with basic validation.

    Parameters
    ----------
    path : str or Path
        File path to the CSV file.
    read_kwargs : dict
        Extra keyword args for pandas.read_csv. For example, dtype={"col": str}.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    ValueError
        If the file is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p, **read_kwargs)
    if df.shape[0] == 0 and df.shape[1] == 0:
        raise ValueError(f"Empty CSV: {p}")
    return df


def to_month(dt_col: pd.Series) -> pd.Series:
    """
    Convert a datetime-like Series to month-start timestamps.

    This helps standardize time granularity for DSO/DPO/DIO calculations.

    Parameters
    ----------
    dt_col : pd.Series
        Datetime or string-like series.

    Returns
    -------
    pd.Series
        Timestamps at month start.
    """
    return pd.to_datetime(dt_col, errors="coerce").dt.to_period("M").dt.to_timestamp()