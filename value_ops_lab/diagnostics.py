from __future__ import annotations

import pandas as pd


def waterfall_ccc_impacts(baseline_row: pd.Series, shifts: dict) -> pd.DataFrame:
    """
    Build a simple waterfall of incremental CCC improvements.

    Parameters
    ----------
    baseline_row : pd.Series
        Row with fields ["CCC", "DSO", "DPO", "DIO"] at baseline.
    shifts : dict
        Desired changes in days for each lever.
        Example: {"DSO": -5, "DIO": -3, "DPO": +4}
        Convention:
          - Negative shift on DSO or DIO improves CCC.
          - Positive shift on DPO improves CCC.

    Returns
    -------
    pd.DataFrame with columns [lever, delta_days, new_CCC]

    Notes
    -----
    The order of keys in 'shifts' matters for the intermediate steps.
    """
    required = {"CCC", "DSO", "DPO", "DIO"}
    if not required.issubset(set(baseline_row.index)):
        missing = required - set(baseline_row.index)
        raise KeyError(f"baseline_row is missing fields: {missing}")

    steps = []
    current_ccc = float(baseline_row["CCC"])
    for lever, shift in shifts.items():
        # For DPO, increasing days reduces CCC, so net effect is subtract shift
        if lever == "DPO":
            new_ccc = current_ccc - shift
            delta_days = shift
        else:
            new_ccc = current_ccc + shift
            delta_days = -shift
        steps.append({"lever": lever, "delta_days": delta_days, "new_CCC": new_ccc})
        current_ccc = new_ccc

    return pd.DataFrame(steps)


def flag_invoice_anomalies(df: pd.DataFrame, amount_col: str = "amount") -> pd.DataFrame:
    """
    Flag basic amount outliers using IQR rule.

    Parameters
    ----------
    df : pd.DataFrame
        Invoice or GL-like DataFrame containing an amount column.
    amount_col : str
        Numeric amount column.

    Returns
    -------
    pd.DataFrame
        Subset of rows flagged as outliers with a 'reason' column.

    Notes
    -----
    This is a lightweight sanity check. For robust anomaly detection in practice,
    consider IsolationForest or robust regression against expected drivers.
    """
    if amount_col not in df.columns:
        raise KeyError(f"Column not found: {amount_col}")

    q1 = df[amount_col].quantile(0.25)
    q3 = df[amount_col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers = df[(df[amount_col] < lo) | (df[amount_col] > hi)].copy()
    outliers["reason"] = "amount_outlier"
    return outliers