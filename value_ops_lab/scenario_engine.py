from __future__ import annotations

import pandas as pd
from typing import Dict


def ebitda_bridge(df: pd.DataFrame, drivers: Dict[str, float]) -> pd.DataFrame:
    """
    Build a simple EBITDA bridge from base to new using driver deltas.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns:
          - price
          - volume
          - cogs_pct (fraction of revenue)
          - opex (absolute)
    drivers : dict
        Percentage deltas for drivers. Example:
        {"price": 0.02, "volume": 0.03, "mix": 0.01, "cogs_pct": -0.01, "opex": -0.02}
        Mix is applied as a revenue uplift proxy.

    Returns
    -------
    pd.DataFrame
        Tidy table with steps and values:
        [Base EBITDA, Price/Vol/Mix, COGS%, Opex, New EBITDA]

    Notes
    -----
    This is a compact summary view for presentations. It is not a full P&L model.
    """
    required = {"price", "volume", "cogs_pct", "opex"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise KeyError(f"df is missing columns: {missing}")

    # Base values
    base_rev = (df["price"] * df["volume"]).sum()
    base_cogs = (df["cogs_pct"] * df["price"] * df["volume"]).sum()
    base_opex = df["opex"].sum()
    base_ebitda = base_rev - base_cogs - base_opex

    # Driver deltas
    price_d = float(drivers.get("price", 0.0))
    vol_d = float(drivers.get("volume", 0.0))
    mix_d = float(drivers.get("mix", 0.0))
    cogs_pct_d = float(drivers.get("cogs_pct", 0.0))
    opex_d = float(drivers.get("opex", 0.0))

    rev_delta = base_rev * ((1.0 + price_d) * (1.0 + vol_d) * (1.0 + mix_d) - 1.0)
    cogs_delta = -base_cogs * cogs_pct_d         # reduce COGS% improves EBITDA
    opex_delta = -base_opex * opex_d             # reduce opex improves EBITDA
    new_ebitda = base_ebitda + rev_delta + cogs_delta + opex_delta

    rows = [
        {"step": "Base EBITDA", "value": float(base_ebitda)},
        {"step": "Price/Vol/Mix", "value": float(rev_delta)},
        {"step": "COGS%", "value": float(cogs_delta)},
        {"step": "Opex", "value": float(opex_delta)},
        {"step": "New EBITDA", "value": float(new_ebitda)},
    ]
    return pd.DataFrame(rows)