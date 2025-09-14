"""
value_ops_lab

A compact toolkit for value creation analytics:
- Working capital diagnostics (DSO, DPO, DIO, CCC)
- Risk-aware cash forecasting and buffer sizing
- Simple dynamic policy optimization examples
- Driver-based EBITDA bridge and scenario utilities
"""

from .synth import make_synthetic
from .ccc import dso, dpo, dio, ccc, cash_unlocked
from .diagnostics import waterfall_ccc_impacts
from .risk_models import (
    quantile_forecast,
    cvar_cash_buffer,
    empirical_left_var,
    empirical_left_cvar,
)

__all__ = [
    "make_synthetic",
    "dso", "dpo", "dio", "ccc", "cash_unlocked",
    "waterfall_ccc_impacts",
    "quantile_forecast",
    "cvar_cash_buffer",
    "empirical_left_var",
    "empirical_left_cvar",
]