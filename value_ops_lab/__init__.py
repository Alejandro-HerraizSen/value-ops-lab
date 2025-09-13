"""
value_ops_lab

A compact toolkit for value creation analytics:
- Working capital diagnostics (DSO, DPO, DIO, CCC)
- Risk-aware cash forecasting and buffer sizing
- Simple dynamic policy optimization examples
- Driver-based EBITDA bridge and scenario utilities
"""

from .ccc import dso, dpo, dio, ccc, cash_unlocked
from .risk_models import quantile_forecast, cvar_cash_buffer
from .diagnostics import waterfall_ccc_impacts, flag_invoice_anomalies
from .dp_policies import dp_pay_terms
from .scenario_engine import ebitda_bridge
from .synth import make_synthetic

__all__ = [
    "dso",
    "dpo",
    "dio",
    "ccc",
    "cash_unlocked",
    "quantile_forecast",
    "cvar_cash_buffer",
    "waterfall_ccc_impacts",
    "flag_invoice_anomalies",
    "dp_pay_terms",
    "ebitda_bridge",
    "make_synthetic",
]