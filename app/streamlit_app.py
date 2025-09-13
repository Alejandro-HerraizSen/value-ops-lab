"""
Streamlit App: Value-Ops Lab
----------------------------

Interactive demo for risk-aware FP&A and working-capital optimization.

Key modules shown:
- CCC diagnostics (DSO, DPO, DIO, CCC)
- Cash-unlock waterfall analysis
- CVaR-based buffer sizing under uncertainty

Audience:
- CFOs / finance managers
- Consultants at Accordion or PE operating teams

Run locally:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import from our library
from value_ops_lab.synth import make_synthetic
from value_ops_lab.ccc import dso, dpo, dio, ccc, cash_unlocked
from value_ops_lab.diagnostics import waterfall_ccc_impacts
from value_ops_lab.risk_models import cvar_cash_buffer


# --- Streamlit Page Config ---
st.set_page_config(page_title="Value-Ops Lab", layout="wide")
st.title("📊 Value-Ops Lab: Risk-aware FP&A and Working-Capital Optimizer")
st.markdown(
    """
    This demo illustrates **how consulting analytics can unlock cash** and improve 
    decision-making under uncertainty.

    👉 Metrics shown here are *synthetic* data, but the methods are client-ready.
    """
)


# --- Sidebar Controls ---
with st.sidebar:
    st.header("Simulation Controls")
    n_months = st.slider("Number of months", 12, 60, 24, 1)
    dso_shift = st.slider("Target DSO improvement (days)", 0, 20, 5)
    dio_shift = st.slider("Target DIO improvement (days)", 0, 20, 3)
    dpo_shift = st.slider("Target DPO extension (days)", 0, 20, 4)
    alpha = st.slider("CVaR confidence level (alpha)", 0.80, 0.99, 0.95, 0.01)


# --- Generate Synthetic Data ---
df = make_synthetic(n_months)

sales_df = df[["month", "sales"]]
cogs_df = df[["month", "cogs"]]

dso_df = dso(df[["month", "ar_balance"]], sales_df)
dpo_df = dpo(df[["month", "ap_balance"]], cogs_df)
dio_df = dio(df[["month", "inventory"]], cogs_df)

ccc_df = ccc(dso_df, dpo_df, dio_df)


# --- Section 1: CCC Metrics ---
st.subheader("📌 Cash Conversion Cycle (CCC) Metrics")
st.markdown("The CCC combines **DSO, DIO, and DPO** to measure liquidity efficiency.")

st.dataframe(ccc_df.tail(6))

st.line_chart(ccc_df.set_index("month")[["DSO", "DPO", "DIO", "CCC"]])


# --- Section 2: Cash-Unlock Waterfall ---
st.subheader("💡 What-if Analysis: Cash Unlock")
baseline = ccc_df.iloc[-1]

steps = waterfall_ccc_impacts(
    baseline_row=baseline,
    shifts={"DSO": -dso_shift, "DIO": -dio_shift, "DPO": +dpo_shift},
)
st.dataframe(steps)

daily_sales = sales_df["sales"].iloc[-1] / 30.0
current_ccc = float(baseline["CCC"])
target_ccc = float(steps["new_CCC"].iloc[-1])
unlocked = cash_unlocked(current_ccc, target_ccc, daily_sales)

st.metric("Estimated cash unlocked", f"${unlocked:,.0f}")


# --- Section 3: Risk-Aware Buffer ---
st.subheader("⚖️ Risk-aware Cash Buffer (CVaR)")
st.markdown(
    """
    Using a **Conditional Value-at-Risk (CVaR)** approach, we estimate 
    the buffer needed to protect against downside scenarios.
    """
)

# Create toy scenarios: last 6 months of net CF with random noise
scenarios = (
    (sales_df["sales"].tail(6).values - cogs_df["cogs"].tail(6).values)
    + np.random.normal(0, 5000, 6)
)

try:
    buffer, t = cvar_cash_buffer(scenarios, alpha=alpha)
    st.write(f"Optimal buffer at α={alpha:.2f}: **${buffer:,.0f}**")
except ImportError:
    st.warning(
        "cvxpy not installed — skipping CVaR optimization. "
        "Install cvxpy to enable this feature."
    )


# --- Footer ---
st.markdown("---")
st.caption("Developed by Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate 2026")