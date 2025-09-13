"""
Streamlit App: Value-Ops Lab
----------------------------

Interactive demo for risk-aware FP&A and working-capital optimization.

Key modules shown:
- CCC diagnostics (DSO, DPO, DIO, CCC)
- Cash-unlock waterfall analysis
- CVaR-based buffer sizing under uncertainty (Clarabel preferred)

Run locally:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# If you didn't package with -e . and pyproject.toml, uncomment the shim:
# import sys
# from pathlib import Path
# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

from value_ops_lab.synth import make_synthetic
from value_ops_lab.ccc import dso, dpo, dio, ccc, cash_unlocked
from value_ops_lab.diagnostics import waterfall_ccc_impacts
from value_ops_lab.risk_models import cvar_cash_buffer

# --- Streamlit Page Config ---
st.set_page_config(page_title="Value-Ops Lab", layout="wide")
st.title("📊 Alejandro’s Value-Ops Lab")
st.markdown(
    """
    This demo illustrates **how consulting analytics can unlock cash** and improve 
    decision-making under uncertainty.

    👉 Metrics shown here are *synthetic* data, but the methods are client-ready.
    """
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Data & What-If Controls")
    n_months = st.slider("Number of months", 12, 60, 24, 1)

    st.divider()
    st.subheader("Working Capital Targets (days)")
    dso_shift = st.slider("Target DSO improvement", 0, 30, 5, 1)
    dio_shift = st.slider("Target DIO improvement", 0, 30, 3, 1)
    dpo_shift = st.slider("Target DPO extension", 0, 30, 4, 1)

    st.divider()
    st.subheader("Risk Settings (CVaR)")
    alpha = st.slider("CVaR confidence (α)", 0.80, 0.99, 0.95, 0.01)

    st.caption("Scenario generator (cashflow shocks)")
    scenario_mode = st.radio(
        "Scenario set",
        (
            "Baseline (no shock)",
            "Mild downside",
            "Moderate downside",
            "Severe downside",
            "Custom",
        ),
        index=1,
    )
    vol = st.slider("Volatility (std dev)", 1000, 30000, 8000, 500)
    if scenario_mode == "Custom":
        shock = st.slider("Downside shock (avg)", -80000, 20000, -20000, 1000)
    else:
        shock_map = {
            "Baseline (no shock)": 0,
            "Mild downside": -10000,
            "Moderate downside": -20000,
            "Severe downside": -40000,
        }
        shock = shock_map[scenario_mode]
    rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

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
st.markdown("CCC combines **DSO, DIO, DPO** to measure liquidity efficiency.")
st.dataframe(ccc_df.tail(6), use_container_width=True)
st.line_chart(ccc_df.set_index("month")[["DSO", "DPO", "DIO", "CCC"]])

# --- Section 2: Cash-Unlock Waterfall ---
st.subheader("💡 What-if Analysis: Cash Unlock")
baseline = ccc_df.iloc[-1]
steps = waterfall_ccc_impacts(
    baseline_row=baseline,
    shifts={"DSO": -dso_shift, "DIO": -dio_shift, "DPO": +dpo_shift},
)
st.dataframe(steps, use_container_width=True)

daily_sales = sales_df["sales"].iloc[-1] / 30.0
current_ccc = float(baseline["CCC"])
target_ccc = float(steps["new_CCC"].iloc[-1])
unlocked = cash_unlocked(current_ccc, target_ccc, daily_sales)
st.metric("Estimated cash unlocked", f"${unlocked:,.0f}")

# --- Section 3: Risk-Aware Buffer (CVaR) ---
st.subheader("⚖️ Risk-aware Cash Buffer (CVaR)")
st.markdown(
    "Using a **Conditional Value-at-Risk (CVaR)** approach, we estimate the "
    "buffer needed to protect against downside scenarios."
)

# Build cashflow scenarios from recent months + shock/volatility
rng = np.random.default_rng(int(rng_seed))
base_cf = (sales_df["sales"].tail(6).to_numpy() - cogs_df["cogs"].tail(6).to_numpy())
scenarios = base_cf + rng.normal(shock, vol, size=base_cf.shape[0])

# Show scenarios so stakeholders see the stress being applied
scen_df = pd.DataFrame({"Month": sales_df["month"].tail(6).dt.strftime("%Y-%m"), "Scenario CF": scenarios})
scen_df_disp = scen_df.copy()
st.caption("Simulated end-of-period cashflow scenarios (last 6 months basis):")
st.dataframe(scen_df_disp, use_container_width=True)
st.bar_chart(scen_df.set_index("Month")["Scenario CF"])

# Solve for buffer
try:
    buffer, t = cvar_cash_buffer(scenarios, alpha=alpha)
    st.success(f"Optimal buffer at α={alpha:.2f}: **${buffer:,.0f}**")
except ImportError:
    st.warning("CVaR solver not available. Ensure cvxpy (and clarabel) is installed.")
except Exception as e:
    st.error(f"CVaR optimization failed: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Developed by Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate 2026")