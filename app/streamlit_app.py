"""
Streamlit App: Value-Ops Lab
----------------------------

Interactive demo for risk-aware FP&A and working-capital optimization.

Key modules shown:
- CCC diagnostics (DSO, DPO, DIO, CCC)
- Cash-unlock waterfall analysis
- CVaR-based buffer sizing under uncertainty (Clarabel preferred; LP fallback)

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
from value_ops_lab.risk_models import (
    cvar_cash_buffer,
    empirical_left_var,
    empirical_left_cvar,
)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Value-Ops Lab", layout="wide")
st.title(" Alejandro’s Value-Ops Lab")
st.markdown(
    """
    This demo illustrates **how consulting analytics can unlock cash** and improve 
    decision-making under uncertainty.

     Metrics shown here are *synthetic* data, but the methods are client-ready.
    """
)

# --- Sidebar Controls ---
# --- Sidebar Controls (clean, grouped) ---
with st.sidebar:
    st.markdown("### Data")
    n_months = st.slider("History window (months)", 12, 60, 24, 1, help="How many months of synthetic history to generate.")

    st.markdown("---")
    st.markdown("### Scenario Set")

    # Attractive labels with emojis (more “consulting demo” vibe)
    scenario_labels = {
        "Baseline": "Baseline (no shock)",
        "Mild":     "Mild downside",
        "Moderate": "Moderate downside",
        "Severe":   "Severe downside",
        "Custom":   "Custom",
    }
    scenario_keys = list(scenario_labels.keys())
    scenario_display = [scenario_labels[k] for k in scenario_keys]

    sel = st.radio(
        "Choose a scenario:",
        scenario_display,
        index=1,
        label_visibility="collapsed",
        help="Pick a stress profile; then tweak details below.",
    )
    # Map back to a key like "Mild", "Severe", etc.
    scenario_mode = scenario_keys[scenario_display.index(sel)]

    # All tuners beneath the scenario set (cleaner!)
    with st.expander("Tune scenario parameters", expanded=True):
        # Risk (alpha) as select slider for cleaner ticks
        alpha = st.select_slider(
            "CVaR confidence (α)",
            options=[0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
            value=0.95,
            help="Higher α = more conservative buffer (protects deeper tail).",
        )

        # Volatility slider always visible; shock only if custom
        vol = st.slider("Volatility σ (std dev of shocks, $)", 1_000, 30_000, 8_000, 500)
        if scenario_mode == "Custom":
            shock = st.slider("Average downside shock μ ($)", -80_000, 20_000, -20_000, 1_000)
        else:
            shock_map = {
                "Baseline": 0,
                "Mild": -10_000,
                "Moderate": -20_000,
                "Severe": -40_000,
            }
            shock = shock_map[scenario_mode]

        rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

    st.markdown("---")
    with st.expander("Operational levers (days)", expanded=False):
        dso_shift = st.slider("↓ DSO improvement", 0, 30, 5, 1)
        dio_shift = st.slider("↓ DIO improvement", 0, 30, 3, 1)
        dpo_shift = st.slider("↑ DPO extension", 0, 30, 4, 1)

    st.markdown("---")
    with st.expander("Solver & runtime", expanded=False):
        solver_mode = st.selectbox(
            "CVaR solver",
            ["Auto (prefer CVXPY, fallback SciPy)", "Force CVXPY", "Force SciPy"],
            index=0,
        )
        prefer_map = {
            "Auto (prefer CVXPY, fallback SciPy)": "auto",
            "Force CVXPY": "cvxpy",
            "Force SciPy": "scipy",
        }
        prefer = prefer_map[solver_mode]

# --- Generate Synthetic Data ---
df = make_synthetic(n_months)
sales_df = df[["month", "sales"]].copy()
cogs_df = df[["month", "cogs"]].copy()

# Ensure datetime
for _df in (df, sales_df, cogs_df):
    if not np.issubdtype(_df["month"].dtype, np.datetime64):
        _df["month"] = pd.to_datetime(_df["month"])

# Compute CCC parts
dso_df = dso(df[["month", "ar_balance"]], sales_df)
dpo_df = dpo(df[["month", "ap_balance"]], cogs_df)
dio_df = dio(df[["month", "inventory"]], cogs_df)
ccc_df = ccc(dso_df, dpo_df, dio_df)

# --- Section 1: CCC Metrics ---
st.subheader(" Cash Conversion Cycle (CCC) Metrics")
st.markdown("CCC combines **DSO, DIO, DPO** to measure liquidity efficiency.")
st.dataframe(ccc_df.tail(6), use_container_width=True)
st.line_chart(ccc_df.set_index("month")[["DSO", "DPO", "DIO", "CCC"]])

# --- Section 2: Cash-Unlock Waterfall ---
st.subheader(" What-if Analysis: Cash Unlock")
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

# Waterfall bar (ΔCCC → cash)
labels = steps["lever"].tolist()
ccc_baseline = float(ccc_df.iloc[-1]["CCC"])
ccc_path = np.r_[ccc_baseline, steps["new_CCC"].to_numpy()]
delta_ccc_per_step = np.diff(ccc_path)           # signed
cash_impacts = -delta_ccc_per_step * daily_sales # negative ΔCCC => positive cash

running = np.cumsum(cash_impacts)
base = np.r_[0, running[:-1]]

import matplotlib.pyplot as plt
plt.figure(figsize=(9, 4))
for i, (b, h) in enumerate(zip(base, cash_impacts)):
    color = "#2ca02c" if h >= 0 else "#d62728"
    plt.bar(i, h, bottom=b, width=0.6, color=color)
plt.xticks(range(len(labels)), labels, rotation=0)
plt.title("Cash Unlock Waterfall (ΔCCC × Daily Sales)")
plt.ylabel("Cash Impact ($)")
plt.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# --- Section 3: Risk-Aware Buffer (CVaR) ---
st.subheader(" Risk-aware Cash Buffer (CVaR)")
st.markdown(
    "Using a **Conditional Value-at-Risk (CVaR)** approach, we estimate the "
    "buffer needed to protect against downside scenarios."
)

# Build cashflow scenarios from recent months + shock/volatility
rng = np.random.default_rng(int(rng_seed))
base_cf = (sales_df["sales"].tail(6).to_numpy() - cogs_df["cogs"].tail(6).to_numpy())
scenarios = base_cf + rng.normal(shock, vol, size=base_cf.shape[0])

# Show scenarios so stakeholders see the stress being applied
scen_df = pd.DataFrame(
    {"Month": sales_df["month"].tail(6).dt.strftime("%Y-%m"), "Scenario CF": scenarios}
)
st.caption("Simulated end-of-period cashflow scenarios (last 6 months basis):")
st.dataframe(scen_df, use_container_width=True)

# Bar chart
st.bar_chart(scen_df.set_index("Month")["Scenario CF"])

# Quick diagnostic stats (helps validate that downside exists)
p_left = np.quantile(scenarios, 1 - alpha)
st.caption(
    f"Scenario stats — min: {scenarios.min():,.0f}, "
    f"p{int(100*(1-alpha))}: {p_left:,.0f}, "
    f"mean: {scenarios.mean():,.0f}, max: {scenarios.max():,.0f}"
)

# --- Empirical buffers (no solver, for validation & display) ---
left_var = empirical_left_var(scenarios, alpha)             # could be negative
left_cvar = empirical_left_cvar(scenarios, alpha)           # average of worst (1-α) tail
buffer_empirical_var = max(0.0, -left_var)                  # b s.t. VaR_{α}(s + b) >= 0
buffer_empirical_cvar = max(0.0, -left_cvar)                # b s.t. CVaR_{α}(s + b) >= 0

cols = st.columns(2)
cols[0].metric("Empirical VaR buffer", f"${buffer_empirical_var:,.0f}")
cols[1].metric("Empirical CVaR buffer", f"${buffer_empirical_cvar:,.0f}")

# --- Optimized buffer (robust solver with degeneracy guard) ---
try:
    buffer_opt, t_opt, solver_used = cvar_cash_buffer(
        scenarios, alpha=alpha, prefer=prefer, return_solver=True
    )
    st.success(
        f"Optimized buffer at α={alpha:.2f}: **${buffer_opt:,.0f}** · "
        f"Solver: {solver_used} · t={t_opt:,.0f}"
    )
except ImportError:
    st.warning("Solver not available. Ensure either cvxpy (with clarabel) or scipy is installed.")
    buffer_opt = None
except Exception as e:
    st.error(f"CVaR optimization failed: {e}")
    buffer_opt = None

# --- Recommendation (choose the safer of empirical vs optimized) ---
recommended = max(
    [b for b in [buffer_empirical_cvar, buffer_empirical_var, (buffer_opt or 0.0)]]
)
st.markdown(
    f"**Recommended buffer (conservative):** ${recommended:,.0f} "
    f"(max of empirical CVaR/VaR and optimized result)"
)

# --- Sensitivity: Buffer vs α ---
st.subheader("Sensitivity: Buffer vs α")
st.caption("How reserve sizing changes as risk tolerance tightens (α → 1).")

alphas = np.linspace(0.80, 0.99, 20)
buf_var, buf_cvar, buf_opt = [], [], []

for a in alphas:
    ev = empirical_left_var(scenarios, a)
    ec = empirical_left_cvar(scenarios, a)
    buf_var.append(max(0.0, -ev))
    buf_cvar.append(max(0.0, -ec))
    try:
        b_opt, _t, _solver = cvar_cash_buffer(scenarios, alpha=a, prefer=prefer, return_solver=True)
        buf_opt.append(max(0.0, float(b_opt)))
    except Exception:
        buf_opt.append(np.nan)

sens_df = pd.DataFrame(
    {
        "alpha": alphas,
        "Empirical VaR buffer": buf_var,
        "Empirical CVAР buffer": buf_cvar,
        "Optimized buffer": buf_opt,
    }
).set_index("alpha")

st.line_chart(sens_df)

st.caption(
    "Empirical lines come from sample quantiles/means; the optimized line solves the CVaR program "
    "with degeneracy guard (Auto = CVXPY→SciPy fallback). Curves should slope upward as α increases "
    "and/or scenarios worsen."
)

# --- Footer ---
st.markdown("---")
st.caption("Developed by Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate 2026")