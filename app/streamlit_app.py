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
import matplotlib.pyplot as plt

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

# --- Page Config & Title ---
st.set_page_config(page_title="Value-Ops Lab", layout="wide")
st.title("Alejandro’s Value-Ops Lab")
st.caption("Penn State (Math & Data Science) • CFA L1 Candidate 2026")

st.markdown(
    """
**What this app demonstrates :**
1) Diagnose **working capital** efficiency with CCC (DSO, DIO, DPO).  
2) Quantify **cash unlocked** from realistic operational levers.  
3) Size a **risk-aware liquidity buffer** using **CVaR** under stressed scenarios.  

_All data are synthetic; the methodology is client-ready and maps cleanly to real ledgers._
"""
)

# ============================= Sidebar (clean & grouped) =============================

with st.sidebar:
    st.markdown("### 📅 Data window")
    n_months = st.slider("History (months)", 12, 60, 24, 1,
                         help="Number of synthetic months to generate.")

    st.markdown("---")
    st.markdown("### 🎯 Scenario set")

    scenario_labels = {
        "Baseline": "Baseline (no shock)",
        "Mild":     "Mild downside",
        "Moderate": "Moderate downside",
        "Severe":   "Severe downside",
        "Custom":   "Custom",
    }
    scenario_keys = list(scenario_labels.keys())
    scenario_display = [scenario_labels[k] for k in scenario_keys]

    sel = st.radio("Choose a scenario:", scenario_display, index=1, label_visibility="collapsed")
    scenario_mode = scenario_keys[scenario_display.index(sel)]

    with st.expander("Tune scenario parameters", expanded=True):
        alpha = st.select_slider(
            "CVaR confidence (α)",
            options=[0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99],
            value=0.95,
            help="Higher α = deeper tail protection → larger buffer."
        )

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

# ============================= Data generation =============================

df = make_synthetic(n_months)
sales_df = df[["month", "sales"]].copy()
cogs_df  = df[["month", "cogs"]].copy()

for _df in (df, sales_df, cogs_df):
    if not np.issubdtype(_df["month"].dtype, np.datetime64):
        _df["month"] = pd.to_datetime(_df["month"])

# CCC metrics
dso_df = dso(df[["month", "ar_balance"]], sales_df)
dpo_df = dpo(df[["month", "ap_balance"]], cogs_df)
dio_df = dio(df[["month", "inventory"]], cogs_df)
ccc_df = ccc(dso_df, dpo_df, dio_df)

# Quick KPIs
latest = ccc_df.iloc[-1]
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("DSO (days)", f"{latest['DSO']:.1f}")
kpi2.metric("DIO (days)", f"{latest['DIO']:.1f}")
kpi3.metric("DPO (days)", f"{latest['DPO']:.1f}")

st.markdown("---")

# ============================= Main content tabs =============================

tab_ccc, tab_unlock, tab_cvar = st.tabs(
    ["CCC Diagnostics", "Cash Unlock (What-If)", "Risk-Aware Buffer (CVaR)"]
)

# ----------------------------- Tab 1: CCC -----------------------------------

with tab_ccc:
    st.subheader("How I diagnose working capital")
    st.markdown(
        """
**Consulting lens:** I start by decomposing CCC into **DSO (AR)**, **DIO (Inventory)** y **DPO (AP)**.  
Esto permite priorizar iniciativas: facturación/cobro, planificación de inventario y negociación con proveedores.
"""
    )

    st.dataframe(ccc_df.tail(6), use_container_width=True)
    st.line_chart(ccc_df.set_index("month")[["DSO", "DPO", "DIO", "CCC"]])

    with st.expander("How these metrics are computed"):
        st.markdown(
            """
- **DSO** ≈ (Cuentas por cobrar / Ventas) × 30  
- **DIO** ≈ (Inventario / COGS) × 30  
- **DPO** ≈ (Cuentas por pagar / COGS) × 30  
- **CCC** = DSO + DIO − DPO  
            """
        )

# ------------------------- Tab 2: Cash Unlock --------------------------------

with tab_unlock:
    st.subheader("What-if levers and cash unlock")
    st.markdown(
        """
I model **target shifts** in days (DSO, DIO, DPO) and translate the delta in CCC into **cash released** using current daily sales.
"""
    )

    baseline = ccc_df.iloc[-1]
    steps = waterfall_ccc_impacts(
        baseline_row=baseline,
        shifts={"DSO": -dso_shift, "DIO": -dio_shift, "DPO": +dpo_shift},
    )
    st.dataframe(steps, use_container_width=True)

    daily_sales = sales_df["sales"].iloc[-1] / 30.0
    current_ccc = float(baseline["CCC"])
    target_ccc  = float(steps["new_CCC"].iloc[-1])
    unlocked = cash_unlocked(current_ccc, target_ccc, daily_sales)

    st.metric("Estimated cash unlocked", f"${unlocked:,.0f}")

    # Waterfall
    labels = steps["lever"].tolist()
    ccc_baseline = float(ccc_df.iloc[-1]["CCC"])
    ccc_path = np.r_[ccc_baseline, steps["new_CCC"].to_numpy()]
    delta_ccc_per_step = np.diff(ccc_path)
    cash_impacts = -delta_ccc_per_step * daily_sales
    running = np.cumsum(cash_impacts)
    base = np.r_[0, running[:-1]]

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

    with st.expander("Why this matters / What I’d do next"):
        st.markdown(
            """
- **Why:** Cash released reduces reliance on short-term credit and creates room for growth/investment.  
- **Next:** Segment customers para **políticas de cobro** específicas, **SKU-level inventory** y **terms benchmarking** por proveedor.
"""
        )

# ------------------------- Tab 3: CVaR Buffer --------------------------------

with tab_cvar:
    st.subheader("Reserve sizing under uncertainty (CVaR)")
    st.markdown(
        """
I stress recent net cash flows with **mean shock (μ)** and **volatility (σ)**, then size a **buffer** such that the
**left-tail expected shortfall (CVaR)** is ≥ 0 at confidence **α**.
"""
    )

    # Build scenarios
    rng = np.random.default_rng(int(rng_seed))
    base_cf = (sales_df["sales"].tail(6).to_numpy() - cogs_df["cogs"].tail(6).to_numpy())
    scenarios = base_cf + rng.normal(shock, vol, size=base_cf.shape[0])

    scen_df = pd.DataFrame({"Month": sales_df["month"].tail(6).dt.strftime("%Y-%m"),
                            "Scenario CF": scenarios})
    st.dataframe(scen_df, use_container_width=True)
    st.bar_chart(scen_df.set_index("Month")["Scenario CF"])

    # Quick stats & empirical checks
    p_left = np.quantile(scenarios, 1 - alpha)
    st.caption(
        f"Scenario stats — min: {scenarios.min():,.0f}, "
        f"p{int(100*(1-alpha))}: {p_left:,.0f}, "
        f"mean: {scenarios.mean():,.0f}, max: {scenarios.max():,.0f}"
    )

    left_var  = empirical_left_var(scenarios, alpha)
    left_cvar = empirical_left_cvar(scenarios, alpha)
    buffer_empirical_var  = max(0.0, -left_var)
    buffer_empirical_cvar = max(0.0, -left_cvar)

    m1, m2 = st.columns(2)
    m1.metric("Empirical VaR buffer",  f"${buffer_empirical_var:,.0f}")
    m2.metric("Empirical CVaR buffer", f"${buffer_empirical_cvar:,.0f}")

    # Optimized buffer (robust solver + degeneracy guard inside cvar_cash_buffer)
    try:
        buffer_opt, t_opt, solver_used = cvar_cash_buffer(
            scenarios, alpha=alpha, prefer=prefer, return_solver=True
        )
        st.success(
            f"Optimized buffer at α={alpha:.2f}: **${buffer_opt:,.0f}** · "
            f"Solver: {solver_used} · t={t_opt:,.0f}"
        )
    except ImportError:
        st.warning("Solver not available. Install cvxpy (with clarabel) or scipy.")
        buffer_opt = None
    except Exception as e:
        st.error(f"CVaR optimization failed: {e}")
        buffer_opt = None

    # Conservative recommendation
    recommended = max(buffer_empirical_cvar, buffer_empirical_var, (buffer_opt or 0.0))
    st.markdown(
        f"**Recommended buffer (conservative):** ${recommended:,.0f} "
        f"(max of empirical CVaR/VaR and optimized)"
    )

    # Sensitivity vs alpha
    st.markdown("**Buffer sensitivity vs α**")
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
        {"alpha": alphas,
         "Empirical VaR buffer": buf_var,
         "Empirical CVAР buffer": buf_cvar,
         "Optimized buffer": buf_opt}
    ).set_index("alpha")
    st.line_chart(sens_df)

    with st.expander("How it’s computed (math sketch)"):
        st.markdown(
            r"""
**Program (Rockafellar–Uryasev, left tail):**

Minimize \( b + \varepsilon\big(t + \frac{1}{N}\sum z_i\big) \)  
subject to \( z_i \ge -(s_i + b) - t \), \( z_i \ge 0 \), \( b \ge 0 \),  
and \( t + \frac{1}{(1-\alpha)N}\sum z_i \le 0 \).

- \(s_i\): scenario cashflow (negative = shortfall)  
- \(b\): buffer to add  
- \(t\): VaR-like threshold  
- \(z_i\): hinge variables for the tail loss  
- \(\varepsilon\): tiny regularizer (avoid b–t degeneracy)
"""
        )

# --- Footer ---
st.markdown("---")
st.caption("This demo is based on synthetic ledgers; methods are production-ready for real data.")
st.caption("Developed by Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate 2026")