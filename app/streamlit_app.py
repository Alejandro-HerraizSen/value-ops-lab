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

# --- Optional solver detection (display only) ---
try:
    import cvxpy as _cp  # noqa: F401
    _HAS_CVX = True
except Exception:
    _HAS_CVX = False
try:
    from scipy.optimize import linprog as _lp  # noqa: F401
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ---------- Small helpers (empirical VaR / CVaR for sanity checks) ----------
def empirical_left_var(x: np.ndarray, alpha: float) -> float:
    """
    Left-tail VaR on a series x (downside). VaR_{alpha} (left) = quantile at (1 - alpha).
    """
    x = np.asarray(x, dtype=float).ravel()
    q = np.quantile(x, 1.0 - alpha, method="linear")
    return float(q)


def empirical_left_cvar(x: np.ndarray, alpha: float) -> float:
    """
    Left-tail CVaR on a series x (empirical average of worst (1-alpha) tail).
    If the tail set is empty (alpha=1), returns min(x).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    q = empirical_left_var(x, alpha)
    tail = x[x <= q]
    if tail.size == 0:
        tail = np.array([x.min()])
    return float(tail.mean())


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
    vol = st.slider("Volatility (std dev)", 1_000, 30_000, 8_000, 500)
    if scenario_mode == "Custom":
        shock = st.slider("Downside shock (avg)", -80_000, 20_000, -20_000, 1_000)
    else:
        shock_map = {
            "Baseline (no shock)": 0,
            "Mild downside": -10_000,
            "Moderate downside": -20_000,
            "Severe downside": -40_000,
        }
        shock = shock_map[scenario_mode]
    rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

# --- Generate Synthetic Data ---
df = make_synthetic(n_months)
sales_df = df[["month", "sales"]]
cogs_df = df[["month", "cogs"]]

# Ensure 'month' is datetime for plotting/formatting
if not np.issubdtype(sales_df["month"].dtype, np.datetime64):
    sales_df["month"] = pd.to_datetime(sales_df["month"])
    cogs_df["month"] = pd.to_datetime(cogs_df["month"])
    df["month"] = pd.to_datetime(df["month"])

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
scen_df = pd.DataFrame(
    {"Month": sales_df["month"].tail(6).dt.strftime("%Y-%m"), "Scenario CF": scenarios}
)
st.caption("Simulated end-of-period cashflow scenarios (last 6 months basis):")
st.dataframe(scen_df, use_container_width=True)
st.bar_chart(scen_df.set_index("Month")["Scenario CF"])

# Quick diagnostic stats (helps validate that downside exists)
st.caption(
    f"Scenario stats — min: {scenarios.min():,.0f}, "
    f"p{int(100*(1-alpha))}: {np.quantile(scenarios, 1-alpha):,.0f}, "
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

# --- Optimized buffer (solver-based; risk_models uses ε-regularized CVaR) ---
try:
    buffer_opt, t_opt = cvar_cash_buffer(scenarios, alpha=alpha)
    solver_label = "cvxpy (Clarabel/ECOS/SCS)" if _HAS_CVX else ("SciPy (HiGHS)" if _HAS_SCIPY else "Unknown")
    st.success(
        f"Optimized buffer at α={alpha:.2f}: **${buffer_opt:,.0f}** "
        f"· Solver: {solver_label} · t={t_opt:,.0f}"
    )
except ImportError:
    st.warning("CVaR solver not available. Ensure cvxpy (and clarabel) is installed.")
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
    # Empirical
    ev = empirical_left_var(scenarios, a)
    ec = empirical_left_cvar(scenarios, a)
    buf_var.append(max(0.0, -ev))
    buf_cvar.append(max(0.0, -ec))

    # Optimized (try; fallback to NaN)
    try:
        b_opt, _ = cvar_cash_buffer(scenarios, alpha=a)
        buf_opt.append(max(0.0, float(b_opt)))
    except Exception:
        buf_opt.append(np.nan)

sens_df = pd.DataFrame(
    {
        "alpha": alphas,
        "Empirical VaR buffer": buf_var,
        "Empirical CVaR buffer": buf_cvar,
        "Optimized buffer": buf_opt,
    }
).set_index("alpha")

st.line_chart(sens_df)
st.caption(
    "Empirical lines come from sample quantiles/means; the optimized line solves the CVaR program. "
    "Curves should slope upward as α increases and/or scenarios worsen."
)

# --- Footer ---
st.markdown("---")
st.caption("Developed by Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate 2026")