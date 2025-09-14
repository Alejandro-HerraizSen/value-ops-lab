#  Value-Ops Lab  
*A consulting analytics demo for unlocking cash and managing risk under uncertainty*  

**Link to live demo:** <https://value-ops-lab.streamlit.app>

---

##  Project Overview  
**Value-Ops Lab** is an end-to-end analytics project that shows how **data science + finance consulting methods** can be combined to:  
1. Diagnose **working capital efficiency** (Cash Conversion Cycle).  
2. Simulate and forecast **cash flows** with uncertainty.  
3. Size **liquidity buffers** using advanced **risk optimization (CVaR)**.  
4. Evaluate **private equity-style value creation** and return sensitivities.  

The project blends **synthetic financial data**, **Python modeling**, and a **Streamlit app** for interactive insights.  

---

## ⚡ Features  

### 🔄 1. Working Capital & CCC  
- Calculate **DSO, DIO, DPO, CCC** over time.  
- Visualize trends in liquidity efficiency.  
- Explore **what-if levers** (reduce DSO, negotiate DPO, optimize inventory).  

### 📊 2. Forecasting & Scenario Analysis  
- Quantile-based **cashflow forecasts** (p10–p90 bands).  
- Scenario shocks to sales/working capital.  
- Sensitivity of buffers to **confidence level (α)**.  

### 📉 3. Risk-aware Liquidity Buffer (CVaR)  
- Implements **Rockafellar–Uryasev CVaR optimization** in `cvxpy`.  
- Computes **empirical VaR & CVaR** vs **optimized buffers**.  
- Includes **ε-regularization + SciPy fallback** for solver robustness.  
- Produces **Buffer vs α curves** for risk tolerance exploration.  

### 📈 4. Private Equity Value Creation (Notebook 3)  
- **EBITDA bridge (waterfall)** to show operational uplift.  
- **Return modeling (MoIC, IRR)** under leverage & multiples.  
- **Heatmaps**: IRR vs Entry/Exit multiples & Leverage.  
- **Tornado sensitivity charts** for key drivers.  
- **Monte Carlo IRR distributions** with risk percentiles.  

---

##  How to run 
###  1. Clone repo & install dependencies  
```bash
git clone https://github.com/Alejandro-HerraizSen/value-ops-lab
cd value-ops-lab
pip install -r requirements.txt
```

###  2. Run Jupyter notebooks 
On each of the 3 notebooks

###  3. Launch Streamlit app
streamlit run app/streamlit_app.py 

## Tech Stack
- Python 3.12
- Pandas, NumPy, Matplotlib, Seaborn – data handling & plotting
- CVXPY + Clarabel/SciPy – risk optimization
- Streamlit – interactive app
- Jupyter – analysis notebooks

## Example Outputs
- CCC & Cash Unlock
- CVaR Buffer Optimization
- Private Equity Value Creation

## Key Insights
- Traditional working capital levers (DSO, DIO, DPO) can free up significant liquidity.
- Quantile forecasts highlight volatility in cash flow planning.
- Optimized CVaR buffers give a rigorous, risk-adjusted measure of liquidity needs.
- PE-style return analysis shows how entry/exit multiples and leverage drive MoIC & IRR, while Monte Carlo illustrates downside risk.

## Author
**Alejandro Herraiz Sen**
- Penn State (Math & Data Science, double major)
- CFA Level 1 Candidate (Feb 2026)
- Aspiring consultant in finance, analytics, and private equity

## Disclaimer
This project uses **synthetic data** and is for **educational/demo purposes** only.
Methods shown are **client-ready** but not based on real company data.