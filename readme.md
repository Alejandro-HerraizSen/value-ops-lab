# Value-Ops Lab

Risk-aware FP&A and Working-Capital Optimizer for consulting and PE contexts.

## Why it matters
Accordion and similar firms thrive on delivering **cash unlock**, **risk-aware decisioning**, and **value creation insights** for CFOs, portfolio companies, and PE sponsors.  
This project brings together:
- **CCC Diagnostics**: DSO, DPO, DIO, CCC calculation with what-if waterfall impacts.
- **Risk-aware Forecasting**: CVaR-based cash buffer sizing under uncertainty.
- **Dynamic Policy Optimizer**: DP-style decision support for vendor/customer payment terms.
- **PE Extras**: Driver-based EBITDA bridge and LBO sensitivities.

## Quickstart
Clone the repo and install requirements:
```bash
git clone https://github.com/<your-username>/value-ops-lab.git
cd value-ops-lab
pip install -r requirements.txt
```

## Run the Streamlit app:
streamlit run app/streamlit_app.py

## Roadmap
1) Expand dynamic programming to multi-obligation, multi-period state spaces.
2) Add GL anomaly detection (Isolation Forest).
3) Integrate dbt + DuckDB pipeline for realistic finance data.
4) Package and publish to PyPI as value-ops-lab.

## Author
Alejandro Herraiz Sen — Penn State (Math & Data Science), CFA L1 Candidate (Feb 2026).
