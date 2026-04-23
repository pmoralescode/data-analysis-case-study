# Interval Partners — Foot Traffic Case Study

## Overview
This project evaluates whether daily foot traffic data can be used to forecast quarterly KPIs — same-store sales growth (SSS%) and revenue ($M) — for 20 publicly traded retail companies before official earnings are reported.

The analysis covers 3 years of data (Feb 2022 – Jan 2025) across 20 tickers and 12 fiscal quarters.

---

## Project Structure
Interval/
├── data/
│   ├── foot_traffic.csv        # Daily foot traffic by ticker and state
│   └── reported_actuals.csv    # Quarterly KPIs for 20 tickers
├── analysis/
│   └── analysis.ipynb          # Full analysis notebook
├── app/
│   └── app.py                  # Streamlit web application
└── README.md

---

## Part 1: Data Analysis

### Setup
1. Clone the repo
2. Create and activate a virtual environment:
```bash
python -m venv proj
source proj/bin/activate
```
3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
4. Launch Jupyter:
```bash
jupyter lab
```
5. Open `analysis/analysis.ipynb` and run all cells top to bottom

### Approach
- Assigned each day to a fiscal quarter (Q1=Feb–Apr, Q2=May–Jul, Q3=Aug–Oct, Q4=Nov–Jan)
- Aggregated daily traffic into quarterly metrics: total traffic, average daily traffic, peak traffic, and quarter-over-quarter growth
- Normalized traffic within each ticker to remove company size bias
- Built per-ticker linear regression models to predict SSS% and revenue

### Key Findings
- Foot traffic is a **strong predictor of revenue** (avg R2 > 0.90 across tickers)
- Foot traffic is a **weak predictor of SSS%** (best R2 = 0.44 for KSS)
- Strongest signals: DLTR, ANF, ACI for revenue; KSS, JWN, WMT for SSS%
- Weakest signal: KR (Kroger) — grocery purchases driven by necessity, not discretionary visits

---

## Part 2: Web Application

### Setup
1. Activate your virtual environment:
```bash
source proj/bin/activate
```
2. Install Streamlit:
```bash
pip install streamlit
```
3. From the root `Interval` folder, run:
```bash
streamlit run app/app.py
```
4. The app will open automatically in your browser

### Features
- **Ticker selector** — choose any of the 20 retail tickers from the sidebar
- **Forecast vs Actuals chart** — compare traffic-derived predictions against reported SSS% and revenue over time, toggled by KPI
- **Forecast quality summary** — table showing MAE and R2 for all 20 tickers at a glance
- **Bonus — Traffic vs KPI scatter** — shows the raw relationship between quarterly foot traffic and financial performance for the selected ticker. This view helps a portfolio manager quickly assess whether foot traffic is a reliable signal for a specific name before relying on it for investment decisions

