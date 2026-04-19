import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load & prepare data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ft = pd.read_csv('data/foot_traffic.csv', parse_dates=['date'])
    ra = pd.read_csv('data/reported_actuals.csv', parse_dates=['earnings_date'])

    def assign_fiscal_quarter(date):
        month, year = date.month, date.year
        if month in [2, 3, 4]: return f'{year}-Q1'
        elif month in [5, 6, 7]: return f'{year}-Q2'
        elif month in [8, 9, 10]: return f'{year}-Q3'
        else: return f'{year - 1}-Q4' if month == 1 else f'{year}-Q4'

    ft['fiscal_quarter'] = ft['date'].apply(assign_fiscal_quarter)

    quarterly = ft.groupby(['ticker', 'fiscal_quarter']).agg(
        total_traffic=('foot_traffic', 'sum'),
        avg_daily_traffic=('foot_traffic', 'mean'),
        peak_traffic=('foot_traffic', 'max')
    ).reset_index()

    quarterly = quarterly.sort_values(['ticker', 'fiscal_quarter'])
    quarterly['traffic_qoq_growth'] = (
        quarterly.groupby('ticker')['total_traffic'].pct_change() * 100
    )
    quarterly['traffic_normalized'] = quarterly.groupby('ticker')['total_traffic'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    model_df = quarterly.merge(
        ra, left_on=['ticker', 'fiscal_quarter'], right_on=['ticker', 'quarter'], how='inner'
    ).drop(columns=['quarter'])

    return model_df.dropna(subset=['traffic_qoq_growth'])
# builds a separate linear regression model for each ticker
# returns all predictions and a summary of model accuracy

def build_predictions(model_df):
    features = ['traffic_normalized', 'traffic_qoq_growth']

    # store results as looping thru tickers
    all_preds = []
    summary = []

    # build a model for each ticker
    for ticker in model_df['ticker'].unique():
        df_t = model_df[model_df['ticker'] == ticker].copy()

        # input (traffic features)
        X = df_t[features]

        # one model to predict SSS% and one to predict revenue
        sss_model = LinearRegression().fit(X, df_t['reported_sss_pct'])
        rev_model = LinearRegression().fit(X, df_t['reported_revenue_mm'])

        # store the predictions back into the dataframe
        df_t['pred_sss'] = sss_model.predict(X)
        df_t['pred_revenue'] = rev_model.predict(X)
        all_preds.append(df_t)

        # calculate and store accuracy metrics for ticker
        summary.append({
            'ticker': ticker,
            # MAE = average prediction error
            'sss_mae': round(mean_absolute_error(df_t['reported_sss_pct'], df_t['pred_sss']), 3),
            # R2 = how much variation the model explains (1.0 = perfect)
            'sss_r2': round(r2_score(df_t['reported_sss_pct'], df_t['pred_sss']), 3),
            'rev_mae': round(mean_absolute_error(df_t['reported_revenue_mm'], df_t['pred_revenue']), 3),
            'rev_r2': round(r2_score(df_t['reported_revenue_mm'], df_t['pred_revenue']), 3),
        })

    # combine all ticker dataframes into one and return alongside the summary
    return pd.concat(all_preds).reset_index(drop=True), pd.DataFrame(summary)

# call the function and store the results
preds_df, summary_df = build_predictions(model_df)

# sidebar dropdown lets the user pick any of the 20 tickers
ticker = st.sidebar.selectbox('Select a Ticker', sorted(preds_df['ticker'].unique()))

# filter predictions to just the selected ticker and sort by quarter
df_t = preds_df[preds_df['ticker'] == ticker].sort_values('fiscal_quarter')

# ── Section 1: Forecast vs Actuals ───────────────────────────────────────────

st.subheader(f'{ticker} — Forecast vs Actuals')

# radio button lets user toggle between the two KPIs
kpi = st.radio('Select KPI', ['SSS%', 'Revenue ($M)'], horizontal=True)

# creating the chart
fig, ax = plt.subplots(figsize=(10, 4))

if kpi == 'SSS%':
    # plot actual and predicted SSS% over time
    ax.plot(df_t['fiscal_quarter'], df_t['reported_sss_pct'], marker='o', label='Actual')
    ax.plot(df_t['fiscal_quarter'], df_t['pred_sss'], marker='o', linestyle='--', label='Predicted')
    ax.set_ylabel('SSS%')
else:
    # plot actual and predicted revenue over time
    ax.plot(df_t['fiscal_quarter'], df_t['reported_revenue_mm'], marker='o', label='Actual')
    ax.plot(df_t['fiscal_quarter'], df_t['pred_revenue'], marker='o', linestyle='--', label='Predicted')
    ax.set_ylabel('Revenue ($M)')

ax.tick_params(axis='x', rotation=45)
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

# render the chart in the app
st.pyplot(fig)

# ── Section 2: Forecast Quality Summary ──────────────────────────────────────

st.subheader('Forecast Quality — All Tickers')
st.caption('R2 closer to 1.0 = stronger signal. MAE = average prediction error.')

# display the summary table sorted by revenue R2 (strongest signal first)
st.dataframe(
    summary_df.sort_values('rev_r2', ascending=False).reset_index(drop=True),
    use_container_width=True
)

# ── Section 3: Bonus — Traffic vs Actuals Scatter ────────────────────────────

st.subheader(f'{ticker} — Traffic Volume vs Reported KPIs')
st.caption(
    'This view shows the raw relationship between quarterly foot traffic and financial performance '
    'for the selected ticker. A clear upward trend indicates foot traffic is a strong predictor '
    'for that company — useful for quickly evaluating data quality on a per-name basis.'
)

# create side by side scatter plots for SSS% and Revenue
fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

# traffic vs SSS%
axes[0].scatter(df_t['total_traffic'], df_t['reported_sss_pct'], color='steelblue')
axes[0].set_xlabel('Total Quarterly Traffic')
axes[0].set_ylabel('SSS%')
axes[0].set_title('Traffic vs SSS%')

# traffic vs Revenue
axes[1].scatter(df_t['total_traffic'], df_t['reported_revenue_mm'], color='darkorange')
axes[1].set_xlabel('Total Quarterly Traffic')
axes[1].set_ylabel('Revenue ($M)')
axes[1].set_title('Traffic vs Revenue')

# clean up chart borders
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
st.pyplot(fig2)