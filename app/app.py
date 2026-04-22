import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load & prepare data ──────────────────────────────────────────────────────

# sets the browser tab title, icon, and wide layout
st.set_page_config(
    page_title='Foot Traffic Signal Dashboard',
    page_icon='📈',
    layout='wide'
)


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

model_df = load_data()
# builds a separate linear regression model for each ticker
# returns all predictions and a summary of model accuracy

st.title('📈 Foot Traffic Signal Dashboard')
st.caption('Interval Partners LP — Data Analysis')
st.divider()

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

# metric cards show key stats for the selected ticker at a glance
ticker_summary = summary_df[summary_df['ticker'] == ticker].iloc[0]

st.subheader('Key Stats')

col1, col2, col3, col4 = st.columns(4)
col1.metric('Ticker', ticker)
col2.metric('SSS% R2', ticker_summary['sss_r2'])
col3.metric('Revenue R2', ticker_summary['rev_r2'])
col4.metric('SSS% MAE', f"{ticker_summary['sss_mae']}%")

st.divider()

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

st.divider()

# ── Section 2: Forecast Quality Summary ──────────────────────────────────────

st.subheader('Forecast Quality — All Tickers')
st.caption('R2 closer to 1.0 = stronger signal. MAE = average prediction error.')

# display the summary table sorted by revenue R2 (strongest signal first)
st.dataframe(
    summary_df.sort_values('rev_r2', ascending=False).reset_index(drop=True),
    use_container_width=True
)

st.divider()

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

st.divider()

# ── Section 4: Bonus — State-Level Traffic Heatmap ───────────────────────────

st.subheader(f'{ticker} — State-Level Traffic Heatmap')
st.caption(
    'This heatmap shows which states drive the most foot traffic for the selected ticker across all quarters. '
    'It helps a portfolio manager assess whether the signal is geographically concentrated or broadly distributed '
    'before relying on it for investment decisions.'
)

# load raw foot traffic data to get state level breakdown
@st.cache_data
def load_raw():
    ft_raw = pd.read_csv('data/foot_traffic.csv', parse_dates=['date'])
    def assign_fiscal_quarter(date):
        month, year = date.month, date.year
        if month in [2, 3, 4]: return f'{year}-Q1'
        elif month in [5, 6, 7]: return f'{year}-Q2'
        elif month in [8, 9, 10]: return f'{year}-Q3'
        else: return f'{year - 1}-Q4' if month == 1 else f'{year}-Q4'
    ft_raw['fiscal_quarter'] = ft_raw['date'].apply(assign_fiscal_quarter)
    return ft_raw

ft_raw = load_raw()

# filter to selected ticker and aggregate by state and quarter
state_data = ft_raw[ft_raw['ticker'] == ticker].groupby(
    ['state', 'fiscal_quarter']
)['foot_traffic'].sum().reset_index()

# pivot to create a matrix of states vs quarters
heatmap_data = state_data.pivot(index='state', columns='fiscal_quarter', values='foot_traffic')

# plot the heatmap
fig3, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(
    heatmap_data,
    ax=ax,
    cmap='YlOrRd',
    fmt='.0f',
    linewidths=0.5,
    cbar_kws={'label': 'Total Foot Traffic'}
)
ax.set_title(f'{ticker} — Foot Traffic by State and Quarter')
ax.set_xlabel('Fiscal Quarter')
ax.set_ylabel('State')
plt.tight_layout()
st.pyplot(fig3)