import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(
    page_title='Coffee Shop Sales Intelligence Dashboard',
    page_icon='☕',
    layout='wide'
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# ---------------------------------
# Load raw data
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "coffee_shop_sales.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["customer_age_group"] = df["customer_age_group"].fillna("Unknown")
    df["customer_gender"] = df["customer_gender"].fillna("Unknown")
    df["weather_condition"] = df["weather_condition"].fillna("Unknown")
    df["holiday_name"] = df["holiday_name"].fillna("No Holiday")

    df["temperature_c"] = df.groupby(["city", "weather_condition"])["temperature_c"] \
        .transform(lambda x: x.fillna(x.median()))
    df["temperature_c"] = df.groupby("city")["temperature_c"] \
        .transform(lambda x: x.fillna(x.median()))
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())

    df["is_holiday"] = (df["holiday_name"] != "No Holiday").astype(int)

    return df

# ---------------------------------
# Load artifacts
# ---------------------------------
@st.cache_resource
def load_artifacts():
    model_tsk1_opt = XGBRegressor()
    model_tsk1_opt.load_model(str(ARTIFACT_DIR / "model_tsk1_opt.json"))

    model_tsk2 = XGBRegressor()
    model_tsk2.load_model(str(ARTIFACT_DIR / "model_tsk2.json"))

    model_tsk3_opt = XGBRegressor()
    model_tsk3_opt.load_model(str(ARTIFACT_DIR / "model_tsk3_opt.json"))

    model_tsk4_opt = XGBRegressor()
    model_tsk4_opt.load_model(str(ARTIFACT_DIR / "model_tsk4_opt.json"))

    tsk1_opt_result = pd.read_csv(ARTIFACT_DIR / "tsk1_opt_result.csv")
    tsk2_result = pd.read_csv(ARTIFACT_DIR / "tsk2_result.csv")
    tsk3_opt_result = pd.read_csv(ARTIFACT_DIR / "tsk3_opt_result.csv")
    tsk4_opt_result = pd.read_csv(ARTIFACT_DIR / "tsk4_opt_result.csv")
    model_summary_opt = pd.read_csv(ARTIFACT_DIR / "model_summary_opt.csv")

    with open(ARTIFACT_DIR / "tsk1_opt_features.pkl", "rb") as f:
        tsk1_opt_features = pickle.load(f)

    with open(ARTIFACT_DIR / "tsk2_features.pkl", "rb") as f:
        tsk2_features = pickle.load(f)

    with open(ARTIFACT_DIR / "tsk3_features.pkl", "rb") as f:
        tsk3_features = pickle.load(f)

    with open(ARTIFACT_DIR / "tsk4_features.pkl", "rb") as f:
        tsk4_features = pickle.load(f)

    return {
        "model_tsk1_opt": model_tsk1_opt,
        "model_tsk2": model_tsk2,
        "model_tsk3_opt": model_tsk3_opt,
        "model_tsk4_opt": model_tsk4_opt,
        "tsk1_opt_result": tsk1_opt_result,
        "tsk2_result": tsk2_result,
        "tsk3_opt_result": tsk3_opt_result,
        "tsk4_opt_result": tsk4_opt_result,
        "model_summary_opt": model_summary_opt,
        "tsk1_opt_features": tsk1_opt_features,
        "tsk2_features": tsk2_features,
        "tsk3_features": tsk3_features,
        "tsk4_features": tsk4_features
    }

# ---------------------------------
# Main
# ---------------------------------
tsk = load_data()
bundle = load_artifacts()

tsk1_opt_result = bundle["tsk1_opt_result"]
tsk2_result = bundle["tsk2_result"]
tsk3_opt_result = bundle["tsk3_opt_result"]
tsk4_opt_result = bundle["tsk4_opt_result"]
model_summary_opt = bundle["model_summary_opt"]

# date parsing for charts
if "date" in tsk1_opt_result.columns:
    tsk1_opt_result["date"] = pd.to_datetime(tsk1_opt_result["date"])
if "date" in tsk2_result.columns:
    tsk2_result["date"] = pd.to_datetime(tsk2_result["date"])
if "datetime_hour" in tsk3_opt_result.columns:
    tsk3_opt_result["datetime_hour"] = pd.to_datetime(tsk3_opt_result["datetime_hour"])
if "date" in tsk4_opt_result.columns:
    tsk4_opt_result["date"] = pd.to_datetime(tsk4_opt_result["date"])

# ---------------------------------
# Dashboard
# ---------------------------------
st.title('☕ Coffee Shop Sales Intelligence Dashboard')
st.markdown('Sales monitoring, demand analysis, and forecasting for coffee shop operations.')

st.sidebar.header('Dashboard Navigation')
page = st.sidebar.radio(
    'Choose a page',
    [
        'Overview',
        'Task 1 - Daily Revenue Forecast',
        'Task 2 - Daily Orders Forecast',
        'Task 3 - Hourly Orders Forecast',
        'Task 4 - Category Demand Forecast',
        'Model Evaluation'
    ]
)

# KPI
total_revenue = tsk['total_amount'].sum()
total_orders = tsk['transaction_id'].nunique()
total_quantity = tsk['quantity'].sum()
avg_order_value_global = total_revenue / total_orders

if page == 'Overview':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Revenue', f'${total_revenue:,.2f}')
    c2.metric('Total Orders', f'{total_orders:,}')
    c3.metric('Total Quantity Sold', f'{total_quantity:,}')
    c4.metric('Average Order Value', f'${avg_order_value_global:,.2f}')

    st.subheader('Daily Revenue Trend')
    daily_revenue_plot = tsk.groupby(tsk['timestamp'].dt.floor('D'))['total_amount'].sum().reset_index()
    daily_revenue_plot.columns = ['date', 'daily_revenue']

    fig_daily = px.line(
        daily_revenue_plot,
        x='date',
        y='daily_revenue',
        title='Historical Daily Revenue'
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader('Revenue by Product Category')
    cat_rev = tsk.groupby('product_category')['total_amount'].sum().reset_index()
    fig_cat = px.bar(
        cat_rev.sort_values('total_amount', ascending=False),
        x='product_category',
        y='total_amount',
        title='Revenue by Product Category'
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader('Hourly Order Pattern')
    hourly_plot = tsk.groupby(tsk['timestamp'].dt.hour)['transaction_id'].count().reset_index()
    hourly_plot.columns = ['hour', 'orders']
    fig_hour = px.bar(
        hourly_plot,
        x='hour',
        y='orders',
        title='Orders by Hour'
    )
    st.plotly_chart(fig_hour, use_container_width=True)

elif page == 'Task 1 - Daily Revenue Forecast':
    st.subheader('Task 1: Daily Revenue Forecast')

    row = model_summary_opt[model_summary_opt['model'] == 'TSK1 Optimized Daily Revenue']
    st.metric('Model MAE', f"{row['MAE'].iloc[0]:.3f}")
    st.metric('Model RMSE', f"{row['RMSE'].iloc[0]:.3f}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=tsk1_opt_result['date'],
        y=tsk1_opt_result['actual'],
        mode='lines',
        name='Actual'
    ))
    fig1.add_trace(go.Scatter(
        x=tsk1_opt_result['date'],
        y=tsk1_opt_result['predicted'],
        mode='lines',
        name='Predicted'
    ))
    fig1.update_layout(title='Daily Revenue: Actual vs Predicted')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('Latest Forecasting Results')
    st.dataframe(tsk1_opt_result.tail(15), use_container_width=True)

elif page == 'Task 2 - Daily Orders Forecast':
    st.subheader('Task 2: Daily Orders Forecast')

    row = model_summary_opt[model_summary_opt['model'] == 'TSK2 Daily Orders']
    st.metric('Model MAE', f"{row['MAE'].iloc[0]:.3f}")
    st.metric('Model RMSE', f"{row['RMSE'].iloc[0]:.3f}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=tsk2_result['date'],
        y=tsk2_result['actual'],
        mode='lines',
        name='Actual'
    ))
    fig2.add_trace(go.Scatter(
        x=tsk2_result['date'],
        y=tsk2_result['predicted'],
        mode='lines',
        name='Predicted'
    ))
    fig2.update_layout(title='Daily Orders: Actual vs Predicted')
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(tsk2_result.tail(15), use_container_width=True)

elif page == 'Task 3 - Hourly Orders Forecast':
    st.subheader('Task 3: Hourly Orders Forecast')

    row = model_summary_opt[model_summary_opt['model'] == 'TSK3 Optimized Hourly Orders']
    st.metric('Model MAE', f"{row['MAE'].iloc[0]:.3f}")
    st.metric('Model RMSE', f"{row['RMSE'].iloc[0]:.3f}")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=tsk3_opt_result['datetime_hour'],
        y=tsk3_opt_result['actual'],
        mode='lines',
        name='Actual'
    ))
    fig3.add_trace(go.Scatter(
        x=tsk3_opt_result['datetime_hour'],
        y=tsk3_opt_result['predicted'],
        mode='lines',
        name='Predicted'
    ))
    fig3.update_layout(title='Hourly Orders: Actual vs Predicted')
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(tsk3_opt_result.tail(24), use_container_width=True)

elif page == 'Task 4 - Category Demand Forecast':
    st.subheader('Task 4: Category Demand Forecast')

    row = model_summary_opt[model_summary_opt['model'] == 'TSK4 Optimized Category Demand']
    st.metric('Model MAE', f"{row['MAE'].iloc[0]:.3f}")
    st.metric('Model RMSE', f"{row['RMSE'].iloc[0]:.3f}")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=tsk4_opt_result['date'],
        y=tsk4_opt_result['actual'],
        mode='markers',
        name='Actual'
    ))
    fig4.add_trace(go.Scatter(
        x=tsk4_opt_result['date'],
        y=tsk4_opt_result['predicted'],
        mode='markers',
        name='Predicted'
    ))
    fig4.update_layout(title='Category Demand: Actual vs Predicted')
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(tsk4_opt_result.tail(20), use_container_width=True)

elif page == 'Model Evaluation':
    st.subheader('Model Evaluation Summary')
    st.dataframe(model_summary_opt, use_container_width=True)

    fig_eval_mae = px.bar(
        model_summary_opt,
        x='model',
        y='MAE',
        title='MAE Comparison Across Tasks'
    )
    st.plotly_chart(fig_eval_mae, use_container_width=True)

    fig_eval_rmse = px.bar(
        model_summary_opt,
        x='model',
        y='RMSE',
        title='RMSE Comparison Across Tasks'
    )
    st.plotly_chart(fig_eval_rmse, use_container_width=True)