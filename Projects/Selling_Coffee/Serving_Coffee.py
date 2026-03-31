import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="Coffee Shop Sales Intelligence Dashboard",
    page_icon="☕",
    layout="wide"
)

# =====================================
# Paths
# =====================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "coffee_shop_sales.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts_r"

# =====================================
# Helpers
# =====================================
def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

@st.cache_data
def load_raw_data() -> pd.DataFrame:
    require_file(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # basic cleaning
    df["customer_age_group"] = df["customer_age_group"].fillna("Unknown")
    df["customer_gender"] = df["customer_gender"].fillna("Unknown")
    df["weather_condition"] = df["weather_condition"].fillna("Unknown")
    df["holiday_name"] = df["holiday_name"].fillna("No Holiday")

    df["temperature_c"] = (
        df.groupby(["city", "weather_condition"])["temperature_c"]
        .transform(lambda x: x.fillna(x.median()))
    )
    df["temperature_c"] = (
        df.groupby("city")["temperature_c"]
        .transform(lambda x: x.fillna(x.median()))
    )
    df["temperature_c"] = df["temperature_c"].fillna(df["temperature_c"].median())

    df["is_holiday"] = (df["holiday_name"] != "No Holiday").astype(int)

    return df

@st.cache_data
def load_artifacts() -> dict:
    required_files = {
        "tsk1_result": ARTIFACT_DIR / "tsk1_opt_result.csv",
        "tsk2_result": ARTIFACT_DIR / "tsk2_result.csv",
        "tsk3_result": ARTIFACT_DIR / "tsk3_opt_result.csv",
        "tsk4_result": ARTIFACT_DIR / "tsk4_opt_result.csv",
        "model_summary": ARTIFACT_DIR / "model_summary_opt.csv",
    }

    for path in required_files.values():
        require_file(path)

    tsk1_result = pd.read_csv(required_files["tsk1_result"])
    tsk2_result = pd.read_csv(required_files["tsk2_result"])
    tsk3_result = pd.read_csv(required_files["tsk3_result"])
    tsk4_result = pd.read_csv(required_files["tsk4_result"])
    model_summary = pd.read_csv(required_files["model_summary"])

    # parse datetime columns
    if "date" in tsk1_result.columns:
        tsk1_result["date"] = pd.to_datetime(tsk1_result["date"])

    if "date" in tsk2_result.columns:
        tsk2_result["date"] = pd.to_datetime(tsk2_result["date"])

    if "datetime_hour" in tsk3_result.columns:
        tsk3_result["datetime_hour"] = pd.to_datetime(tsk3_result["datetime_hour"])

    if "date" in tsk4_result.columns:
        tsk4_result["date"] = pd.to_datetime(tsk4_result["date"])

    return {
        "tsk1_result": tsk1_result,
        "tsk2_result": tsk2_result,
        "tsk3_result": tsk3_result,
        "tsk4_result": tsk4_result,
        "model_summary": model_summary,
    }

def get_metric_row(summary_df: pd.DataFrame, model_name: str) -> pd.Series | None:
    row = summary_df[summary_df["model"] == model_name]
    if row.empty:
        return None
    return row.iloc[0]

# =====================================
# Load
# =====================================
df = load_raw_data()
bundle = load_artifacts()

tsk1_result = bundle["tsk1_result"]
tsk2_result = bundle["tsk2_result"]
tsk3_result = bundle["tsk3_result"]
tsk4_result = bundle["tsk4_result"]
model_summary = bundle["model_summary"]

# =====================================
# Sidebar
# =====================================
st.title("☕ Coffee Shop Sales Intelligence Dashboard")
st.markdown("Static serving dashboard based on precomputed artifacts in `artifacts_r`.")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Overview",
        "Task 1 - Daily Revenue Forecast",
        "Task 2 - Daily Orders Forecast",
        "Task 3 - Hourly Orders Forecast",
        "Task 4 - Category Demand Forecast",
        "Model Evaluation",
    ]
)

# =====================================
# Global KPI
# =====================================
total_revenue = df["total_amount"].sum()
total_orders = df["transaction_id"].nunique()
total_quantity = df["quantity"].sum()
avg_order_value = total_revenue / total_orders if total_orders else 0

# =====================================
# Pages
# =====================================
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.2f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Total Quantity Sold", f"{total_quantity:,}")
    c4.metric("Average Order Value", f"${avg_order_value:,.2f}")

    st.subheader("Daily Revenue Trend")
    daily_revenue = df.groupby(df["timestamp"].dt.floor("D"))["total_amount"].sum().reset_index()
    daily_revenue.columns = ["date", "daily_revenue"]

    fig_daily = px.line(
        daily_revenue,
        x="date",
        y="daily_revenue",
        title="Historical Daily Revenue"
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    st.subheader("Revenue by Product Category")
    cat_rev = df.groupby("product_category")["total_amount"].sum().reset_index()
    fig_cat = px.bar(
        cat_rev.sort_values("total_amount", ascending=False),
        x="product_category",
        y="total_amount",
        title="Revenue by Product Category"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Hourly Order Pattern")
    hourly_plot = df.groupby(df["timestamp"].dt.hour)["transaction_id"].count().reset_index()
    hourly_plot.columns = ["hour", "orders"]
    fig_hour = px.bar(
        hourly_plot,
        x="hour",
        y="orders",
        title="Orders by Hour"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

elif page == "Task 1 - Daily Revenue Forecast":
    st.subheader("Task 1: Daily Revenue Forecast")

    row = get_metric_row(model_summary, "TSK1 Optimized Daily Revenue")
    if row is not None:
        c1, c2 = st.columns(2)
        c1.metric("Model MAE", f"{row['MAE']:.3f}")
        c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=tsk1_result["date"],
        y=tsk1_result["actual"],
        mode="lines",
        name="Actual"
    ))
    fig1.add_trace(go.Scatter(
        x=tsk1_result["date"],
        y=tsk1_result["predicted"],
        mode="lines",
        name="Predicted"
    ))
    fig1.update_layout(title="Daily Revenue: Actual vs Predicted")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Latest Forecasting Results")
    st.dataframe(tsk1_result.tail(15), use_container_width=True)

elif page == "Task 2 - Daily Orders Forecast":
    st.subheader("Task 2: Daily Orders Forecast")

    row = get_metric_row(model_summary, "TSK2 Daily Orders")
    if row is not None:
        c1, c2 = st.columns(2)
        c1.metric("Model MAE", f"{row['MAE']:.3f}")
        c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=tsk2_result["date"],
        y=tsk2_result["actual"],
        mode="lines",
        name="Actual"
    ))
    fig2.add_trace(go.Scatter(
        x=tsk2_result["date"],
        y=tsk2_result["predicted"],
        mode="lines",
        name="Predicted"
    ))
    fig2.update_layout(title="Daily Orders: Actual vs Predicted")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(tsk2_result.tail(15), use_container_width=True)

elif page == "Task 3 - Hourly Orders Forecast":
    st.subheader("Task 3: Hourly Orders Forecast")

    row = get_metric_row(model_summary, "TSK3 Optimized Hourly Orders")
    if row is not None:
        c1, c2 = st.columns(2)
        c1.metric("Model MAE", f"{row['MAE']:.3f}")
        c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=tsk3_result["datetime_hour"],
        y=tsk3_result["actual"],
        mode="lines",
        name="Actual"
    ))
    fig3.add_trace(go.Scatter(
        x=tsk3_result["datetime_hour"],
        y=tsk3_result["predicted"],
        mode="lines",
        name="Predicted"
    ))
    fig3.update_layout(title="Hourly Orders: Actual vs Predicted")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(tsk3_result.tail(24), use_container_width=True)

elif page == "Task 4 - Category Demand Forecast":
    st.subheader("Task 4: Category Demand Forecast")

    row = get_metric_row(model_summary, "TSK4 Optimized Category Demand")
    if row is not None:
        c1, c2 = st.columns(2)
        c1.metric("Model MAE", f"{row['MAE']:.3f}")
        c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=tsk4_result["date"],
        y=tsk4_result["actual"],
        mode="markers",
        name="Actual"
    ))
    fig4.add_trace(go.Scatter(
        x=tsk4_result["date"],
        y=tsk4_result["predicted"],
        mode="markers",
        name="Predicted"
    ))
    fig4.update_layout(title="Category Demand: Actual vs Predicted")
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(tsk4_result.tail(20), use_container_width=True)

elif page == "Model Evaluation":
    st.subheader("Model Evaluation Summary")
    st.dataframe(model_summary, use_container_width=True)

    fig_eval_mae = px.bar(
        model_summary,
        x="model",
        y="MAE",
        title="MAE Comparison Across Tasks"
    )
    st.plotly_chart(fig_eval_mae, use_container_width=True)

    fig_eval_rmse = px.bar(
        model_summary,
        x="model",
        y="RMSE",
        title="RMSE Comparison Across Tasks"
    )
    st.plotly_chart(fig_eval_rmse, use_container_width=True)