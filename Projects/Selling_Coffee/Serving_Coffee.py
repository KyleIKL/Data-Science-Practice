import streamlit as st
import pandas as pd
import numpy as np
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

    df["date"] = df["timestamp"].dt.floor("D")
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week_num"] = df["timestamp"].dt.dayofweek
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["month_num"] = df["timestamp"].dt.month
    df["month_name"] = df["timestamp"].dt.month_name()

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

def get_metric_row(summary_df: pd.DataFrame, model_name: str):
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
# Sidebar Filters
# =====================================
st.title("☕ Coffee Shop Sales Intelligence Dashboard")
st.markdown("Interactive static dashboard powered by historical sales data and precomputed forecast artifacts.")

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "Executive Overview",
        "Demand Patterns",
        "Product Intelligence",
        "Customer & Payment Insights",
        "Store & City Performance",
        "Forecast Views",
        "Model Evaluation"
    ]
)

city_options = ["All"] + sorted(df["city"].dropna().unique().tolist())
category_options = ["All"] + sorted(df["product_category"].dropna().unique().tolist())
weather_options = ["All"] + sorted(df["weather_condition"].dropna().unique().tolist())

selected_city = st.sidebar.selectbox("City", city_options)
selected_category = st.sidebar.selectbox("Category", category_options)
selected_weather = st.sidebar.selectbox("Weather", weather_options)

filtered_df = df.copy()

if selected_city != "All":
    filtered_df = filtered_df[filtered_df["city"] == selected_city]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["product_category"] == selected_category]

if selected_weather != "All":
    filtered_df = filtered_df[filtered_df["weather_condition"] == selected_weather]

# =====================================
# Global KPI
# =====================================
total_revenue = filtered_df["total_amount"].sum()
total_orders = filtered_df["transaction_id"].nunique()
total_quantity = filtered_df["quantity"].sum()
avg_order_value = total_revenue / total_orders if total_orders else 0
avg_items_per_order = total_quantity / total_orders if total_orders else 0

# =====================================
# Executive Overview
# =====================================
if page == "Executive Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", f"${total_revenue:,.2f}")
    c2.metric("Total Orders", f"{total_orders:,}")
    c3.metric("Units Sold", f"{total_quantity:,}")
    c4.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    c5.metric("Items / Order", f"{avg_items_per_order:.2f}")

    left, right = st.columns([1.7, 1])

    with left:
        daily_rev = filtered_df.groupby("date", as_index=False)["total_amount"].sum()
        fig = px.area(
            daily_rev,
            x="date",
            y="total_amount",
            title="Daily Revenue Trend"
        )
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        cat_rev = filtered_df.groupby("product_category", as_index=False)["total_amount"].sum()
        fig = px.pie(
            cat_rev,
            names="product_category",
            values="total_amount",
            hole=0.55,
            title="Revenue Share by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

    b1, b2 = st.columns(2)

    with b1:
        weekday_rev = (
            filtered_df.groupby(["day_of_week_num", "day_of_week"], as_index=False)["total_amount"]
            .sum()
            .sort_values("day_of_week_num")
        )
        fig = px.bar(
            weekday_rev,
            x="day_of_week",
            y="total_amount",
            title="Revenue by Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True)

    with b2:
        hourly_orders = filtered_df.groupby("hour", as_index=False)["transaction_id"].count()
        hourly_orders.columns = ["hour", "orders"]
        fig = px.bar(
            hourly_orders,
            x="hour",
            y="orders",
            title="Orders by Hour"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================
# Demand Patterns
# =====================================
elif page == "Demand Patterns":
    st.subheader("Demand Patterns")

    left, right = st.columns(2)

    with left:
        heat_df = (
            filtered_df.groupby(["day_of_week_num", "day_of_week", "hour"], as_index=False)["transaction_id"]
            .count()
        )
        heat_df.rename(columns={"transaction_id": "orders"}, inplace=True)
        heat_pivot = heat_df.pivot(index="day_of_week", columns="hour", values="orders")
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heat_pivot = heat_pivot.reindex(ordered_days)

        fig = px.imshow(
            heat_pivot,
            aspect="auto",
            title="Order Heatmap: Day of Week × Hour"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        holiday_comp = filtered_df.groupby("is_holiday", as_index=False).agg(
            revenue=("total_amount", "sum"),
            orders=("transaction_id", "count")
        )
        holiday_comp["is_holiday"] = holiday_comp["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(
            holiday_comp,
            x="is_holiday",
            y="revenue",
            text="orders",
            title="Holiday vs Non-Holiday Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

    monthly_rev = filtered_df.groupby(["month_num", "month_name"], as_index=False)["total_amount"].sum()
    monthly_rev = monthly_rev.sort_values("month_num")

    fig = px.line(
        monthly_rev,
        x="month_name",
        y="total_amount",
        markers=True,
        title="Monthly Revenue Pattern"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# Product Intelligence
# =====================================
elif page == "Product Intelligence":
    st.subheader("Product Intelligence")

    product_summary = filtered_df.groupby("product_name", as_index=False).agg(
        revenue=("total_amount", "sum"),
        quantity=("quantity", "sum"),
        orders=("transaction_id", "count")
    )
    product_summary["avg_revenue_per_order"] = product_summary["revenue"] / product_summary["orders"]

    left, right = st.columns(2)

    with left:
        top_rev = product_summary.sort_values("revenue", ascending=False).head(10)
        fig = px.bar(
            top_rev,
            x="product_name",
            y="revenue",
            title="Top 10 Products by Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        top_qty = product_summary.sort_values("quantity", ascending=False).head(10)
        fig = px.bar(
            top_qty,
            x="product_name",
            y="quantity",
            title="Top 10 Products by Quantity"
        )
        st.plotly_chart(fig, use_container_width=True)

    bubble = product_summary.sort_values("revenue", ascending=False).head(25)
    fig = px.scatter(
        bubble,
        x="quantity",
        y="revenue",
        size="orders",
        hover_name="product_name",
        title="Top Products Bubble Map: Quantity vs Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    category_summary = filtered_df.groupby("product_category", as_index=False).agg(
        revenue=("total_amount", "sum"),
        quantity=("quantity", "sum"),
        orders=("transaction_id", "count")
    )
    st.dataframe(category_summary.sort_values("revenue", ascending=False), use_container_width=True)

# =====================================
# Customer & Payment Insights
# =====================================
elif page == "Customer & Payment Insights":
    st.subheader("Customer & Payment Insights")

    c1, c2 = st.columns(2)

    with c1:
        payment_dist = filtered_df.groupby("payment_method", as_index=False)["transaction_id"].count()
        payment_dist.rename(columns={"transaction_id": "orders"}, inplace=True)
        fig = px.pie(
            payment_dist,
            names="payment_method",
            values="orders",
            hole=0.55,
            title="Payment Method Mix"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        loyalty_summary = filtered_df.groupby("loyalty_member", as_index=False).agg(
            revenue=("total_amount", "sum"),
            orders=("transaction_id", "count")
        )
        loyalty_summary["loyalty_member"] = loyalty_summary["loyalty_member"].map({True: "Member", False: "Non-member"})
        fig = px.bar(
            loyalty_summary,
            x="loyalty_member",
            y="revenue",
            color="loyalty_member",
            title="Member vs Non-member Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

    a1, a2 = st.columns(2)

    with a1:
        age_summary = filtered_df.groupby("customer_age_group", as_index=False)["total_amount"].sum()
        fig = px.bar(
            age_summary,
            x="customer_age_group",
            y="total_amount",
            title="Revenue by Age Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        gender_summary = filtered_df.groupby("customer_gender", as_index=False)["total_amount"].sum()
        fig = px.bar(
            gender_summary,
            x="customer_gender",
            y="total_amount",
            title="Revenue by Gender"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================
# Store & City Performance
# =====================================
elif page == "Store & City Performance":
    st.subheader("Store & City Performance")

    left, right = st.columns(2)

    with left:
        city_revenue = filtered_df.groupby("city", as_index=False)["total_amount"].sum()
        fig = px.bar(
            city_revenue.sort_values("total_amount", ascending=False),
            x="city",
            y="total_amount",
            title="Revenue by City"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        weather_revenue = filtered_df.groupby("weather_condition", as_index=False)["total_amount"].sum()
        fig = px.bar(
            weather_revenue.sort_values("total_amount", ascending=False),
            x="weather_condition",
            y="total_amount",
            title="Revenue by Weather Condition"
        )
        st.plotly_chart(fig, use_container_width=True)

    store_revenue = filtered_df.groupby("store_id", as_index=False).agg(
        revenue=("total_amount", "sum"),
        orders=("transaction_id", "count")
    )
    top_store = store_revenue.sort_values("revenue", ascending=False).head(15)
    fig = px.bar(
        top_store,
        x="store_id",
        y="revenue",
        color="orders",
        title="Top 15 Stores by Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# Forecast Views
# =====================================
elif page == "Forecast Views":
    task = st.selectbox(
        "Select Forecast View",
        [
            "Task 1 - Daily Revenue",
            "Task 2 - Daily Orders",
            "Task 3 - Hourly Orders",
            "Task 4 - Category Demand"
        ]
    )

    if task == "Task 1 - Daily Revenue":
        row = get_metric_row(model_summary, "TSK1 Optimized Daily Revenue")
        if row is not None:
            c1, c2 = st.columns(2)
            c1.metric("Model MAE", f"{row['MAE']:.3f}")
            c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsk1_result["date"],
            y=tsk1_result["actual"],
            mode="lines",
            name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=tsk1_result["date"],
            y=tsk1_result["predicted"],
            mode="lines",
            name="Predicted"
        ))
        fig.update_layout(title="Daily Revenue: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tsk1_result.tail(20), use_container_width=True)

    elif task == "Task 2 - Daily Orders":
        row = get_metric_row(model_summary, "TSK2 Daily Orders")
        if row is not None:
            c1, c2 = st.columns(2)
            c1.metric("Model MAE", f"{row['MAE']:.3f}")
            c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsk2_result["date"],
            y=tsk2_result["actual"],
            mode="lines",
            name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=tsk2_result["date"],
            y=tsk2_result["predicted"],
            mode="lines",
            name="Predicted"
        ))
        fig.update_layout(title="Daily Orders: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tsk2_result.tail(20), use_container_width=True)

    elif task == "Task 3 - Hourly Orders":
        row = get_metric_row(model_summary, "TSK3 Optimized Hourly Orders")
        if row is not None:
            c1, c2 = st.columns(2)
            c1.metric("Model MAE", f"{row['MAE']:.3f}")
            c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsk3_result["datetime_hour"],
            y=tsk3_result["actual"],
            mode="lines",
            name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=tsk3_result["datetime_hour"],
            y=tsk3_result["predicted"],
            mode="lines",
            name="Predicted"
        ))
        fig.update_layout(title="Hourly Orders: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tsk3_result.tail(24), use_container_width=True)

    elif task == "Task 4 - Category Demand":
        row = get_metric_row(model_summary, "TSK4 Optimized Category Demand")
        if row is not None:
            c1, c2 = st.columns(2)
            c1.metric("Model MAE", f"{row['MAE']:.3f}")
            c2.metric("Model RMSE", f"{row['RMSE']:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsk4_result["date"],
            y=tsk4_result["actual"],
            mode="markers",
            name="Actual"
        ))
        fig.add_trace(go.Scatter(
            x=tsk4_result["date"],
            y=tsk4_result["predicted"],
            mode="markers",
            name="Predicted"
        ))
        fig.update_layout(title="Category Demand: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tsk4_result.tail(20), use_container_width=True)

# =====================================
# Model Evaluation
# =====================================
elif page == "Model Evaluation":
    st.subheader("Model Evaluation Summary")
    st.dataframe(model_summary, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            model_summary,
            x="model",
            y="MAE",
            title="MAE Comparison Across Tasks"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(
            model_summary,
            x="model",
            y="RMSE",
            title="RMSE Comparison Across Tasks"
        )
        st.plotly_chart(fig, use_container_width=True)