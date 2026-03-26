# Selling Coffee
# This project is about analyzing the sales data from coffee shops around the world. 


# A.Import
# Import streamlit
import streamlit as st 
# Import data analysis basic libraries
import pandas as pd
import numpy as np
# Import datetime preocessing libraries
import datetime as dt
from datetime import datetime, timedelta
# Import visualization libraries
import plotly.express as px
import plotly.graph_objects as go   # Main
import matplotlib.pyplot as plt
import seaborn as sns   # Support
# Import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
# Import feature engineering libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# More of time series analysis libraries
from sklearn.model_selection import TimeSeriesSplit
# Filter warnings
import warnings
warnings.filterwarnings('ignore')
# About streamlit
@st.cache_data
def load_data():
    return pd.read_csv("coffee_shop_sales.csv")

# B.Load and sort data
# Proceed basic cleaning process for the original dataset
tsk = pd.read_csv(r"E:\Project95\Projects\Selling_Coffee\coffee_shop_sales.csv")
# TEST1
tsk.head()
tsk.info()


# Filling Nas
# Categorical features
tsk["customer_age_group"] = tsk["customer_age_group"].fillna("Unknown")
tsk["customer_gender"] = tsk["customer_gender"].fillna("Unknown") # Abandon median due to low missing level
tsk["weather_condition"] = tsk["weather_condition"].fillna("Unknown")
tsk["holiday_name"] = tsk["holiday_name"].fillna("No Holiday")
# Temparature by median
tsk["temperature_c"] = tsk.groupby(["city", "weather_condition"])["temperature_c"]\
    .transform(lambda x: x.fillna(x.median()))

tsk["temperature_c"] = tsk.groupby("city")["temperature_c"]\
    .transform(lambda x: x.fillna(x.median()))

tsk["temperature_c"] = tsk["temperature_c"].fillna(tsk["temperature_c"].median())
# About Holidays
tsk["is_holiday"] = (tsk["holiday_name"] != "No Holiday").astype(int)

# Dealing with date and time
tsk['timestamp'] = pd.to_datetime(tsk['timestamp'])
# TEST2
tsk.info()


# C.Feature Engineering

# Sort data into four sets according to accomplish four main tasks:

# Task1: Daily Revenue
# target: daily_revenue

tsk1 = tsk.copy()

# 基础日期列
tsk1['date'] = tsk1['timestamp'].dt.floor('D')

# 先按天聚合出目标和外部变量
tsk1 = tsk1.groupby('date').agg(
    daily_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean'),
    order_count=('transaction_id', 'count')
).reset_index()

# 节假日特征：当天只要出现 holiday_name 就视为节假日
holiday_daily = tsk.copy()
holiday_daily['date'] = holiday_daily['timestamp'].dt.floor('D')
holiday_daily = holiday_daily.groupby('date')['holiday_name'].apply(
    lambda x: int(x.notna().any())
).reset_index(name='is_holiday')

tsk1 = tsk1.merge(holiday_daily, on='date', how='left')

# 时间特征
tsk1['day_of_week'] = tsk1['date'].dt.dayofweek
tsk1['month'] = tsk1['date'].dt.month
tsk1['day_of_month'] = tsk1['date'].dt.day
tsk1['week_of_year'] = tsk1['date'].dt.isocalendar().week.astype(int)
tsk1['is_weekend'] = tsk1['day_of_week'].isin([5, 6]).astype(int)

# 滞后特征
tsk1['lag_1'] = tsk1['daily_revenue'].shift(1)
tsk1['lag_7'] = tsk1['daily_revenue'].shift(7)
tsk1['lag_14'] = tsk1['daily_revenue'].shift(14)

# 滚动统计特征
tsk1['rolling_mean_7'] = tsk1['daily_revenue'].shift(1).rolling(7).mean()
tsk1['rolling_mean_14'] = tsk1['daily_revenue'].shift(1).rolling(14).mean()
tsk1['rolling_std_7'] = tsk1['daily_revenue'].shift(1).rolling(7).std()

# 清理
tsk1 = tsk1.dropna().reset_index(drop=True)


# TASK2: Daily Orders
# target: daily_orders


tsk2 = tsk.copy()

tsk2['date'] = tsk2['timestamp'].dt.floor('D')

tsk2 = tsk2.groupby('date').agg(
    daily_orders=('transaction_id', 'count'),
    daily_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean')
).reset_index()

holiday_daily = tsk.copy()
holiday_daily['date'] = holiday_daily['timestamp'].dt.floor('D')
holiday_daily = holiday_daily.groupby('date')['holiday_name'].apply(
    lambda x: int(x.notna().any())
).reset_index(name='is_holiday')

tsk2 = tsk2.merge(holiday_daily, on='date', how='left')

# 衍生交叉特征
tsk2['avg_order_value'] = tsk2['daily_revenue'] / tsk2['daily_orders']

# 时间特征
tsk2['day_of_week'] = tsk2['date'].dt.dayofweek
tsk2['month'] = tsk2['date'].dt.month
tsk2['day_of_month'] = tsk2['date'].dt.day
tsk2['week_of_year'] = tsk2['date'].dt.isocalendar().week.astype(int)
tsk2['is_weekend'] = tsk2['day_of_week'].isin([5, 6]).astype(int)

# 滞后特征
tsk2['lag_orders_1'] = tsk2['daily_orders'].shift(1)
tsk2['lag_orders_7'] = tsk2['daily_orders'].shift(7)
tsk2['lag_orders_14'] = tsk2['daily_orders'].shift(14)

# 滚动特征
tsk2['rolling_orders_7'] = tsk2['daily_orders'].shift(1).rolling(7).mean()
tsk2['rolling_orders_std_7'] = tsk2['daily_orders'].shift(1).rolling(7).std()

# 清理
tsk2 = tsk2.dropna().reset_index(drop=True)


# TASK3: Hourly Orders
# target: hourly_orders


tsk3 = tsk.copy()

tsk3['datetime_hour'] = tsk3['timestamp'].dt.floor('H')
tsk3['date'] = tsk3['datetime_hour'].dt.floor('D')
tsk3['hour'] = tsk3['datetime_hour'].dt.hour

tsk3 = tsk3.groupby('datetime_hour').agg(
    hourly_orders=('transaction_id', 'count'),
    hourly_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean')
).reset_index()

# 再拆时间
tsk3['date'] = tsk3['datetime_hour'].dt.floor('D')
tsk3['hour'] = tsk3['datetime_hour'].dt.hour
tsk3['day_of_week'] = tsk3['datetime_hour'].dt.dayofweek
tsk3['month'] = tsk3['datetime_hour'].dt.month
tsk3['is_weekend'] = tsk3['day_of_week'].isin([5, 6]).astype(int)
tsk3['is_peak_hour'] = tsk3['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# 小时级节假日
holiday_hourly = tsk.copy()
holiday_hourly['datetime_hour'] = holiday_hourly['timestamp'].dt.floor('H')
holiday_hourly = holiday_hourly.groupby('datetime_hour')['holiday_name'].apply(
    lambda x: int(x.notna().any())
).reset_index(name='is_holiday')

tsk3 = tsk3.merge(holiday_hourly, on='datetime_hour', how='left')

# 滞后特征
tsk3['lag_1h'] = tsk3['hourly_orders'].shift(1)
tsk3['lag_24'] = tsk3['hourly_orders'].shift(24)     # 前一天同小时
tsk3['lag_168'] = tsk3['hourly_orders'].shift(168)   # 前一周同小时

# 滚动特征
tsk3['rolling_mean_24'] = tsk3['hourly_orders'].shift(1).rolling(24).mean()
tsk3['rolling_std_24'] = tsk3['hourly_orders'].shift(1).rolling(24).std()

# 清理
tsk3 = tsk3.dropna().reset_index(drop=True)


# TASK4: Category Demand
# target: category_demand


tsk4 = tsk.copy()

tsk4['date'] = tsk4['timestamp'].dt.floor('D')

tsk4 = tsk4.groupby(['date', 'product_category']).agg(
    category_demand=('quantity', 'sum'),
    category_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean')
).reset_index()

# 节假日
holiday_daily = tsk.copy()
holiday_daily['date'] = holiday_daily['timestamp'].dt.floor('D')
holiday_daily = holiday_daily.groupby('date')['holiday_name'].apply(
    lambda x: int(x.notna().any())
).reset_index(name='is_holiday')

tsk4 = tsk4.merge(holiday_daily, on='date', how='left')

# 时间特征
tsk4['day_of_week'] = tsk4['date'].dt.dayofweek
tsk4['month'] = tsk4['date'].dt.month
tsk4['day_of_month'] = tsk4['date'].dt.day
tsk4['week_of_year'] = tsk4['date'].dt.isocalendar().week.astype(int)
tsk4['is_weekend'] = tsk4['day_of_week'].isin([5, 6]).astype(int)

# 按品类排序后构造 lag / rolling
tsk4 = tsk4.sort_values(['product_category', 'date']).reset_index(drop=True)

tsk4['lag_cat_1'] = tsk4.groupby('product_category')['category_demand'].shift(1)
tsk4['lag_cat_7'] = tsk4.groupby('product_category')['category_demand'].shift(7)
tsk4['lag_cat_14'] = tsk4.groupby('product_category')['category_demand'].shift(14)

tsk4['rolling_cat_mean_7'] = (
    tsk4.groupby('product_category')['category_demand']
        .shift(1)
        .rolling(7)
        .mean()
)

tsk4['rolling_cat_std_7'] = (
    tsk4.groupby('product_category')['category_demand']
        .shift(1)
        .rolling(7)
        .std()
)

# 清理
tsk4 = tsk4.dropna().reset_index(drop=True)

# TEST3
print(tsk1.head())
print(tsk2.head())
print(tsk3.head())
print(tsk4.head())