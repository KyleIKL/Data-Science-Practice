# Selling Coffee
# This project is about analyzing the sales data from coffee shops around the world. 


# A.Import

import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "coffee_shop_sales.csv")

tsk = load_data()

# B.Load and sort data
# Proceed basic cleaning process for the original dataset

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

# D. Modeling and Evaluation
# D.Modeling


# 0. Fix holiday logic

holiday_daily = tsk.copy()
holiday_daily['date'] = holiday_daily['timestamp'].dt.floor('D')
holiday_daily = holiday_daily.groupby('date')['holiday_name'].apply(
    lambda x: int((x != 'No Holiday').any())
).reset_index(name='is_holiday')

holiday_hourly = tsk.copy()
holiday_hourly['datetime_hour'] = holiday_hourly['timestamp'].dt.floor('H')
holiday_hourly = holiday_hourly.groupby('datetime_hour')['holiday_name'].apply(
    lambda x: int((x != 'No Holiday').any())
).reset_index(name='is_holiday')



# 1. Rebuild is_holiday in each task set

# TSK1
tsk1 = tsk1.drop(columns=['is_holiday'], errors='ignore')
tsk1 = tsk1.merge(holiday_daily, on='date', how='left')

# TSK2
tsk2 = tsk2.drop(columns=['is_holiday'], errors='ignore')
tsk2 = tsk2.merge(holiday_daily, on='date', how='left')

# TSK3
tsk3 = tsk3.drop(columns=['is_holiday'], errors='ignore')
tsk3 = tsk3.merge(holiday_hourly, on='datetime_hour', how='left')

# TSK4
tsk4 = tsk4.drop(columns=['is_holiday'], errors='ignore')
tsk4 = tsk4.merge(holiday_daily, on='date', how='left')



# 2. Train-test split function for time series

def time_train_test_split(df, date_col, test_ratio=0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df



# 3. Evaluation function

def evaluate_regression(y_true, y_pred, model_name='Model'):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f'{model_name} MAE: {mae:.4f}')
    print(f'{model_name} RMSE: {rmse:.4f}')
    
    return {
        'model': model_name,
        'MAE': mae,
        'RMSE': rmse
    }



# 4. TSK1: Daily Revenue Model

tsk1_model = tsk1.copy()

tsk1_features = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday', 'avg_temp',
    'discount_rate', 'loyalty_ratio', 'order_count',
    'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7'
]

tsk1_target = 'daily_revenue'

train1, test1 = time_train_test_split(tsk1_model, 'date', test_ratio=0.2)

X_train1 = train1[tsk1_features]
y_train1 = train1[tsk1_target]
X_test1 = test1[tsk1_features]
y_test1 = test1[tsk1_target]

model_tsk1 = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_tsk1.fit(X_train1, y_train1)
pred1 = model_tsk1.predict(X_test1)

tsk1_eval = evaluate_regression(y_test1, pred1, model_name='TSK1 Daily Revenue')

tsk1_result = test1[['date']].copy()
tsk1_result['actual_daily_revenue'] = y_test1.values
tsk1_result['predicted_daily_revenue'] = pred1



# 5. TSK2: Daily Orders Model

tsk2_model = tsk2.copy()

tsk2_features = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday', 'avg_temp',
    'discount_rate', 'loyalty_ratio', 'daily_revenue', 'avg_order_value',
    'lag_orders_1', 'lag_orders_7', 'lag_orders_14',
    'rolling_orders_7', 'rolling_orders_std_7'
]

tsk2_target = 'daily_orders'

train2, test2 = time_train_test_split(tsk2_model, 'date', test_ratio=0.2)

X_train2 = train2[tsk2_features]
y_train2 = train2[tsk2_target]
X_test2 = test2[tsk2_features]
y_test2 = test2[tsk2_target]

model_tsk2 = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_tsk2.fit(X_train2, y_train2)
pred2 = model_tsk2.predict(X_test2)

tsk2_eval = evaluate_regression(y_test2, pred2, model_name='TSK2 Daily Orders')

tsk2_result = test2[['date']].copy()
tsk2_result['actual_daily_orders'] = y_test2.values
tsk2_result['predicted_daily_orders'] = pred2



# 6. TSK3: Hourly Orders Model

tsk3_model = tsk3.copy()

tsk3_features = [
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_peak_hour', 'is_holiday',
    'avg_temp', 'discount_rate', 'loyalty_ratio', 'hourly_revenue',
    'lag_1h', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24'
]

tsk3_target = 'hourly_orders'

train3, test3 = time_train_test_split(tsk3_model, 'datetime_hour', test_ratio=0.2)

X_train3 = train3[tsk3_features]
y_train3 = train3[tsk3_target]
X_test3 = test3[tsk3_features]
y_test3 = test3[tsk3_target]

model_tsk3 = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_tsk3.fit(X_train3, y_train3)
pred3 = model_tsk3.predict(X_test3)

tsk3_eval = evaluate_regression(y_test3, pred3, model_name='TSK3 Hourly Orders')

tsk3_result = test3[['datetime_hour']].copy()
tsk3_result['actual_hourly_orders'] = y_test3.values
tsk3_result['predicted_hourly_orders'] = pred3



# 7. TSK4: Category Demand Model

tsk4_model = tsk4.copy()

# category needs encoding
tsk4_model = pd.get_dummies(tsk4_model, columns=['product_category'], drop_first=True)

tsk4_feature_base = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday', 'avg_temp',
    'discount_rate', 'loyalty_ratio', 'category_revenue',
    'lag_cat_1', 'lag_cat_7', 'lag_cat_14',
    'rolling_cat_mean_7', 'rolling_cat_std_7'
]

tsk4_category_cols = [col for col in tsk4_model.columns if col.startswith('product_category_')]
tsk4_features = tsk4_feature_base + tsk4_category_cols
tsk4_target = 'category_demand'

train4, test4 = time_train_test_split(tsk4_model, 'date', test_ratio=0.2)

X_train4 = train4[tsk4_features]
y_train4 = train4[tsk4_target]
X_test4 = test4[tsk4_features]
y_test4 = test4[tsk4_target]

model_tsk4 = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_tsk4.fit(X_train4, y_train4)
pred4 = model_tsk4.predict(X_test4)

tsk4_eval = evaluate_regression(y_test4, pred4, model_name='TSK4 Category Demand')

tsk4_result = test4[['date']].copy()
tsk4_result['actual_category_demand'] = y_test4.values
tsk4_result['predicted_category_demand'] = pred4



# 8. Summary of model performance

model_summary = pd.DataFrame([
    tsk1_eval,
    tsk2_eval,
    tsk3_eval,
    tsk4_eval
])

print(model_summary)


# E. Optimization
# E.Optimization

# --------------------------------------------------
# 1. Better holiday feature
# --------------------------------------------------
holiday_daily_fix = tsk.copy()
holiday_daily_fix['date'] = holiday_daily_fix['timestamp'].dt.floor('D')
holiday_daily_fix = holiday_daily_fix.groupby('date')['holiday_name'].apply(
    lambda x: int((x != 'No Holiday').any())
).reset_index(name='is_holiday_fix')

holiday_hourly_fix = tsk.copy()
holiday_hourly_fix['datetime_hour'] = holiday_hourly_fix['timestamp'].dt.floor('H')
holiday_hourly_fix = holiday_hourly_fix.groupby('datetime_hour')['holiday_name'].apply(
    lambda x: int((x != 'No Holiday').any())
).reset_index(name='is_holiday_fix')


# --------------------------------------------------
# 2. Rebuild TSK1 with richer features
# --------------------------------------------------
tsk1_opt = tsk.copy()
tsk1_opt['date'] = tsk1_opt['timestamp'].dt.floor('D')

# main daily aggregation
tsk1_opt = tsk1_opt.groupby('date').agg(
    daily_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean'),
    order_count=('transaction_id', 'count'),
    avg_unit_price=('unit_price', 'mean'),
    total_quantity=('quantity', 'sum')
).reset_index()

# holiday
tsk1_opt = tsk1_opt.merge(holiday_daily_fix, on='date', how='left')
tsk1_opt.rename(columns={'is_holiday_fix': 'is_holiday'}, inplace=True)

# derived feature
tsk1_opt['avg_order_value'] = tsk1_opt['daily_revenue'] / tsk1_opt['order_count']

# category mix features
daily_cat = tsk.copy()
daily_cat['date'] = daily_cat['timestamp'].dt.floor('D')

cat_pivot = daily_cat.pivot_table(
    index='date',
    columns='product_category',
    values='total_amount',
    aggfunc='sum',
    fill_value=0
).reset_index()

# convert to revenue share
cat_cols = [c for c in cat_pivot.columns if c != 'date']
for c in cat_cols:
    cat_pivot[f'{c}_share'] = cat_pivot[c] / cat_pivot[cat_cols].sum(axis=1)

share_cols = ['date'] + [f'{c}_share' for c in cat_cols]
cat_share = cat_pivot[share_cols].copy()

tsk1_opt = tsk1_opt.merge(cat_share, on='date', how='left')

# time features
tsk1_opt['day_of_week'] = tsk1_opt['date'].dt.dayofweek
tsk1_opt['month'] = tsk1_opt['date'].dt.month
tsk1_opt['day_of_month'] = tsk1_opt['date'].dt.day
tsk1_opt['week_of_year'] = tsk1_opt['date'].dt.isocalendar().week.astype(int)
tsk1_opt['is_weekend'] = tsk1_opt['day_of_week'].isin([5, 6]).astype(int)

# cyclic encoding
tsk1_opt['dow_sin'] = np.sin(2 * np.pi * tsk1_opt['day_of_week'] / 7)
tsk1_opt['dow_cos'] = np.cos(2 * np.pi * tsk1_opt['day_of_week'] / 7)
tsk1_opt['month_sin'] = np.sin(2 * np.pi * tsk1_opt['month'] / 12)
tsk1_opt['month_cos'] = np.cos(2 * np.pi * tsk1_opt['month'] / 12)

# lag features
tsk1_opt['lag_1'] = tsk1_opt['daily_revenue'].shift(1)
tsk1_opt['lag_7'] = tsk1_opt['daily_revenue'].shift(7)
tsk1_opt['lag_14'] = tsk1_opt['daily_revenue'].shift(14)

# rolling features
tsk1_opt['rolling_mean_7'] = tsk1_opt['daily_revenue'].shift(1).rolling(7).mean()
tsk1_opt['rolling_mean_14'] = tsk1_opt['daily_revenue'].shift(1).rolling(14).mean()
tsk1_opt['rolling_std_7'] = tsk1_opt['daily_revenue'].shift(1).rolling(7).std()

tsk1_opt = tsk1_opt.dropna().reset_index(drop=True)


# --------------------------------------------------
# 3. Rebuild TSK3 holiday feature only
# --------------------------------------------------
tsk3_opt = tsk3.copy()
tsk3_opt = tsk3_opt.drop(columns=['is_holiday'], errors='ignore')
tsk3_opt = tsk3_opt.merge(holiday_hourly_fix, on='datetime_hour', how='left')
tsk3_opt.rename(columns={'is_holiday_fix': 'is_holiday'}, inplace=True)


# --------------------------------------------------
# 4. Rebuild TSK4 with safer grouped rolling
# --------------------------------------------------
tsk4_opt = tsk.copy()
tsk4_opt['date'] = tsk4_opt['timestamp'].dt.floor('D')

tsk4_opt = tsk4_opt.groupby(['date', 'product_category']).agg(
    category_demand=('quantity', 'sum'),
    category_revenue=('total_amount', 'sum'),
    avg_temp=('temperature_c', 'mean'),
    discount_rate=('discount_applied', 'mean'),
    loyalty_ratio=('loyalty_member', 'mean'),
    avg_unit_price=('unit_price', 'mean')
).reset_index()

tsk4_opt = tsk4_opt.merge(holiday_daily_fix, on='date', how='left')
tsk4_opt.rename(columns={'is_holiday_fix': 'is_holiday'}, inplace=True)

tsk4_opt['day_of_week'] = tsk4_opt['date'].dt.dayofweek
tsk4_opt['month'] = tsk4_opt['date'].dt.month
tsk4_opt['day_of_month'] = tsk4_opt['date'].dt.day
tsk4_opt['week_of_year'] = tsk4_opt['date'].dt.isocalendar().week.astype(int)
tsk4_opt['is_weekend'] = tsk4_opt['day_of_week'].isin([5, 6]).astype(int)

tsk4_opt = tsk4_opt.sort_values(['product_category', 'date']).reset_index(drop=True)

tsk4_opt['lag_cat_1'] = tsk4_opt.groupby('product_category')['category_demand'].transform(lambda s: s.shift(1))
tsk4_opt['lag_cat_7'] = tsk4_opt.groupby('product_category')['category_demand'].transform(lambda s: s.shift(7))
tsk4_opt['lag_cat_14'] = tsk4_opt.groupby('product_category')['category_demand'].transform(lambda s: s.shift(14))

tsk4_opt['rolling_cat_mean_7'] = tsk4_opt.groupby('product_category')['category_demand'].transform(
    lambda s: s.shift(1).rolling(7).mean()
)
tsk4_opt['rolling_cat_std_7'] = tsk4_opt.groupby('product_category')['category_demand'].transform(
    lambda s: s.shift(1).rolling(7).std()
)

tsk4_opt = tsk4_opt.dropna().reset_index(drop=True)

# F. Training Utilities

def time_train_test_split(df, date_col, test_ratio=0.2):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def evaluate_regression(y_true, y_pred, model_name='Model'):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'model': model_name,
        'MAE': mae,
        'RMSE': rmse
    }


def train_xgb_task(df, date_col, feature_cols, target_col, model_name='Task Model'):
    train_df, test_df = time_train_test_split(df, date_col=date_col, test_ratio=0.2)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    eval_dict = evaluate_regression(y_test, y_pred, model_name=model_name)

    result_df = test_df[[date_col]].copy()
    result_df['actual'] = y_test.values
    result_df['predicted'] = y_pred

    return model, eval_dict, result_df
    

# G.Optimized Modeling

# -----------------------------
# TSK1 optimized
# -----------------------------
tsk1_opt_features = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday',
    'avg_temp', 'discount_rate', 'loyalty_ratio',
    'order_count', 'avg_unit_price', 'total_quantity', 'avg_order_value',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'lag_1', 'lag_7', 'lag_14',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7'
]

# add category share cols dynamically
tsk1_share_cols = [c for c in tsk1_opt.columns if c.endswith('_share')]
tsk1_opt_features = tsk1_opt_features + tsk1_share_cols

model_tsk1_opt, tsk1_opt_eval, tsk1_opt_result = train_xgb_task(
    df=tsk1_opt,
    date_col='date',
    feature_cols=tsk1_opt_features,
    target_col='daily_revenue',
    model_name='TSK1 Optimized Daily Revenue'
)

# -----------------------------
# TSK2 baseline keep
# -----------------------------
tsk2_features = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday', 'avg_temp',
    'discount_rate', 'loyalty_ratio', 'daily_revenue', 'avg_order_value',
    'lag_orders_1', 'lag_orders_7', 'lag_orders_14',
    'rolling_orders_7', 'rolling_orders_std_7'
]

model_tsk2, tsk2_eval, tsk2_result = train_xgb_task(
    df=tsk2,
    date_col='date',
    feature_cols=tsk2_features,
    target_col='daily_orders',
    model_name='TSK2 Daily Orders'
)

# -----------------------------
# TSK3 optimized holiday fix
# -----------------------------
tsk3_features = [
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_peak_hour', 'is_holiday',
    'avg_temp', 'discount_rate', 'loyalty_ratio', 'hourly_revenue',
    'lag_1h', 'lag_24', 'lag_168',
    'rolling_mean_24', 'rolling_std_24'
]

model_tsk3_opt, tsk3_opt_eval, tsk3_opt_result = train_xgb_task(
    df=tsk3_opt,
    date_col='datetime_hour',
    feature_cols=tsk3_features,
    target_col='hourly_orders',
    model_name='TSK3 Optimized Hourly Orders'
)

# -----------------------------
# TSK4 optimized
# -----------------------------
tsk4_model = pd.get_dummies(tsk4_opt, columns=['product_category'], drop_first=True)

tsk4_base_features = [
    'day_of_week', 'month', 'day_of_month', 'week_of_year',
    'is_weekend', 'is_holiday', 'avg_temp',
    'discount_rate', 'loyalty_ratio', 'category_revenue',
    'avg_unit_price',
    'lag_cat_1', 'lag_cat_7', 'lag_cat_14',
    'rolling_cat_mean_7', 'rolling_cat_std_7'
]
tsk4_cat_cols = [c for c in tsk4_model.columns if c.startswith('product_category_')]
tsk4_features = tsk4_base_features + tsk4_cat_cols

model_tsk4_opt, tsk4_opt_eval, tsk4_opt_result = train_xgb_task(
    df=tsk4_model,
    date_col='date',
    feature_cols=tsk4_features,
    target_col='category_demand',
    model_name='TSK4 Optimized Category Demand'
)

model_summary_opt = pd.DataFrame([
    tsk1_opt_eval,
    tsk2_eval,
    tsk3_opt_eval,
    tsk4_opt_eval
])

print(model_summary_opt)

# H.Streamlit Dashboard

st.set_page_config(
    page_title='Coffee Shop Sales Intelligence Dashboard',
    page_icon='☕',
    layout='wide'
)

st.title('☕ Coffee Shop Sales Intelligence Dashboard')
st.markdown('Sales monitoring, demand analysis, and forecasting for coffee shop operations.')

# -----------------------------
# Sidebar
# -----------------------------
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

# -----------------------------
# Global KPI
# -----------------------------
total_revenue = tsk['total_amount'].sum()
total_orders = tsk['transaction_id'].nunique()
total_quantity = tsk['quantity'].sum()
avg_order_value_global = total_revenue / total_orders

# -----------------------------
# Overview
# -----------------------------
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


# -----------------------------
# Task 1
# -----------------------------
elif page == 'Task 1 - Daily Revenue Forecast':
    st.subheader('Task 1: Daily Revenue Forecast')

    st.metric('Model MAE', f"{tsk1_opt_eval['MAE']:.3f}")
    st.metric('Model RMSE', f"{tsk1_opt_eval['RMSE']:.3f}")

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


# -----------------------------
# Task 2
# -----------------------------
elif page == 'Task 2 - Daily Orders Forecast':
    st.subheader('Task 2: Daily Orders Forecast')

    st.metric('Model MAE', f"{tsk2_eval['MAE']:.3f}")
    st.metric('Model RMSE', f"{tsk2_eval['RMSE']:.3f}")

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


# -----------------------------
# Task 3
# -----------------------------
elif page == 'Task 3 - Hourly Orders Forecast':
    st.subheader('Task 3: Hourly Orders Forecast')

    st.metric('Model MAE', f"{tsk3_opt_eval['MAE']:.3f}")
    st.metric('Model RMSE', f"{tsk3_opt_eval['RMSE']:.3f}")

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


# -----------------------------
# Task 4
# -----------------------------
elif page == 'Task 4 - Category Demand Forecast':
    st.subheader('Task 4: Category Demand Forecast')

    st.metric('Model MAE', f"{tsk4_opt_eval['MAE']:.3f}")
    st.metric('Model RMSE', f"{tsk4_opt_eval['RMSE']:.3f}")

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


# -----------------------------
# Model Evaluation
# -----------------------------
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