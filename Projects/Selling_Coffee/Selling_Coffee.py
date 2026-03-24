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
tsk1 = pd.read_csv(r"E:\Project95\Projects\Selling_Coffee\coffee_shop_sales.csv")
# TST
tsk1.head()
tsk1.info()

# Sort data into four sets according to accomplish four main tasks:
# Task 1: Predict daily revenue

