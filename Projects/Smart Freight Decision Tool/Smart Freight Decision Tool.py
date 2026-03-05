
# I.Import necssary libraries

# Import libraries for data preprocessing
import pandas as pd        
import numpy as np         
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# Import libraries for visualization
import matplotlib.pyplot as plt  
import seaborn as sns    
# Import libraries for machine learning models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
# Import libraries for evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# II.Data loading and preprocessing
# Load the dataset and check missing values
data = pd.read_csv(r"E:\Project95\Projects\Smart Freight Decision Tool\global_supply_chain_risk_2026.csv")
print(data.head())
print(data.isnull().sum())

# III.Exploratory Data Analysis (EDA)
# Visualize
sns.scatterplot(x=data['Distance_km'], y=data['Geopolitical_Risk_Score'], hue=data['Transport_Mode'])
plt.show()
# 


