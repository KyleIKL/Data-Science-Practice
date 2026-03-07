
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

# III.PRE
# Numerical features
num_features = ['Distance_km','Weight_MT','Fuel_Price_Index','Geopolitical_Risk_Score','Carrier_Reliability_Score']
# Categorical features
cat_features = ['Transport_Mode','Product_Category','Weather_Condition','Origin_Port','Destination_Port']
# Initialize LabelEncoder
le = LabelEncoder()
# Encode categorical features
for col in cat_features:
    data[col] = le.fit_transform(data[col])
# Verify if encoding is successful
print(data[cat_features].head())

# IV.Feature selection and model training
# 检查数据类型
print(data.dtypes)  # 查看所有列的数据类型

# 将所有非数值列从 X 中移除
data = data.drop(['Shipment_ID', 'Date'], axis=1)
X = data.drop('Transport_Mode', axis=1)  # 删除目标列
y = data['Transport_Mode']  # 目标列

# 检查数据形状和类型
print(X.shape, y.shape)  # 确保 X 是二维数据，y 是一维数据

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查划分后的数据形状
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查划分后的数据形状
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



# 初始化 StandardScaler
scaler = StandardScaler()

# 标准化数值特征
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])


# 初始化 XGBoost 分类器
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)

# 训练模型
xgb_classifier.fit(X_train, y_train)

# 预测
y_pred_xgb = xgb_classifier.predict(X_test)

# 评估模型
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))