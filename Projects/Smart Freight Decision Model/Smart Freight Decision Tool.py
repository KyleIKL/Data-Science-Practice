
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
# 对每个类别特征进行编码
for col in cat_features:
    # 为每个类别特征创建新的 LabelEncoder 实例
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # 对当前列进行编码
    
    # 打印每个类别特征的映射关系
    print(f"Mapping for {col}:")
    print(f"Original Categories: {le.classes_}")  # 显示原始类别标签
    mapping = dict(zip(le.classes_, range(len(le.classes_))))  # 映射为字典
    print(f"Corresponding encoded values: {mapping}\n")
# Verify if encoding is successful
print(data[cat_features].head())

# IV.Feature selection and model training

print(data.dtypes)


data = data.drop(['Shipment_ID', 'Date'], axis=1)
X = data.drop('Transport_Mode', axis=1) 
y = data['Transport_Mode']  


print(X.shape, y.shape)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = StandardScaler()

X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)

xgb_classifier.fit(X_train, y_train)

y_pred_xgb = xgb_classifier.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [3, 5, 7],            # 树的最大深度
    'learning_rate': [0.01, 0.1, 0.2], # 学习率
    'subsample': [0.8, 0.9, 1.0],      # 训练样本的比例
    'colsample_bytree': [0.8, 0.9, 1.0], # 特征列的比例
    'min_child_weight': [1, 2, 3],     # 子节点的最小权重
    'gamma': [0, 0.1, 0.2]             # 剪枝参数
}


xgb_classifier = xgb.XGBClassifier(random_state=42)

grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# 使用最佳参数训练模型
best_xgb_classifier = grid_search.best_estimator_

# 预测
y_pred_best_xgb = best_xgb_classifier.predict(X_test)

# 评估模型
from sklearn.metrics import accuracy_score, classification_report

print("Best XGBoost Accuracy:", accuracy_score(y_test, y_pred_best_xgb))
print("Best XGBoost Classification Report:")
print(classification_report(y_test, y_pred_best_xgb))

# 保存训练好的模型和标准化器
from joblib import dump

# 假设 classifier 是你的训练好的模型，scaler 是标准化器
dump(best_xgb_classifier, 'best_xgb_model.joblib')  # 保存模型
dump(scaler, 'scaler.joblib')  # 保存标准化器

