import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ===============================
# 1. Load encoded dataset
# ===============================

file_path = input("Enter encoded CSV path: ").strip()

df = pd.read_csv(file_path, encoding="utf-8-sig")

print("Dataset loaded")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ===============================
# 2. Set target variable
# ===============================

target = "available_size_count"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")


# ===============================
# 3. Build features and target
# ===============================
# Remove size_count to avoid leakage / near-deterministic prediction

drop_cols = [target, "size_count"]

existing_drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=existing_drop_cols).copy()
y = df[target].copy()

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

print("\nFeature columns used:")
print(X.columns.tolist())

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)


# ===============================
# 4. Train-test split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# 5. Random Forest Regressor
# ===============================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)


# ===============================
# 6. Gradient Boosting Regressor
# ===============================

gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)

gb.fit(X_train, y_train)
pred_gb = gb.predict(X_test)


# ===============================
# 7. Evaluation
# ===============================

def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


rmse_rf, mae_rf, r2_rf = evaluate_regression(y_test, pred_rf)
rmse_gb, mae_gb, r2_gb = evaluate_regression(y_test, pred_gb)

results = pd.DataFrame({
    "Model": ["RandomForest", "GradientBoosting"],
    "RMSE": [rmse_rf, rmse_gb],
    "MAE": [mae_rf, mae_gb],
    "R2": [r2_rf, r2_gb]
})

print("\nModel performance")
print(results)


# ===============================
# 8. Plot model comparison
# ===============================

plt.figure(figsize=(8, 5))
plt.bar(results["Model"], results["RMSE"])
plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()


# ===============================
# 9. Plot actual vs predicted
# ===============================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_rf)

min_val = min(y_test.min(), pred_rf.min())
max_val = max(y_test.max(), pred_rf.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual available_size_count")
plt.ylabel("Predicted available_size_count")
plt.title("Actual vs Predicted (Random Forest)")
plt.tight_layout()
plt.show()


# ===============================
# 10. Plot residuals
# ===============================

residuals_rf = y_test - pred_rf

plt.figure(figsize=(6, 5))
plt.scatter(pred_rf, residuals_rf)
plt.axhline(0)
plt.xlabel("Predicted available_size_count")
plt.ylabel("Residual")
plt.title("Residual Plot (Random Forest)")
plt.tight_layout()
plt.show()


# ===============================
# 11. Feature importance
# ===============================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature importance")
print(importance)


plt.figure(figsize=(8, 6))
plt.barh(importance["feature"], importance["importance"])
plt.title("Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ===============================
# 12. Save outputs
# ===============================

results.to_csv("model_performance_updated.csv", index=False, encoding="utf-8-sig")
importance.to_csv("feature_importance_updated.csv", index=False, encoding="utf-8-sig")

prediction_df = pd.DataFrame({
    "actual": y_test.values,
    "pred_random_forest": pred_rf,
    "pred_gradient_boosting": pred_gb
})

prediction_df.to_csv("prediction_results_updated.csv", index=False, encoding="utf-8-sig")

print("\nAnalysis completed successfully.")
print("Files saved:")
print("1. model_performance_updated.csv")
print("2. feature_importance_updated.csv")
print("3. prediction_results_updated.csv")

# ===============================
# 13. Correlation heatmap
# ===============================

corr_df = df.select_dtypes(include=[np.number]).copy()

corr_matrix = corr_df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, aspect="auto")

plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ===============================
# 14. Target distribution
# ===============================

plt.figure(figsize=(8, 5))
plt.hist(y, bins=20)

plt.title("Distribution of available_size_count")
plt.xlabel("available_size_count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ===============================
# 15. Prediction error distribution
# ===============================

prediction_error_rf = y_test - pred_rf
prediction_error_gb = y_test - pred_gb

plt.figure(figsize=(8, 5))
plt.hist(prediction_error_rf, bins=20, alpha=0.6, label="Random Forest")
plt.hist(prediction_error_gb, bins=20, alpha=0.6, label="Gradient Boosting")

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 16. Improved actual vs predicted comparison
# ===============================

plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred_rf, label="Random Forest", alpha=0.7)
plt.scatter(y_test, pred_gb, label="Gradient Boosting", alpha=0.7)

min_val = min(y_test.min(), pred_rf.min(), pred_gb.min())
max_val = max(y_test.max(), pred_rf.max(), pred_gb.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="Perfect Prediction")

plt.xlabel("Actual available_size_count")
plt.ylabel("Predicted available_size_count")
plt.title("Actual vs Predicted Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 17. Top feature importance only
# ===============================

top_n = 8
top_importance = importance.head(top_n)

plt.figure(figsize=(8, 5))
plt.barh(top_importance["feature"], top_importance["importance"])

plt.title(f"Top {top_n} Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ===============================
# 18. Save correlation matrix
# ===============================

corr_matrix.to_csv("correlation_matrix.csv", encoding="utf-8-sig")

print("\nAdditional visualization completed.")
print("4. correlation_matrix.csv")