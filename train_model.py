
# ==========================================
# SALES PREDICTION USING MACHINE LEARNING
# ==========================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. LOAD DATASET
data = pd.read_csv("Dataset/Advertising.csv")

# Remove unnecessary index column, if present
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

print("\nFIRST FIVE ROWS:")
print(data.head())

# 2. DATASET INFORMATION
print("\nDATASET SHAPE:")
print(data.shape)

print("\nCOLUMN NAMES:")
print(data.columns.tolist())

print("\nDATA TYPES:")
print(data.dtypes)

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# 3. CREATE OUTPUT FOLDER
os.makedirs("model", exist_ok=True)

# 4. DATA VISUALIZATION
sns.pairplot(data)
plt.savefig("model/pairplot.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("model/correlation.png")
plt.close()

# 5. SELECT FEATURES AND TARGET
X = data[["TV", "radio", "newspaper"]]
y = data["sales"]

# 6. SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. TRAIN LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# 8. MAKE PREDICTIONS
y_pred = model.predict(X_test)

# 9. EVALUATE MODEL
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print("Mean Absolute Error:", round(mae, 4))
print("Mean Squared Error:", round(mse, 4))
print("Root Mean Squared Error:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

print("\nMODEL INTERCEPT:", model.intercept_)
print("MODEL COEFFICIENTS:", model.coef_)

# 10. ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.tight_layout()
plt.savefig("model/actual_vs_predicted.png")
plt.close()

# 11. RESIDUAL GRAPH
residuals = y_test - y_pred

plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("model/residual_plot.png")
plt.close()

# 12. SAVE MODEL
joblib.dump(model, "model/sales_prediction_model.pkl")

# 13. SAVE TEST RESULTS
results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})
results.to_csv("model/prediction_results.csv", index=False)

print("\nModel saved successfully!")
print("Graphs and prediction results saved in model folder.")
