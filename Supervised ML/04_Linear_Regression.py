# Linear Regression on Tips Dataset

# Import libraries
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Load dataset
df = sns.load_dataset("tips")

# Features and target
X = df[["total_bill"]]
y = df["tip"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=33
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for a new value (must be scaled)
new_value = scaler.transform([[1500]])
print("Prediction for total_bill = 1500:", model.predict(new_value)[0])

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
print("R²   :", r2_score(y_test, y_pred))
