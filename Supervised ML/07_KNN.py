# -------------------- Libraries --------------------
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# -------------------- Load Dataset --------------------
df = sns.load_dataset("iris")

# =====================================================
# 🔹 Part 1: KNN CLASSIFIER
# =====================================================

# Features and Target
X = df.drop("species", axis=1)
y = df["species"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("✅ KNN Classifier Evaluation")
print("Accuracy          :", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =====================================================
# 🔹 Part 2: KNN REGRESSOR
# =====================================================

# Use the same dataset but predict sepal_length (numeric column)
X_reg = df.drop("sepal_length", axis=1)
# Drop 'species' because it's categorical
X_reg = X_reg.drop("species", axis=1)
y_reg = df["sepal_length"]

# Train-Test Split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Feature Scaling
Xr_train = scaler.fit_transform(Xr_train)
Xr_test = scaler.transform(Xr_test)

# Model Training
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(Xr_train, yr_train)

# Predictions
yr_pred = reg.predict(Xr_test)

# Evaluation
print("\n✅ KNN Regressor Evaluation")
print("Mean Absolute Error      :", mean_absolute_error(yr_test, yr_pred))
print("Mean Squared Error       :", mean_squared_error(yr_test, yr_pred))
print("R² Score                 :", r2_score(yr_test, yr_pred))
