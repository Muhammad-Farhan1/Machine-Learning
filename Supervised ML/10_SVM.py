"""
========================================================
Support Vector Machine (SVM) - Classification & Regression
========================================================

SVM (Support Vector Machine) is a supervised ML algorithm
used for both classification and regression tasks.

- In classification, SVM finds the "optimal hyperplane"
  that separates data points of different classes with the
  maximum margin.
- In regression (SVR), it tries to fit the data within a
  tube (epsilon margin) while minimizing prediction errors.

Kernels (linear, polynomial, RBF, etc.) allow SVM to handle
both linear and non-linear data effectively.
========================================================
"""

# ================== Imports ===================
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
#                 SVM CLASSIFIER
# =====================================================
# Load dataset (Iris dataset)
df = sns.load_dataset('iris')
print("First 5 rows of Iris dataset:\n", df.head())

# Features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM Classifier with RBF Kernel
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n===== SVM Classification Results (Iris) =====")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =====================================================
#                 SVM REGRESSOR
# =====================================================
# Load dataset (Tips dataset)
df = sns.load_dataset('tips')
print("\nFirst 5 rows of Tips dataset:\n", df.head())

# Features (X) and target (y)
X = df.drop('tip', axis=1)
y = df['tip']

# Define numeric & categorical features
numeric_features = ['total_bill', 'size']
categorical_features = ['sex', 'smoker', 'day', 'time']

# Preprocessing (Scaling + One-Hot Encoding)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Pipeline: Preprocessing + SVR Model
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svr', SVR(kernel='rbf', C=100, gamma=0.1))  # Tuned SVR
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
print("\n===== SVM Regression Results (Tips) =====")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))
