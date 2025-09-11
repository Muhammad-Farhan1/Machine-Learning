# =====================================================
# Import Libraries
# =====================================================
import numpy as np
import seaborn as sns
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# =====================================================
# ============== Random Forest Classifier =============
# =====================================================

# Load Titanic dataset
df = sns.load_dataset('titanic')

# --- Data Cleaning ---
df.drop('deck', axis=1, inplace=True)                     # Drop column with many missing values
df['age'].fillna(df['age'].mean(), inplace=True)          # Fill missing ages with mean
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)  # Fill missing embarkation with mode
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Features (X) and Target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Separate Numeric and Categorical Features
numeric_features = ['pclass', 'age', 'sibsp', 'fare']
categorical_features = ['embarked', 'class', 'who', 'embark_town', 'alone']

# Preprocessing: Scale numeric + One-hot encode categorical
preprocessing = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with preprocessing + RandomForestClassifier
clf_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('random_forest', RandomForestClassifier(n_estimators=300, random_state=42))
])

# Train model
clf_pipeline.fit(X_train, y_train)

# Predictions
y_pred = clf_pipeline.predict(X_test)

# --- Classification Metrics ---
print("=== Random Forest Classifier Metrics ===")
print("Accuracy       :", accuracy_score(y_test, y_pred))
print("Precision      :", precision_score(y_test, y_pred))
print("Recall         :", recall_score(y_test, y_pred))
print("F1 Score       :", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =====================================================
# ============== Random Forest Regressor ==============
# =====================================================

# Load Tips dataset
df = sns.load_dataset('tips')

# Features (X) and Target (y)
X = df.drop('tip', axis=1)
y = df['tip']

# Separate Numeric and Categorical Features
numeric_features = ['total_bill', 'size']
categorical_features = ['sex', 'smoker', 'day', 'time']

# Preprocessing: Scale numeric + One-hot encode categorical
preprocessing = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with preprocessing + RandomForestRegressor
reg_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('random_forest', RandomForestRegressor(n_estimators=500, random_state=42))
])

# Train model
reg_pipeline.fit(X_train, y_train)

# Predictions
y_pred = reg_pipeline.predict(X_test)

# --- Regression Metrics ---
print("\n=== Random Forest Regressor Metrics ===")
print("MAE   :", mean_absolute_error(y_test, y_pred))
print("MSE   :", mean_squared_error(y_test, y_pred))
print("RMSE  :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAPE  :", mean_absolute_percentage_error(y_test, y_pred))
print("R²    :", r2_score(y_test, y_pred))
