# ================================
# Logistic Regression on Titanic Dataset
# ================================

# Import libraries
import pandas as pd
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------------
# Step 1: Load dataset
# -------------------------------
df = sns.load_dataset('titanic')

# -------------------------------
# Step 2: Data Cleaning & Preprocessing
# -------------------------------

# Drop column with too many missing values
df.drop('deck', axis=1, inplace=True)

# Fill missing numerical values with mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing categorical values with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Encode categorical variables into numeric using LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        df[col] = le.fit_transform(df[col])

# -------------------------------
# Step 3: Define Features (X) and Target (y)
# -------------------------------
X = df.drop('survived', axis=1)   # Features
y = df['survived']                # Target variable

# -------------------------------
# Step 4: Split dataset into Train & Test sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# -------------------------------
# Step 5: Train Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Step 6: Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 7: Model Evaluation (Classification Metrics)
# -------------------------------
print("Model Performance on Test Data")
print("-" * 40)
print("Accuracy        :", accuracy_score(y_test, y_pred))
print("Precision       :", precision_score(y_test, y_pred))
print("F1-Score        :", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
