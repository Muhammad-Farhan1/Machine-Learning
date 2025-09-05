# Import libraries
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# Load the sample dataset
df = sns.load_dataset("tips")



# -------------------------------
# 1. Label Encoding
# -------------------------------
# Converts categorical values into numeric labels (e.g., 'Lunch' → 0, 'Dinner' → 1)
le = LabelEncoder()
df['time_encoded'] = le.fit_transform(df['time'])



# -------------------------------
# 2. Ordinal Encoding
# -------------------------------
# Converts categorical values into ordered numeric values.
# Here, we define the order of days: Thur < Fri < Sat < Sun
ode = OrdinalEncoder(categories=[['Thur', 'Fri', 'Sat', 'Sun']])
df['day_encoded'] = ode.fit_transform(df[['day']])



# -------------------------------
# 3. One Hot Encoding
# -------------------------------
# Converts categorical values into multiple binary columns (dummy variables).
# Example: 'Male' → [1,0], 'Female' → [0,1]
ohe = OneHotEncoder()
onehot_encoded = ohe.fit_transform(df[['sex']]).toarray()

# Create a DataFrame for one-hot encoded values
ohe_df = pd.DataFrame(onehot_encoded, columns=ohe.get_feature_names_out(['sex']))

# Concatenate with the original DataFrame
df = pd.concat([df, ohe_df], axis=1)

# Display the first few rows
print(df.head())
