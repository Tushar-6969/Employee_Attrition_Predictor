# train_churn_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Load dataset
df = pd.read_csv("hr.csv")

# Drop columns that aren't useful (customize as needed)
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True, errors='ignore')

# Encode target label
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Optional: save encoders if you need for future use

# Split features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model/ folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model to disk
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model/churn_model.pkl")
