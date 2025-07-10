import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# load the dataset (make sure hr.csv is there)
df = pd.read_csv("hr.csv")

# drop useless cols (not needed for prediction)
df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True, errors='ignore')

# map target col Attrition to 0,1   (Yes = 1)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# encode other text cols (catgorical ones)
label_encoders = {}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # in case needed again but optional

# split input and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model (using rf, 100 trees)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# make model/ folder if not there
os.makedirs("model", exist_ok=True)

# save model in that folder (as .pkl)
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model saved at model/churn_model.pkl")
