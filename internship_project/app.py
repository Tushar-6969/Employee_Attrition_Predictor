from flask import Flask, render_template, request
import pandas as pd
import pickle
from utils import generate_recommendation
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
with open("./models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']

        # Check file type
        if not file.filename.endswith('.csv'):
            raise ValueError("Only CSV files are supported.")

        df = pd.read_csv(file)

        # Drop unused columns if present
        df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True, errors='ignore')

        # Handle EmployeeID safely
        df_input = df.drop(columns=['EmployeeID'], errors='ignore')

        # Load the expected column names (based on model training)
        expected_columns = model.feature_names_in_

        # Ensure all required columns are present
        missing_cols = [col for col in expected_columns if col not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Filter only relevant columns, drop extras
        df_input = df_input[expected_columns]

        # Encode object-type (categorical) columns
        for col in df_input.select_dtypes(include='object'):
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col].astype(str))

        # Make prediction
        predictions = model.predict(df_input)
        df['Attrition_Prediction'] = predictions

        # Generate recommendation
        output = []
        for i, row in df.iterrows():
            rec = generate_recommendation(row) if row['Attrition_Prediction'] == 1 else "Employee not at risk."
            output.append({
                'id': row.get('EmployeeID', f"Emp-{i+1}"),
                'prediction': 'Yes' if row['Attrition_Prediction'] == 1 else 'No',
                'recommendation': rec
            })

        output.sort(key=lambda x: x['prediction'], reverse=True)
        return render_template('results.html', output=output)

    except ValueError as ve:
        return render_template('error.html', message=str(ve))
    except Exception as e:
        print(f"⚠️ Unexpected Error: {str(e)}")
        return render_template('error.html', message="Invalid file or format. Please upload a correct CSV matching the required schema.")

if __name__ == '__main__':
    app.run()  # Default: http://127.0.0.1:5000
