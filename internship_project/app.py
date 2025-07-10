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
        df = pd.read_csv(file)

        # Drop unused columns safely
        df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True, errors='ignore')
        df_input = df.drop(columns=['EmployeeID'], errors='ignore')

        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        for col in df_input.select_dtypes(include='object'):
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col])

        # Check if input matches model columns
        predictions = model.predict(df_input)
        df['Attrition_Prediction'] = predictions

        # Prepare output
        output = []
        for i, row in df.iterrows():
            if row['Attrition_Prediction'] == 1:
                rec = generate_recommendation(row)
            else:
                rec = "Employee not at risk."

            output.append({
                'id': row.get('EmployeeID', f"Emp-{i+1}"),
                'prediction': 'Yes' if row['Attrition_Prediction'] == 1 else 'No',
                'recommendation': rec
            })

        output.sort(key=lambda x: x['prediction'], reverse=True)
        return render_template('results.html', output=output)

    except Exception as e:
        print(f"⚠️ Error: {str(e)}")  # optional: log to console
        return render_template('error.html', message="Upload a valid file or refer to the README for what is a valid CSV file.")


if __name__ == '__main__':
    app.run(debug=True)
