
# AI Intern Assignment — Use Case 2  
### 👤 Employee Attrition Risk Prediction & HR Recommendation Agent  
**Author**: Tushar Rathor  
**Tech Stack**: Python · Flask · scikit-learn · Google Gemini API

---

## 📌 Overview

This project addresses **Use Case 2** from the AI Intern assignment:  
> ⚠️ **Predict the risk of employee attrition using ML**, and use a **Gemini AI agent** to recommend HR actions for at-risk cases.
Live Link: https://employee-attrition-predictor-ycoi.onrender.com/
---

## 🎯 What It Does

- ✅ Predicts whether an employee is likely to leave using a trained ML model
- ✅ Uses Google Gemini (`gemini-1.5-flash`) to suggest 2 HR actions for each risky employee
- ✅ Displays all predictions and recommendations in a clean web interface
- ✅ Includes error handling and a loading spinner for better UX

---

## 📁 Folder Structure

```
AI_Intern_Assignment/
├── app.py                  # Main Flask app
├── train.py                # Model training script
├── churn_model.pkl         # Trained Random Forest model (~3.7 MB)
├── utils.py                # Gemini recommendation logic
├── test_employees.csv      # Sample input file
├── requirements.txt        # Required packages
├── templates/
│   ├── index.html          # Upload form
│   ├── results.html        # Prediction + Gemini output
│   └── error.html          # Error page for invalid CSVs
├── static/
│   └── style.css (optional custom CSS)
└── README.md
```

---

## 🧠 Model Info

- **Model Used**: Random Forest Classifier (scikit-learn)
- **Target**: `Attrition` → `Yes`/`No`
- **Features**: Age, Job Role, Monthly Income, Overtime, Years at Company, etc.
- **Imbalance Handling**: SMOTE
- ✅ **Model already included** as `churn_model.pkl`

---

## 🔁 Retrain the Model (Optional)

If you want to retrain the model using your own data:

### ✅ Step 1: Prepare the CSV

- Your dataset should contain an `Attrition` column with `Yes`/`No`
- Example filename: `hr.csv`

### ✅ Step 2: Use `train.py`

```bash
python train.py
```

This will:
- Train the model
- Save it as `churn_model.pkl`

### 💡 Custom CSV Support

If you're using a differently named file, edit this line in `train.py`:

```python
df = pd.read_csv("your_file.csv")  # Change this if not using hr.csv
```

---

## 🧪 How to Run the Web App

### ✅ 1. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ 2. Run the Flask app

```bash
python app.py
```

### ✅ 3. Upload a valid CSV

Use the provided `test_employees.csv` or your own file with matching columns.

---

## 🤖 Gemini Recommendations

- Employees flagged as at-risk get **2 HR suggestions** from **Gemini AI**
- Uses `google-generativeai` and model: `gemini-1.5-flash`
- Requires a valid API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

## ❗ Error Handling

If a user uploads an invalid CSV:
- They are shown a message:  
  > "Upload a valid file or refer to the README for what is a valid CSV file."

---

## 🎥 Optional Enhancements

- ✅ Spinner while processing
- ✅ Clean error display
- ✅ Styled UI with upload buffer and result table
- ✅ “Go to Home” button for retry

---

## ✅ To Push on GitHub

No need to ignore the model — it's under 100 MB and included.  
Still, here’s a safe `.gitignore` suggestion:

---

## 📄 .gitignore

```gitignore
# Python cache
__pycache__/
*.pyc

# IDE configs
.vscode/
.idea/

# OS files
.DS_Store

# Optional: ignore large datasets
*.csv

# Keep the trained model
!churn_model.pkl
```

---

## ✍️ Credits

- Dataset: IBM HR Analytics  
- ML: scikit-learn  
- LLM: Google Gemini  
- Dev: [Tushar Rathor](https://github.com/Tushar-6969)

---

## 🏁 Conclusion

This project showcases how machine learning and AI agents like Gemini can be combined to:
- Predict employee behavior
- Support HR decision-making
- Deliver smart insights in a clean UI

🎯 Assignment-ready. Clean. Functional. AI-powered.

---
