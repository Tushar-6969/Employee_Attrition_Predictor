# ai intern assignment — use case 2

### 👤 employee attrition risk prediction + rag-enhanced hr recommendation agent

author tushar rathor
technology stack python · flask · scikit-learn · chromadb · sentence transformers · google gemini api

---

##  overview

this project addresses use case 2 from the ai intern assignment

predict employee attrition risk using ml and use a rag-enhanced gemini ai agent to recommend hr actions for at-risk employees

---

live link [https://employee-attrition-predictor-ycoi.onrender.com](https://employee-attrition-predictor-ycoi.onrender.com)

##  what it does

* predicts whether an employee is likely to leave using a trained ml model
* retrieves hr retention strategies from a local chromadb knowledge base using rag (retrieval augmented generation)
* combines retrieved strategies with google gemini to suggest 2 hr actions for each risky employee
* displays all predictions and recommendations in a clean web interface
* includes error handling and a loading spinner for better user experience

---

## 📁 folder structure

```
ai_intern_assignment/
├── app.py                  main flask app
├── train.py                model training script
├── churn_model.pkl         trained random forest model
├── utils.py                rag + gemini recommendation logic
├── embed.py                rag content embedding script
├── knowledge_base/
│   └── content.md          hr strategies markdown file
├── chroma_db/              local rag database
├── requirements.txt        required packages
├── templates/
│   ├── index.html          upload form
│   ├── results.html        prediction + rag + gemini output
│   └── error.html          error page for invalid csvs
├── static/
│   └── style.css           optional custom css
└── readme.md
```

---

##  model info

* model used random forest classifier (scikit-learn)
* target attrition → yes/no
* features age, job role, monthly income, overtime, years at company, etc.
* imbalance handling smote
* model already included as churn\_model.pkl

---

##  retrain the model (optional)

### step 1 prepare the csv

your dataset should contain an attrition column with yes/no
example filename hr.csv

### step 2 use train.py

```bash
python train.py
```

this will

* train the model
* save it as churn\_model.pkl

### custom csv support

if you're using a differently named file, edit this line in train.py

```python
df = pd.read_csv("your_file.csv")  # change this if not using hr.csv
```

---

## how to run the web app

### step 1 install dependencies

```bash
pip install -r requirements.txt
```

### step 2 run the flask app

```bash
python app.py
```

### step 3 upload a valid csv

use the provided test\_employees.csv or your own file with matching columns

---

## 🤖 rag + gemini recommendations

* employees flagged as at-risk get 2 hr suggestions from gemini ai
* before calling gemini, top 2 related hr strategies are retrieved from chromadb
* uses sentence transformers all-minilm-l6-v2 for embeddings
* requires a valid api key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

---

##  error handling

if a user uploads an invalid csv

* they are shown a message

upload a valid file or refer to the readme for what is a valid csv file

---

## 🎥 optional enhancements

* spinner while processing
* clean error display
* styled ui with upload buffer and result table
* go to home button for retry
* rag setup using local markdown knowledge base

## to push on github

no need to ignore the model — it's under 100 mb and included
still, here’s a safe .gitignore suggestion

```
# python cache
__pycache__/
*.pyc

# ide configs
.vscode/
.idea/

# os files
.ds_store

# optional ignore large datasets
*.csv

# keep the trained model
!churn_model.pkl
```

---

## credits

* dataset ibm hr analytics
* ml scikit-learn
* llm google gemini
* rag chromadb + sentence transformers
* dev [https://github.com/tushar-6969](https://github.com/tushar-6969)

---

## conclusion

this project shows how ml + rag + ai agents like gemini can work together to

* predict employee behavior
* support hr decision-making
* deliver smart insights in a clean ui

assignment-ready. clean. functional. ai-powered.
