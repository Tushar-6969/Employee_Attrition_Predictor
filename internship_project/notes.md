Project Title: Employee Attrition Predictor with RAG-enhanced Recommendations

Short Introduction:
"I built a web-based application where HR professionals can upload employee data and predict who is likely to leave the organization. But instead of just predicting, it also suggests actionable steps using Retrieval-Augmented Generation (RAG) powered by Gemini AI and ChromaDB."

When they ask for more details:

✅ Step 1 — Prediction Model:
"I trained a Random Forest Classifier using HR data (like Age, Job Role, Satisfaction, Income). It predicts employee attrition: 'Yes' or 'No'."

✅ Step 2 — Knowledge Base Embedding:
"I collected HR retention strategies, stored them as Markdown, and converted them into vector embeddings using SentenceTransformer. These embeddings are stored in ChromaDB."

✅ Step 3 — RAG (Retrieval-Augmented Generation):
"When an employee is predicted to leave, my app queries ChromaDB using their details as input. The two most relevant retention strategies from the knowledge base are retrieved."

✅ Step 4 — Gemini AI Integration:
"Gemini receives both employee details and the retrieved HR strategies as context, then generates a tailored recommendation."

✅ Tech Stack Summary:

Python (Flask)

Scikit-learn

ChromaDB

Google Gemini API

SentenceTransformer for embeddings

HTML/CSS frontend with Flask templates

If they ask ‘Why RAG?’
"Instead of relying on generic AI answers, RAG allows my app to stay consistent with company-specific HR policies and retention strategies. It grounds the AI’s recommendations on predefined business knowledge."