import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re

# Load .env variabls
load_dotenv()

#  Configuer Gmini API Key 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# print("api =", os.getenv("GEMINI_API_KEY"))

#  Initializ models
model = genai.GenerativeModel("gemini-2.0-flash")
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  


# Set up ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("employee_retention")


def clean_response(text):
    text = re.sub(r"Okay,.*?(?=1\.)", "", text, flags=re.DOTALL).strip()
    actions = re.findall(r"(?:\d\.\s*)?\*\*Action:\*\*([\s\S]*?)(?:\n\s*\d\.|\Z)", text)
    if not actions:
        actions = re.findall(r"\d\.\s+(.*?)(?=\n\d\.|\Z)", text, flags=re.DOTALL)
    cleaned_actions = []
    for a in actions:
        a = re.sub(r"[\*\_]", "", a).strip()
        a = re.sub(r"^(Action:)\s*", "", a)
        a = re.sub(r"\n+", " ", a)
        a = re.sub(r"\s{2,}", " ", a)
        a = re.split(r"Rationale\:", a)[0].strip()
        cleaned_actions.append(a)
    if not cleaned_actions:
        lines = [l for l in text.split("\n") if l.strip()]
        cleaned_actions = [re.sub(r"^[\*\-\d\.]+\s*", "", l).strip() for l in lines if l.strip()]
    cleaned_actions = [a for a in cleaned_actions if a]
    return "\n".join(f"{i+1}. {a}" for i, a in enumerate(cleaned_actions))


def generate_recommendation(row):
    query = f"""
    job role: {row.get('JobRole', '')}
    job satisfaction: {row.get('JobSatisfaction', '')}
    overtime: {row.get('OverTime', '')}
    monthly income: {row.get('MonthlyIncome', '')}
    years at company: {row.get('YearsAtCompany', '')}
    """.replace("\n", " ").strip()

    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    documents = results.get('documents', [])
    if documents and documents[0]:
      context = "\n\n".join(documents[0])
    else:
      context = "No relevant HR knowledge found in the database."

    # ✅ Print the context retrieved from ChromaDB
    print("\n==== Retrieved Context from ChromaDB ====\n")
    print(context)
    print("\n==== End of Context ====\n")


    prompt = f"""
You are an HR expert. The following employee is at risk of leaving the company.

Employee Info:
- Age: {row.get('Age', 'N/A')}
- Job Role: {row.get('JobRole', 'N/A')}
- Job Satisfaction: {row.get('JobSatisfaction', 'N/A')}
- Overtime: {row.get('OverTime', 'N/A')}
- Monthly Income: {row.get('MonthlyIncome', 'N/A')}
- Years at Company: {row.get('YearsAtCompany', 'N/A')}

Relevant HR Knowledge:
{context}

👉 Based on this, **suggest exactly 2 distinct and actionable HR strategies** to retain this employee.  
👉 Format each as:  
1. **Action:** <strategy>  
   **Rationale:** <why it will help>  
2. **Action:** <strategy>  
   **Rationale:** <why it will help>
"""

    try:
        response = model.generate_content(prompt)
        cleaned = clean_response(response.text.strip())
        return cleaned
    except Exception as e:
        return f"Gemini Error: {str(e)}"
