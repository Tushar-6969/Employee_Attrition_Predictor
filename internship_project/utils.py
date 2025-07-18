import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import re

# set gemini api key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# initialize gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# set up chromadb client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("employee_retention")

# embedding model for queries
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_response(text):
    """
    Extract only the core 'Action' steps from Gemini's verbose output and return as a clean, numbered list.
    """
    # Remove preamble if present
    text = re.sub(r"Okay,.*?(?=1\.)", "", text, flags=re.DOTALL).strip()

    # Pattern to extract "**Action:**" or "**Action:**Text" blocks, possibly after a numbered heading
    actions = re.findall(r"(?:\d\.\s*)?\*\*Action:\*\*([\s\S]*?)(?:\n\s*\d\.|\Z)", text)
    if not actions:
        # Fallback: try to extract numbered items if no "Action" sections
        actions = re.findall(r"\d\.\s+(.*?)(?=\n\d\.|\Z)", text, flags=re.DOTALL)

    cleaned_actions = []
    for a in actions:
        # Remove any markdown formatting and extra whitespace
        a = re.sub(r"[\*\_]", "", a).strip()
        # Optional: remove "Action:" if GPT adds it without bold
        a = re.sub(r"^(Action:)\s*", "", a)
        # Remove unnecessary newlines and collapse whitespace
        a = re.sub(r"\n+", " ", a)
        a = re.sub(r"\s{2,}", " ", a)
        # Remove trailing rationale, if any (heuristic: if text like "Rationale:")
        a = re.split(r"Rationale\:", a)[0].strip()
        cleaned_actions.append(a)

    # If got nothing (Gemini output structure changes), fall back to whole text cleaned up
    if not cleaned_actions:
        lines = [l for l in text.split("\n") if l.strip()]
        cleaned_actions = [re.sub(r"^[\*\-\d\.]+\s*", "", l).strip() for l in lines if l.strip()]

    # Remove empties and collate as a numbered list
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

    # retrieve top 2 relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    documents = results.get('documents', [])
    if documents and documents[0]:
        context = "\n\n".join(documents[0])
    else:
        context = "no relevant hr knowledge found in the database."

    # debug print retrieved context
    print("🔍 retrieved context from chromadb:")
    print(context[:500])  # show first 500 characters only

    prompt = f"""
    this employee is predicted to leave.

    employee info:
    - age: {row.get('Age', 'N/A')}
    - job role: {row.get('JobRole', 'N/A')}
    - job satisfaction: {row.get('JobSatisfaction', 'N/A')}
    - overtime: {row.get('OverTime', 'N/A')}
    - monthly income: {row.get('MonthlyIncome', 'N/A')}
    - years at company: {row.get('YearsAtCompany', 'N/A')}

    relevant hr knowledge base:
    {context}

    using the above knowledge base, suggest 2 hr actions to retain this employee.
    """

    print("debug prompt:")
    print(prompt[:500])  # print only first 500 characters

    try:
        response = model.generate_content(prompt)
        cleaned_text = clean_response(response.text.strip())
        return cleaned_text
    except Exception as e:
        return f"gemini error: {str(e)}"
