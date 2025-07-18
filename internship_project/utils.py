import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Set Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Set up ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("employee_retention")

# Embedding model for queries
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_recommendation(row):
    query = f"""
    Job Role: {row.get('JobRole', '')}
    Job Satisfaction: {row.get('JobSatisfaction', '')}
    OverTime: {row.get('OverTime', '')}
    Monthly Income: {row.get('MonthlyIncome', '')}
    Years at Company: {row.get('YearsAtCompany', '')}
    """.replace("\n", " ").strip()

    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve top 2 relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    documents = results.get('documents', [])
    if documents and documents[0]:
        context = "\n\n".join(documents[0])
    else:
        context = "No relevant HR knowledge found in the database."

    # Debug: Print retrieved context
    print("🔍 Retrieved Context from ChromaDB:")
    print(context[:500])  # Show first 500 characters only

    prompt = f"""
    This employee is predicted to leave.

    Employee Info:
    - Age: {row.get('Age', 'N/A')}
    - Job Role: {row.get('JobRole', 'N/A')}
    - Job Satisfaction: {row.get('JobSatisfaction', 'N/A')}
    - OverTime: {row.get('OverTime', 'N/A')}
    - Monthly Income: {row.get('MonthlyIncome', 'N/A')}
    - Years at Company: {row.get('YearsAtCompany', 'N/A')}

    Relevant HR Knowledge Base:
    {context}

    Using the above knowledge base, suggest 2 HR actions to retain this employee.
    """

    print(" DEBUG PROMPT:")
    print(prompt[:500])  # Print only first 500 characters

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {str(e)}"
