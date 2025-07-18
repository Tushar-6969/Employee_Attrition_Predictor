 Project Title:
Employee Attrition Predictor with RAG-Enhanced Recommendations

📝 Short Introduction:
I developed a web application where HR teams can upload employee data and predict which employees are at risk of leaving. Beyond just prediction, it suggests actionable HR strategies using Retrieval-Augmented Generation (RAG), combining Google Gemini AI with a local knowledge base stored in ChromaDB.

🔄 Project Workflow Explanation:

Step 1 — Prediction Model:

I trained a Random Forest Classifier using HR data (features like Age, Job Role, Satisfaction, Income).

It predicts the attrition risk as "Yes" or "No" for each employee.

Tools: scikit-learn, pandas

Step 2 — Building Knowledge Base Embeddings:

Collected HR retention strategies and saved them in Markdown format.

Converted that text into vector embeddings using SentenceTransformer (all-MiniLM-L6-v2).

Stored embeddings in ChromaDB as a local vector database.

Tools: ChromaDB, SentenceTransformer

Step 3 — Retrieval-Augmented Generation (RAG):

When an employee is marked as "at risk," the app queries ChromaDB using their profile details.

It fetches the top 2 most relevant HR strategies from the knowledge base.

Tools: ChromaDB client, SentenceTransformer for query embedding

Step 4 — Gemini AI Integration:

The app sends both employee information and the retrieved HR strategies to Gemini (gemini-2.0-flash model).

Gemini generates 2 specific HR recommendations tailored to that employee.

Tools: Google Generative AI Python SDK

💻 Tech Stack Summary:

Python (Flask) → Web backend

HTML/CSS → Frontend via Flask templates

scikit-learn → Machine learning model

ChromaDB → Vector database for RAG

SentenceTransformer → Embedding text data

Google Gemini API → AI-powered HR recommendations

📖 Why RAG?
Instead of generating random or generic AI suggestions, RAG ensures recommendations align with company-specific HR policies. It grounds Gemini’s output in predefined, trusted HR strategies — improving reliability and business relevance.

If you'd like, I can help you package this as:

A PDF portfolio version

A LinkedIn post draft

Or even plain text formatted for your resume profile summary.

Let me know!







You said:
what is vector embedding 


ChatGPT said:
Vector embedding is a way of representing words, sentences, or even larger pieces of information as a list (vector) of numbers. These numbers capture the meaning or features of the text in a format that machines (like machine learning models or databases) can easily work with.

In simpler terms:
When we convert human language into numbers that carry meaning — that’s called embedding.

✅ Example Analogy:
Word: "Employee Retention"

Embedding: [0.12, -0.55, 0.91, ..., 0.07] → A long list of numbers representing that phrase’s meaning.

✅ Why We Use It:
Computers can’t understand text like humans do.

By converting text into vectors, we can compare, search, or group similar things together.

For example, if two employee profiles have similar embeddings, it means their job situations might be similar.

✅ Where It Fits in Your Project:
Knowledge Base Chunks → Vector Embedding:
You converted HR strategies into embeddings using SentenceTransformer.

Employee Query → Vector Embedding:
The app turns the employee’s details into an embedding and finds the closest matching strategies from the knowledge base.

