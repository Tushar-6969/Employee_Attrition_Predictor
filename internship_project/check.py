import chromadb

# Connect to your local ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("employee_retention")

# List all documents
results = collection.get(include=['documents', 'metadatas'])

print(f"Total documents: {len(results['ids'])}")
for i, doc in enumerate(results['documents']):
    print(f"Doc {i}: {doc[:100]}...")  # Print first 100 characters
