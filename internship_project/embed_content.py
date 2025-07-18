!pip install chromadb

from sentence_transformers import SentenceTransformer
import chromadb
import os

# step 1 load content from file if available otherwise exit program
content_path = "content.md"
if not os.path.exists(content_path):
    print(f"file not found {content_path}")
    exit()

with open(content_path, "r", encoding="utf-8") as f:
    content = f.read()

# step 2 split content based on empty lines basically 2 newlines in between
chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]

if not chunks:
    print("no content chunks found check content md format again maybe spacing issue")
    exit()

print(f"loaded {len(chunks)} chunks for embedding") # prints how many pieces we got for embedding

# step 3 init embedding model for sentence vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# step 4 generate embeddings from the text chunks we have above
embeddings = model.encode(chunks)
print("embeddings generated") # embedding process is done

# step 5 setting up chroma db client to save vectors locally
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("employee_retention")

# step 6 clearing previous stored docs if there are any to avoid duplicates
existing = collection.get(include=["documents"])
if existing.get("ids"):
    collection.delete(ids=existing["ids"])
    print(f"cleared {len(existing['ids'])} old documents from collection")
else:
    print("no old documents found skipping clear step") # no old data is present

# step 7 now adding chunks and embeddings freshly
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding.tolist()],
        metadatas=[{"source": "content.md"}],
        documents=[chunk]
    )

print("embeddings stored in chromadb successfully") # all saved to chromadb

# step 8 verification check if things saved correctly
results = collection.get(include=['documents'])
print(f"verification total stored documents {len(results['documents'])}") # final verification print

import shutil

# last step make zip file from chroma db folder to backup or download
shutil.make_archive('chroma_db_backup', 'zip', './chroma_db')

# for colab only download it using colab files module
from google.colab import files
files.download('chroma_db_backup.zip')
