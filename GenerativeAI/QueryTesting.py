import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# embed_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

query = "شرایط وام دانشجویی چیست ؟"
query_embedding = embed_model.encode([query])

# Load the FAISS index
index = faiss.read_index("vector_index.faiss")

with open("documents.pkl", "rb") as f:
    documents = pickle.loadd(f)

k = 3  # Number of results to retrieve
distances, indices = index.search(query_embedding, k)

# Display the results
print(f"Query: {query}")
for i, idx in enumerate(indices[0]):
    print(f"\nMatch {i + 1} (Distance: {distances[0][i]:.2f}):")
    print(documents[idx])

with open("query_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Query: {query}\n\n")
    for i, idx in enumerate(indices[0]):
        f.write(f"Match {i + 1} (Distance: {distances[0][i]:.2f}):\n")
        f.write(documents[idx] + "\n\n")