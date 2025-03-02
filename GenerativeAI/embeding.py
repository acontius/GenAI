import faiss
import pickle
import docx2txt
import numpy as np
from sentence_transformers import SentenceTransformer

# Process aein.docx
docx_path = "aein.docx"
text_content = docx2txt.process(docx_path)

paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

print(f"Total paragraphs extracted from aein.docx: {len(paragraphs)}")
print("\nExample paragraph from aein.docx:\n", paragraphs[5])

# Save paragraphs to paragraphs.txt
with open("paragraphs.txt", "w", encoding="utf-8") as f:
    for paragraph in paragraphs:
        f.write(paragraph + "\n\n")

# Process pardakht.docx
docx_path = "pardakht.docx"
text_content = docx2txt.process(docx_path)

paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]

print(f"Total paragraphs extracted from pardakht.docx: {len(paragraphs)}")
print("\nExample paragraph from pardakht.docx:\n", paragraphs[0])

# Save paragraphs to pardakht_paragraphs.txt
with open("pardakht_paragraphs.txt", "w", encoding="utf-8") as f:
    for paragraph in paragraphs:
        f.write(paragraph + "\n\n")

# Load paragraphs from the text files
with open("paragraphs.txt", "r", encoding="utf-8") as f:
    aein_paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

with open("pardakht_paragraphs.txt", "r", encoding="utf-8") as f:
    pardakht_paragraphs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

# Combine paragraphs into a single list
all_documents = aein_paragraphs + pardakht_paragraphs

# Print the total number of documents
print("Total documents:", len(all_documents))


embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
aein_embeddings = embed_model.encode(aein_paragraphs, show_progress_bar=True)

pardakht_embeddings = embed_model.encode(pardakht_paragraphs, show_progress_bar=True)

combined_embeddings = np.concatenate((aein_embeddings, pardakht_embeddings), axis=0)

np.save("combined_paragraph_embeddings.npy", combined_embeddings)
print("Combined embeddings saved to combined_paragraph_embeddings.npy")


# ------------Indexing----------------#

combined_embeddings = np.load("combined_paragraph_embeddings.npy")

# Get the embedding dimension
embedding_dim = combined_embeddings.shape[1]

# Create a FAISS index
index = faiss.IndexFlatL2(embedding_dim)  #Euclidean distance

# Add embeddings to the FAISS index
index.add(combined_embeddings)

# Save the FAISS index
faiss.write_index(index, "vector_index.faiss")
print("FAISS index created and saved to vector_index.faiss")


# Save the combined documents (paragraphs)
with open("documents.pkl", "wb") as f:
    pickle.dump(all_documents, f)

print("Documents saved to documents.pkl")