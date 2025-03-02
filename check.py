import numpy as np

# Load the embeddings from the .npy file
embeddings = np.load("combined_paragraph_embeddings.npy")

# Print the shape of the embeddings
print("Embeddings shape:", embeddings.shape)

# Print the first embedding (first paragraph)
print("\nFirst embedding (first paragraph):\n", embeddings[0])

# Print the first 5 embeddings (first 5 paragraphs)
print("\nFirst 5 embeddings (first 5 paragraphs):\n", embeddings[:5])