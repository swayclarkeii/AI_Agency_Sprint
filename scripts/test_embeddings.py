import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
kb_file = "data/kb/transcripts/transcripts_with_embeddings.csv"
df = pd.read_csv(kb_file)

# Convert string representations of embeddings to numpy arrays
embeddings = []
for emb_str in df['embedding']:
    embeddings.append(np.array(eval(emb_str)))
embeddings = np.array(embeddings)

# Pick a sample chunk and find similar chunks
sample_idx = 0  # Index of sample chunk to test
sample_text = df['chunk_text'].iloc[sample_idx]
print(f"Sample chunk: {sample_text[:200]}...")

# Calculate similarities to all other chunks
similarities = cosine_similarity([embeddings[sample_idx]], embeddings)[0]

# Get top 3 most similar chunks (excluding self)
similar_indices = np.argsort(similarities)[::-1][1:4]  # Skip first (self)
print("\nMost similar chunks:")
for idx in similar_indices:
    print(f"Similarity: {similarities[idx]:.4f}")
    print(f"Chunk: {df['chunk_text'].iloc[idx][:200]}...\n")