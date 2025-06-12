"""
HNSW search test with texts loaded from CSV 🐶
"""

import clip
import torch
import numpy as np
import pandas as pd
import os
import time

# Import PuppyDB
from ..puppyDB import PuppyDB

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

VECTOR_FILE_PATH = "test_vectors.bin"
METADATA_STORE_PATH = "test_metadata_store"

# Init PuppyDB
db = PuppyDB(VECTOR_FILE_PATH, METADATA_STORE_PATH)

# Truncate DB first
db.truncate()

# Helper: get CLIP text embedding
def get_clip_text_embedding(text):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    embedding = text_features[0].cpu().numpy()
    return embedding / np.linalg.norm(embedding)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "test_texts.csv")

# Load texts from CSV
df = pd.read_csv(csv_path)
texts = df["text"].tolist()

# Insert all texts into PuppyDB
for i, text in enumerate(texts):
    embedding = get_clip_text_embedding(text)
    vector_id = f"vec_{i+1:03d}"
    db.insert_vector(vector_id, embedding.astype('float32'), {"text": text})
    print(f"Inserted {vector_id}: '{text}'")

# Choose query text (you can vary this!)
query_text = "The cutest puppy playing with a ball"

# Get query embedding
query_embedding = get_clip_text_embedding(query_text)

# Build index (just gets vector in memory for HNSW search)
db.build_index(method="hnsw")

# Run search
start_time = time.time()
results = db.search(query_embedding, k=5, method="hnsw")
end_time = time.time()
print(f"\nSearch completed in {end_time - start_time:.4f} seconds.")

# Display results
print(f"\nHNSW Search Results for query: '{query_text}'\n")

for i, r in enumerate(results):
    print(f"{i+1}. Vector ID: {r[0]}")
    print(f"   Similarity: {r[1]:.4f}")
    print(f"   Metadata: {r[2]['text']}\n")

# Close PuppyDB
db.close()
print("PuppyDB closed.")
