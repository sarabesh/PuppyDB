"""
Example: Insert CLIP text embeddings into PuppyDB 🐶
Using OpenAI CLIP (clip package)
"""

import clip
import torch
import numpy as np

# Import PuppyDB
from puppydb.core import PuppyDB

# Load CLIP model + tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

VECTOR_FILE_PATH = "test_vectors.bin"
METADATA_STORE_PATH = "test_metadata_store"

# Init PuppyDB
db = PuppyDB(VECTOR_FILE_PATH, METADATA_STORE_PATH)

# Example texts
texts = [
    "A photo of a cat",
    "A photo of a dog",
    "An image of a beautiful beach",
    "A futuristic city skyline"
]

# Instert CLIP text embeddings into PuppyDB
for i, text in enumerate(texts):
    # create CLIP text embedding
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    embedding = text_features[0].cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)

    # Insert into PuppyDB, using vector_id and metadata
    # Vector ID format: clip_text_001, clip_text_002, etc.
    # Metadata includes type and original text
    vector_id = f"clip_text_{i+1:03d}"
    metadata = {"type": "text", "text": text}
    db.insert_vector(vector_id, embedding.astype('float32'), metadata)
    print(f"Inserted vector {vector_id}: '{text}'")

# Retrieve one example to verify
vector_id = "clip_text_001"
retrieved = db.get_vector(vector_id)
if retrieved:
    retrieved_vector, retrieved_metadata = retrieved
    print(f"\nRetrieved {vector_id}:")
    print("Metadata:", retrieved_metadata)
else:
    print(f"Failed to retrieve {vector_id}.")

# TEST: Delete a vector
vector_id_to_delete = "clip_text_002"
db.delete_vector(vector_id_to_delete)
print(f"\nDeleted vector {vector_id_to_delete}.")

# Verify deletion
retrieved = db.get_vector(vector_id_to_delete)
if retrieved is None:
    print(f"Verified: {vector_id_to_delete} was deleted.")
else:
    print(f"ERROR: {vector_id_to_delete} still exists!")

# TEST: Re-insert (update) a vector with same vector_id
text_update = "An updated description of a cat"
text_tokens = clip.tokenize([text_update]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
embedding = text_features[0].cpu().numpy()
embedding = embedding / np.linalg.norm(embedding)

vector_id_update = "clip_text_001"
metadata_update = {"type": "text", "text": text_update}

db.insert_vector(vector_id_update, embedding.astype('float32'), metadata_update)
print(f"\nRe-inserted (updated) vector {vector_id_update}: '{text_update}'")

# Verify update
retrieved = db.get_vector(vector_id_update)
if retrieved:
    retrieved_vector, retrieved_metadata = retrieved
    print(f"\nRetrieved updated {vector_id_update}:")
    print("Metadata:", retrieved_metadata)
else:
    print(f"Failed to retrieve updated {vector_id_update}.")

# TEST: Truncate entire PuppyDB
db.truncate()
print("\nPuppyDB truncated.")

# Verify truncation
retrieved = db.get_vector("clip_text_001")
if retrieved is None:
    print("Verified: PuppyDB is empty after truncation.")
else:
    print("ERROR: Data still exists after truncation!")


# Close PuppyDB
db.close()
print("\nPuppyDB closed.")