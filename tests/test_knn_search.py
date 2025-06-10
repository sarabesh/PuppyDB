# test_knn_search.py
import os
import numpy as np
from ..puppyDB import PuppyDB

VECTOR_SIZE = 512
VECTOR_FILE_PATH = "test_vectors.bin"
METADATA_STORE_PATH = "test_metadata_store"

# Helper: random unit vector
def random_unit_vector(size):
    v = np.random.randn(size).astype('float32')
    return v / np.linalg.norm(v)

# Init PuppyDB
db = PuppyDB(VECTOR_FILE_PATH, METADATA_STORE_PATH)
print("PuppyDB initialized.")

# Truncate to start clean
db.truncate()
print("PuppyDB truncated.")

# Insert known vectors
v1 = random_unit_vector(VECTOR_SIZE)
v2 = v1 + 0.01 * random_unit_vector(VECTOR_SIZE) # very similar
v2 /= np.linalg.norm(v2)
v3 = random_unit_vector(VECTOR_SIZE)

db.insert_vector("vec_001", v1, {"desc": "base vector"})
db.insert_vector("vec_002", v2, {"desc": "similar to v1"})
db.insert_vector("vec_003", v3, {"desc": "unrelated"})
print("Inserted 3 known vectors.")

# Run KNN search using v1 as query
db.build_index(method="knn")  # Ensure index is built before search
results = db.search(v1, k=3, method="knn")

# Print results
print("\nKNN Search Results:")
for i, r in enumerate(results):
    print(f"{i+1}. Vector ID: {r[0]}")
    print(f"   Similarity: {r[1]:.6f}")
    print(f"   Metadata: {r[2]['desc']}\n")

# Check that top 2 results are vec_001 and vec_002
returned_ids = [r[0] for r in results]

assert returned_ids[0] == "vec_001", "Top result should be vec_001!"
assert returned_ids[1] == "vec_002", "Second result should be vec_002!"

# Optional: check that similarities are in decreasing order
sims = [r[1] for r in results]
assert sims[0] >= sims[1] >= sims[2], "Similarities not in decreasing order!"

print("KNN search test passed ✅")

# Cleanup
db.close()
print("PuppyDB closed.")

# Cleanup files
if os.path.exists(VECTOR_FILE_PATH):
    os.remove(VECTOR_FILE_PATH)
if os.path.exists(METADATA_STORE_PATH):
    for filename in os.listdir(METADATA_STORE_PATH):
        os.remove(os.path.join(METADATA_STORE_PATH, filename))
    os.rmdir(METADATA_STORE_PATH)

print("All KNN search tests passed! 🎉")