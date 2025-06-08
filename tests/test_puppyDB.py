# test_puppydb.py

import numpy as np
import os
from ..puppyDB import PuppyDB

VECTOR_SIZE = 512
VECTOR_FILE_PATH = "test_vectors.bin"
METADATA_STORE_PATH = "test_metadata_store"

# Helper to generate random vector
def random_vector():
    return np.random.randn(VECTOR_SIZE).astype('float32')

# Init PuppyDB
db = PuppyDB(VECTOR_FILE_PATH, METADATA_STORE_PATH)
print("PuppyDB initialized.")

# Test 1: Insert vector
vector_id = "vec_001"
vector = random_vector()
metadata = {"user": "alice", "tag": "puppydb_test"}

offset = db.insert_vector(vector_id, vector, metadata)
print(f"Inserted vector {vector_id} at offset {offset}")

# Test 2: Retrieve vector
retrieved = db.get_vector(vector_id)
assert retrieved is not None, "Failed to retrieve vector!"
retrieved_vector, retrieved_metadata = retrieved
assert np.allclose(vector, retrieved_vector), "Vector data mismatch!"
assert retrieved_metadata == metadata, "Metadata mismatch!"
print("Vector retrieval OK ✅")

# Test 3: Delete vector
db.delete_vector(vector_id)
retrieved_after_delete = db.get_vector(vector_id)
assert retrieved_after_delete is None, "Vector not deleted!"
print("Vector deletion OK ✅")

# Test 4: Truncate DB
# Insert multiple vectors
for i in range(3):
    vector_id = f"vec_{i}"
    vec = random_vector()
    meta = {"index": i}
    db.insert_vector(vector_id, vec, meta)

# Truncate PuppyDB
db.truncate()
print("PuppyDB truncated.")

# Verify all vectors are gone
for i in range(3):
    vector_id = f"vec_{i}"
    retrieved = db.get_vector(vector_id)
    assert retrieved is None, f"Vector {vector_id} not cleared!"

print("Truncation verification OK ✅")

# Cleanup
db.close()
print("All PuppyDB tests passed! 🎉")

# Optionally: cleanup test files
if os.path.exists(VECTOR_FILE_PATH):
    os.remove(VECTOR_FILE_PATH)
if os.path.exists(METADATA_STORE_PATH):
    for filename in os.listdir(METADATA_STORE_PATH):
        os.remove(os.path.join(METADATA_STORE_PATH, filename))
    os.rmdir(METADATA_STORE_PATH)
