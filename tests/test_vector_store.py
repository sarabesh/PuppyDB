# test_vector_store.py
import numpy as np
from ..vector_store import VectorStore  # Change to VectorStore if you rename class!

VECTOR_SIZE = 512
VECTOR_FILE_PATH = "test_vectors.bin"

# Helper to generate random vector
def random_vector():
    return np.random.randn(VECTOR_SIZE).astype('float32')

# Init VectorStore
vs = VectorStore(VECTOR_FILE_PATH)

# Test 1: Insert first vector
vec1 = random_vector()
offset1 = vs.append_vector(vec1)
print(f"Inserted vec1 at offset {offset1}")

# Read vec1 back
retrieved_vec1 = vs.read_vector(offset1)
assert np.allclose(vec1, retrieved_vec1), "vec1 mismatch!"
print("vec1 retrieval OK ✅")

# Test 2: Insert second vector
vec2 = random_vector()
offset2 = vs.append_vector(vec2)
print(f"Inserted vec2 at offset {offset2}")

# Read vec2 back
retrieved_vec2 = vs.read_vector(offset2)
assert np.allclose(vec2, retrieved_vec2), "vec2 mismatch!"
print("vec2 retrieval OK ✅")

# Test 3: Truncate
vs.truncate()
print("File truncated.")

# Test 4: Insert new vector after truncate
vec3 = random_vector()
offset3 = vs.append_vector(vec3)
assert offset3 == 0, "Expected offset 0 after truncate"
retrieved_vec3 = vs.read_vector(offset3)
assert np.allclose(vec3, retrieved_vec3), "vec3 mismatch after truncate!"
print("vec3 retrieval after truncate OK ✅")

# Cleanup
vs.close()
print("All VectorStore tests passed! 🎉")
