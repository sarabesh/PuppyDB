# tests/test_vector_store.py

import os
import numpy as np
import pytest
from puppydb.vector_store import VectorStore

VECTOR_SIZE = 512
VECTOR_FILE_PATH = "test_vectors.bin"

# Helper to generate random vector
def random_vector():
    return np.random.randn(VECTOR_SIZE).astype('float32')

@pytest.fixture
def vs():
    # Setup: create VectorStore instance
    vs = VectorStore(VECTOR_FILE_PATH)
    yield vs
    # Teardown: cleanup
    vs.close()
    if os.path.exists(VECTOR_FILE_PATH):
        os.remove(VECTOR_FILE_PATH)

def test_append_and_read_vectors(vs):
    # Insert first vector
    vec1 = random_vector()
    offset1 = vs.append_vector(vec1)

    # Read vec1 back
    retrieved_vec1 = vs.read_vector(offset1)
    assert np.allclose(vec1, retrieved_vec1), "vec1 mismatch!"

    # Insert second vector
    vec2 = random_vector()
    offset2 = vs.append_vector(vec2)

    # Read vec2 back
    retrieved_vec2 = vs.read_vector(offset2)
    assert np.allclose(vec2, retrieved_vec2), "vec2 mismatch!"

def test_truncate_and_append(vs):
    # Insert and truncate
    vec1 = random_vector()
    vs.append_vector(vec1)
    vs.truncate()

    # Insert new vector after truncate
    vec3 = random_vector()
    offset3 = vs.append_vector(vec3)
    assert offset3 == 0, "Expected offset 0 after truncate"

    retrieved_vec3 = vs.read_vector(offset3)
    assert np.allclose(vec3, retrieved_vec3), "vec3 mismatch after truncate!"
