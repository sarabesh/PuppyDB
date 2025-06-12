import numpy as np

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cosine_distance(vec1, vec2):
    """Compute cosine distance between two vectors."""
    return 1 - cosine_similarity(vec1, vec2)