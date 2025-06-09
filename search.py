import numpy as np
import os

class Search:
    def search(self, query_vector, k):
        raise NotImplementedError

class KnnSearch(Search):
    def __init__(self, puppydb):
        self.puppydb = puppydb

    def search(self, query_vector, k=5):
        # Exact KNN logic here
        # get all vectors from the vector store
        # compute distances to query_vector
        # return the k nearest vectors
        vectors = self.puppydb.get_all_vectors()
        if not vectors:
            return []
        # Extract vector data and compute distances
        similarities = []
        print(f"query_vector: {query_vector.shape}")
        
        for vector_id, vector in vectors:
            
            similarity = np.dot(vector, query_vector) / (np.linalg.norm(vector) * np.linalg.norm(query_vector))
            similarities.append((vector_id, similarity))
        
        # Sort by distance and get the top k
        similarities.sort(key=lambda x: x[1])  

        # find metadata for the top k vectors
        results = []
        for vector_id, similarity in similarities[:k]:
            vector, metadata = self.puppydb.get_vector(vector_id)
            results.append((vector_id, similarity , metadata))
        
        return results
        

class HNSWSearch(Search):
    def __init__(self, puppydb):
        self.puppydb = puppydb

    def search(self, query_vector, k=5):
        # HNSW logic here
        # This is a placeholder for HNSW logic
        # In a real implementation, you would use an HNSW library or algorithm
        raise NotImplementedError("HNSW search is not implemented yet.")