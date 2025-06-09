#Puppy DB main class
#wraps vector store and metadata store
#provides a simple interface for inserting and retrieving vectors with metadata

from .vector_store import VectorStore
from .metadata_store import MetadataStore
from .search import KnnSearch, HNSWSearch



class PuppyDB:
    # inits vector store and metadata store
    def __init__(self, vector_file_path, metadata_db_path):
        self.vector_store = VectorStore(vector_file_path)
        self.metadata_store = MetadataStore(metadata_db_path)
        self.search_engines = {
            "knn": KnnSearch(self),
            "hnsw": HNSWSearch(self)
        }

    # Inserts a vector into vector store and ofset + its metadata into the metadata store
    def insert_vector(self, vector_id, vector, metadata):
        offset = self.vector_store.append_vector(vector)
        self.metadata_store.put_metadata(vector_id, offset, metadata)
        return offset

    # Retrieves a vector and its metadata by vector_id
    def get_vector(self, vector_id):
        record = self.metadata_store.get_metadata(vector_id)
        if record is None:
            return None
        offset = record['offset']
        return self.vector_store.read_vector(offset), record['metadata']
    
    #Retrieves all vector_ids from metadata store to be used for search
    def get_all_vectors(self):
        ids = self.metadata_store.get_all_vector_ids()
        vectors = []
        for vector_id in ids:
            vector = self.get_vector(vector_id)[0]
            if vector is not None:  # Ensure vector exists
                vectors.append((vector_id, vector))
        return vectors

    # Search for similar vectors using the specified method
    def search(self, query_vector, k=5, method="knn"):
        if method not in self.search_engines:
            raise ValueError(f"Unknown search method: {method}")
        return self.search_engines[method].search(query_vector, k)

    # Deletes a vector metadata by vector_id
    def delete_vector(self, vector_id):
        record = self.metadata_store.get_metadata(vector_id)
        if record is not None:
            self.metadata_store.delete_metadata(vector_id)

    # Truncates both vector store file and metadata store db
    def truncate(self):
        self.vector_store.truncate()
        self.metadata_store.truncate()

    # closes connections to vector store and metadata store
    def close(self):
        self.vector_store.close()
        self.metadata_store.close()