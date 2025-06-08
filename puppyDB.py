#Puppy DB main class
#wraps vector store and metadata store
#provides a simple interface for inserting and retrieving vectors with metadata

from .vector_store import VectorStore
from .metadata_store import MetadataStore

class PuppyDB:
    # inits vector store and metadata store
    def __init__(self, vector_file_path, metadata_db_path):
        self.vector_store = VectorStore(vector_file_path)
        self.metadata_store = MetadataStore(metadata_db_path)

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