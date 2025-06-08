# Metadata Store for managing metadata in a database
# Using LMDB
# vectorid -> {offset, metadata}

import lmdb
import json

class MetadataStore:
    def __init__(self, db_path, map_size=1<<30):
        self.env = lmdb.open(db_path, map_size=map_size, max_dbs=1) #initialize LMDB environment
        self.db = self.env.open_db(b'metadata', create=True)  # Open or create the metadata database
    
    def put_metadata(self, vector_id, offset, metadata):
        ## Store vector_id → {offset, metadata} in LMDB.
        with self.env.begin(write=True, db=self.db) as txn:
            # Convert metadata to JSON string
            metadata_json = json.dumps({'offset': offset, 'metadata': metadata})
            txn.put(vector_id.encode('utf-8'), metadata_json.encode('utf-8'))
    
    def get_metadata(self, vector_id):
        with self.env.begin(db=self.db) as txn:
            metadata_json = txn.get(vector_id.encode('utf-8'))
            if metadata_json is None:
                return None
            # Convert JSON string back to dictionary
            records = json.loads(metadata_json.decode('utf-8'))
            return records
    
    def delete_metadata(self, vector_id):
        with self.env.begin(write=True, db=self.db) as txn:
            txn.delete(vector_id.encode('utf-8'))

    def close(self):
        self.env.close()

    #truncate the metadata store
    def truncate(self):
        with self.env.begin(write=True, db=self.db) as txn:
            txn.drop(db=self.db, delete=True)
            # Reopen DB after dropping
        self.db = self.env.open_db(b'metadata', create=True)
        
