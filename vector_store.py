import os
import mmap
import numpy as np

#Base class for vector store
#Defines class VectorStore with methods for adding, deleting, and searching documents
#Vector store saves the embeddings in a vector file
#Serves as a handler for binary files containing vectors

VECTOR_SIZE = 512
VECTOR_BYTES = VECTOR_SIZE * 4  # Assuming float32 (4 bytes) for each vector component

class vectorStore:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'ab+')
        self.file.flush()  # Ensure the file is ready for reading/writing
        self.file_size = os.path.getsize(file_path)
        # Initialize memory-mapped object
        if self.file_size > 0:
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0)
        else:
            self.mmap_obj = None 
    
    # Append a vector to the file and return its offset
    def append_vector(self, vector):
        if len(vector) != VECTOR_SIZE:
            raise ValueError(f"Vector must be of size {VECTOR_SIZE}, got {len(vector)}")
        
        # Convert vector to bytes and write to the file
        data = vector.astype('float32').tobytes()
        offset = self.file_size #before writing, the offset is the current file size. This is where the new vector will be written.

        # vectors.bin:

        # 0       512      1024      ...
        # | vec1 | vec2 | vec3 |
        # offsets: 0, 512, 1024, ... 

        self.file.write(data)
        self.file.flush()

        self.file_size += VECTOR_BYTES

        #update mmap
        if self.mmap_obj is None:
            # creating mmap for newly updated file which is not empty
            self.mmap_obj = mmap.mmap(self.file.fileno(), 0)
        else:
            
            self.mmap_obj.resize(self.file_size)


        return offset
    
    # Delete a vector at a given offset
    def read_vector(self, offset):

        if self.mmap_obj is None:
            raise ValueError("Vector file is empty — no vectors to read")

        if offset < 0 or offset >= self.file_size:
            raise ValueError(f"Offset {offset} is out of bounds for file size {self.file_size}")
        
        # reads vector of size VECTOR_BYTES at offset
        self.mmap_obj.seek(offset)
        vector_bytes = self.mmap_obj.read(VECTOR_BYTES)
        
        # Check if the read data size matches the expected vector size
        if len(vector_bytes) != VECTOR_BYTES:
            raise ValueError("Read data size does not match expected vector size")
        
        # Convert bytes back to numpy array
        vector = np.frombuffer(vector_bytes, dtype='float32')
        return vector
    
    # truncate the file to zero size
    def truncate(self):
        # Truncate the file and reset mmap
        self.close()
        # Truncate the file (empty it)
        open(self.file_path, 'wb').close()

        # Reopen file and mmap
        self.file = open(self.file_path, 'a+b')
        self.file.flush()
        self.file_size = os.path.getsize(self.file_path)

        # Check if the file is truncated properly
        if self.file_size!=0:
            raise ValueError("File is not truncated properly, size is not zero")
        # Resize mmap to zero size
        self.mmap_obj = None


    # Close the memory-mapped object and file
    def close(self):
        # Close the memory-mapped object and file
        self.mmap_obj.close()
        self.file.close()