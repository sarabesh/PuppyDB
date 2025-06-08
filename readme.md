# PuppyDB 🐶

**PuppyDB** is a playful vector database built from scratch with:

- **Flat files + mmap** for fast vector storage
- **LMDB** for metadata and offsets
- **HNSW** (Hierarchical Navigable Small World graphs) for approximate nearest neighbor search (coming soon!)
- A simple WAL for durability (planned)

Fetch your vectors fast — just like a good puppy! 🐾

---

### Project Status

🚧 **In Progress**  
This is an experimental learning project to explore how vector databases work under the hood.  
Major components are being built step by step.

---

### Planned Features

- ✅ Basic vector db operations (Persistent vector and metadata storage)
- 🛠️ Perform top-k nearest neighbor search (**coming soon with HNSW
- Lightweight and fun to use

---

### Project Goals

- Learn the internals of building a vector DB
- Explore custom HNSW implementation
- Keep the architecture simple and hackable

---

### Usage

```python
from puppydb import PuppyDB
import numpy as np

# Initialize PuppyDB
db = PuppyDB("puppydb/data/vectors.bin", "puppydb/data/metadata_store")

# Insert vector
vector_id = "vec_001"
vector = np.random.randn(512).astype('float32')
metadata = {"user": "alice"}

db.insert_vector(vector_id, vector, metadata)

# Retrieve vector
retrieved = db.get_vector(vector_id)
if retrieved:
    retrieved_vector, retrieved_metadata = retrieved
    print("Vector metadata:", retrieved_metadata)

# Delete vector
db.delete_vector(vector_id)

# Truncate entire DB
db.truncate()

# Close DB
db.close()

```