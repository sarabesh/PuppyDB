# PuppyDB 🐶

**PuppyDB** is a playful vector database built from scratch with:

- **Flat files + mmap** for fast vector storage
- **LMDB** for metadata and offsets
- **HNSW** (Hierarchical Navigable Small World graphs) for approximate nearest neighbor search
- Simple and hackable architecture

Fetch your vectors fast — just like a good puppy! 🐾

---

## Project Status

🚧 **In Progress**  
This is an experimental learning project to explore how vector databases work under the hood.  
Major components are being built step by step.

---

## Features

- ✅ Persistent vector + metadata storage
- ✅ Brute-force KNN search
- ✅ Approximate nearest neighbor search (HNSW-based)
- 🛠️ Lightweight and fun to use
- 🛠️ Comprehensive test suite (pytest)

---

## Project Goals

- Learn the internals of building a vector database
- Implement a custom HNSW search
- Keep the architecture simple and hackable
- Provide clean examples and tests

---

## Usage

```python
from puppydb.core import PuppyDB
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

---

## Project Structure

puppydb/            # Core package code
tests/              # Pytest-based unit tests
examples/           # Example scripts
setup.py            # Package configuration
requirements.txt    # Project dependencies
README.md           # Project readme

---

## License
MIT License — use and modify freely.