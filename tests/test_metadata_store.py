# test_metadata_store.py
import os
from ..metadata_store import MetadataStore

METADATA_STORE_PATH = "test_metadata_store"

# Init MetadataStore
ms = MetadataStore(METADATA_STORE_PATH)
print("MetadataStore initialized.")

# Put and get metadata
vector_id = "vec_001"
offset = 2048
metadata = {"user": "alice", "tag": "test"}

ms.put_metadata(vector_id, offset, metadata)
print(f"Inserted metadata for {vector_id}.")

record = ms.get_metadata(vector_id)
assert record is not None, "Failed to retrieve metadata!"
assert record["offset"] == offset, "Offset mismatch!"
assert record["metadata"] == metadata, "Metadata mismatch!"
print("Metadata retrieval OK ✅")

# Delete metadata
ms.delete_metadata(vector_id)
record_after_delete = ms.get_metadata(vector_id)
assert record_after_delete is None, "Metadata not deleted!"
print("Metadata deletion OK ✅")

# Truncate metadata store
# Insert multiple records
for i in range(3):
    vector_id = f"vec_{i}"
    offset = i * 2048
    metadata = {"index": i}
    ms.put_metadata(vector_id, offset, metadata)

# Verify they exist
for i in range(3):
    vector_id = f"vec_{i}"
    record = ms.get_metadata(vector_id)
    assert record is not None, f"Missing record for {vector_id}"

#get all vector IDs
vector_ids = ms.get_all_vector_ids()
assert len(vector_ids) == 3, "Expected 3 vector IDs!"
print(f"Found vector IDs: {vector_ids}")

# Truncate
ms.truncate()
print("Metadata store truncated.")

# Verify all records are gone
for i in range(3):
    vector_id = f"vec_{i}"
    record = ms.get_metadata(vector_id)
    assert record is None, f"Record not cleared for {vector_id}"

print("Metadata truncation OK ✅")

# Cleanup
ms.close()
print("All MetadataStore tests passed! 🎉")

if os.path.exists(METADATA_STORE_PATH):
    for filename in os.listdir(METADATA_STORE_PATH):
        os.remove(os.path.join(METADATA_STORE_PATH, filename))
    os.rmdir(METADATA_STORE_PATH)