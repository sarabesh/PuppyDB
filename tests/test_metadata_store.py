# tests/test_metadata_store.py

import os
import pytest
from puppydb.metadata_store import MetadataStore

METADATA_STORE_PATH = "test_metadata_store"

@pytest.fixture
def ms():
    # Setup: create MetadataStore instance
    ms = MetadataStore(METADATA_STORE_PATH)
    yield ms
    # Teardown: cleanup
    ms.close()
    if os.path.exists(METADATA_STORE_PATH):
        for filename in os.listdir(METADATA_STORE_PATH):
            os.remove(os.path.join(METADATA_STORE_PATH, filename))
        os.rmdir(METADATA_STORE_PATH)

def test_put_get_delete_metadata(ms):
    vector_id = "vec_001"
    offset = 2048
    metadata = {"user": "alice", "tag": "test"}

    # Put metadata
    ms.put_metadata(vector_id, offset, metadata)

    # Get metadata
    record = ms.get_metadata(vector_id)
    assert record is not None, "Failed to retrieve metadata!"
    assert record["offset"] == offset, "Offset mismatch!"
    assert record["metadata"] == metadata, "Metadata mismatch!"

    # Delete metadata
    ms.delete_metadata(vector_id)
    record_after_delete = ms.get_metadata(vector_id)
    assert record_after_delete is None, "Metadata not deleted!"

def test_bulk_insert_and_truncate(ms):
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

    # Get all vector IDs
    vector_ids = ms.get_all_vector_ids()
    assert len(vector_ids) == 3, "Expected 3 vector IDs!"

    # Truncate
    ms.truncate()

    # Verify all records are gone
    for i in range(3):
        vector_id = f"vec_{i}"
        record = ms.get_metadata(vector_id)
        assert record is None, f"Record not cleared for {vector_id}"
