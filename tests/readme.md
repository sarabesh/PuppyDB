# PuppyDB - Tests

## Running the tests

The recommended way to run the tests is using [pytest](https://pytest.pypi.org/), which automatically discovers and runs all tests.

### Steps:

1. Move to the project root folder (where `puppydb/`, `tests/`, `setup.py` are located).

2. Install PuppyDB in editable mode (once):

    ```bash
    pip install -e .
    ```

3. Run the full test suite with:

    ```bash
    pytest -v -ra
    ```

    - `-v` → verbose output (shows test names)
    - `-ra` → shows extra test summary info (skipped, xfail, etc.)

4. To run a specific test file:

    ```bash
    pytest -v -ra tests/test_vector_store.py
    ```

5. To run a specific test function:

    ```bash
    pytest -v -ra -k test_knn_search
    ```

---
