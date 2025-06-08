# PuppyDB - Tests

## Running the tests

To run the tests, the recommended way is to run them as a **module** so that relative imports work properly.

### Steps:

1. Move to the folder **containing the `puppydb/` folder** (your project root).
2. From this project root folder, run:

```bash
python -m puppydb.tests.test_vector_store
```

This ensures that Python treats puppydb as a package and relative imports will work correctly.