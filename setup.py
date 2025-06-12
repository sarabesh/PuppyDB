from setuptools import setup, find_packages

setup(
    name="pup-py-db",
    version="0.1.0",
    author="Sarabesh",
    description="An experimental vector database built from scratch with flat files, LMDB, and HNSW.",
    url="https://github.com/sarabesh/PuppyDB",  # Add your GitHub repo URL here
    project_urls={
        "Bug Tracker": "https://github.com/sarabesh/PuppyDB/issues",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "lmdb",
    ],
    python_requires=">=3.9",
)