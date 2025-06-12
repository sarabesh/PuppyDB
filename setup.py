from setuptools import setup, find_packages

setup(
    name="puppydb",
    version="0.1.0",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "lmdb",
    ],
    python_requires=">=3.9",
)