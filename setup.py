# This file is used to install the package using pip
# To install the package, run the following command:
# pip install .
from setuptools import setup, find_packages

setup(
    name="mushroom-classifier",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "joblib",
    ],
    author="Vatsal Kathiriya",
    author_email="vatsalkathiriya2@gmail.com",
    description="A web application to classify mushrooms as edible or poisonous",
)