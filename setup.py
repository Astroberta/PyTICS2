# setup.py

from setuptools import setup, find_packages

setup(
    name="PyTICS",
    version="0.1.0",
    author="Roberta Vieliute",
    author_email="rv4@st-andrews.ac.uk",
    description="PyTICS: A Python package for intercalibration of photometric astronomical fields.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/astroberta/PyTICS",  # Update with your repository URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],  # List dependencies here if any
    project_urls={
        "Citation": "https://doi.org/yourdoi",  # Update with your DOI if applicable
    },
)