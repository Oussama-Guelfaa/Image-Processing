from setuptools import setup, find_packages
import os

# Lire le README.md s'il existe, sinon utiliser une description par défaut
long_description = "A collection of image processing tools for educational purposes."
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="imgproc-tools",  # Changement de nom pour éviter les conflits
    version="0.1.0",
    author="Oussama GUELFAA",
    author_email="your.email@example.com",
    description="A collection of image processing tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Image-Processing",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.17.0",
    ],
    entry_points={
        "console_scripts": [
            "imgproc=src.cli:main",
        ],
    },
)
