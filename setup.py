from setuptools import setup, find_packages

setup(
    name="image-processing-tools",
    version="0.1.0",
    author="Oussama GUELFAA",
    author_email="your.email@example.com",
    description="A collection of image processing tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Oussama-Guelfaa/Image-Processing",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "imgproc=src.cli:main",
        ],
    },
)
