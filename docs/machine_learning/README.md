# Machine Learning Module

**Author:** Oussama GUELFAA  
**Date:** 01-04-2025

## Overview

The Machine Learning module provides tools for applying machine learning techniques to image processing tasks. This includes feature extraction, classification, and neural networks for image analysis.

## Features

- **Feature Extraction**: Extract meaningful features from images using region properties
- **Classification**: Train and evaluate classifiers on image features
- **Neural Networks**: Apply neural networks for image classification tasks

## Modules

### Feature Extraction

The feature extraction module (`feature_extraction.py`) provides functions to extract features from images for machine learning tasks. It uses scikit-image's region properties to compute geometric features of binary images.

```python
from src.image_processing.machine_learning.feature_extraction import extract_region_props, load_dataset

# Extract features from a single image
features = extract_region_props('path/to/image.bmp')

# Load a dataset and extract features from all images
features, targets = load_dataset('data/images_Kimia', classes)
```

#### Available Features

The following features are extracted from each image:

1. **Area**: Number of pixels in the region
2. **Perimeter**: Perimeter of the region
3. **Eccentricity**: Eccentricity of the region
4. **Equivalent Diameter**: Diameter of circle with same area
5. **Euler Number**: Euler characteristic of the region
6. **Extent**: Ratio of pixels in region to pixels in bounding box
7. **Major Axis Length**: Length of major axis of ellipse that has the same normalized second central moments as the region
8. **Minor Axis Length**: Length of minor axis of ellipse that has the same normalized second central moments as the region
9. **Solidity**: Ratio of pixels in the region to pixels in the convex hull

### Classification

The classification module (`classification.py`) provides functions for training and evaluating classifiers on image features.

```python
from src.image_processing.machine_learning.classification import train_classifier, evaluate_classifier

# Train a classifier
classifier, scaler = train_classifier(X_train, y_train, classifier_type='mlp')

# Evaluate the classifier
accuracy, confusion_matrix = evaluate_classifier(classifier, scaler, X_test, y_test, class_names=classes)
```

#### Supported Classifiers

1. **Multi-Layer Perceptron (MLP)**: Neural network classifier
2. **Support Vector Machine (SVM)**: Support vector machine classifier

## Examples

### Kimia Database Classification

The `kimia_classification.ipynb` notebook demonstrates how to use the machine learning module to classify images from the Kimia database. It includes:

1. Loading the Kimia dataset
2. Extracting features from binary images
3. Training neural network and SVM classifiers
4. Evaluating classifier performance
5. Visualizing results with confusion matrices

## Usage

To use the machine learning module in your own code:

```python
from src.image_processing.machine_learning import extract_region_props, train_classifier, evaluate_classifier

# Extract features
features = extract_region_props('path/to/image.bmp')

# Train a classifier
classifier, scaler = train_classifier(X_train, y_train)

# Evaluate the classifier
accuracy, cm = evaluate_classifier(classifier, scaler, X_test, y_test)
```

## Requirements

- NumPy
- scikit-image
- scikit-learn
- matplotlib
- seaborn

## References

- Kimia Database: A dataset of binary shape images used for shape recognition and classification
- scikit-image: Image processing in Python
- scikit-learn: Machine learning in Python
