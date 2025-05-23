# Machine Learning for Image Classification

This document provides an overview of the machine learning module for image classification in the Image Processing project.

**Author:** Oussama GUELFAA  
**Date:** 01-04-2025

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
   - [Feature Extraction](#feature-extraction)
   - [Classification Algorithms](#classification-algorithms)
3. [Implementation](#implementation)
   - [Feature Extraction](#feature-extraction-implementation)
   - [Classification](#classification-implementation)
   - [Visualization](#visualization-implementation)
4. [Usage](#usage)
   - [Command-Line Interface](#command-line-interface)
   - [Examples](#examples)
5. [Results](#results)

## Introduction

The machine learning module provides functionality for image classification using various feature extraction techniques and machine learning algorithms. It is designed to work with the Kimia dataset, which contains binary shape images from different classes.

The module supports:
- Feature extraction from binary images
- Training and evaluation of classifiers
- Visualization of results
- Command-line interface for easy use

## Theoretical Background

### Feature Extraction

Feature extraction is a critical step in image classification. It involves transforming raw image data into a set of features that can be used by machine learning algorithms. The module implements several feature extraction techniques:

#### Hu Moments

Hu moments are a set of seven image moments that are invariant to translation, scale, and rotation. They are particularly useful for shape recognition tasks. The moments are calculated from the central moments of the image:

1. Calculate the raw moments of the image
2. Calculate the central moments
3. Calculate the normalized central moments
4. Compute the seven Hu moments

Hu moments are effective for recognizing shapes regardless of their position, size, or orientation in the image.

#### Zernike Moments

Zernike moments are based on Zernike polynomials, which form a complete orthogonal set over the unit circle. They are rotation-invariant and can capture more detailed features than Hu moments. The computation involves:

1. Map the image to the unit circle
2. Calculate the Zernike polynomials
3. Compute the Zernike moments

Zernike moments provide a more detailed description of the shape and are particularly useful for complex shapes.

#### Geometric Features

Geometric features describe the shape's basic properties:

- Area: The number of pixels in the shape
- Perimeter: The length of the shape's boundary
- Circularity: A measure of how circular the shape is (4π × Area / Perimeter²)
- Aspect ratio: The ratio of width to height of the bounding rectangle
- Solidity: The ratio of the shape's area to its convex hull area

These features provide a simple but effective description of the shape's geometry.

### Classification Algorithms

The module implements several classification algorithms:

#### Multi-Layer Perceptron (MLP)

A neural network with multiple layers of neurons. Each neuron applies a non-linear activation function to a weighted sum of its inputs. The network is trained using backpropagation to minimize the error between predicted and actual classes.

#### Support Vector Machine (SVM)

SVM finds a hyperplane that best separates the classes in the feature space. It maximizes the margin between the hyperplane and the nearest data points from each class. For non-linearly separable data, kernel functions are used to map the data to a higher-dimensional space.

#### Random Forest

An ensemble method that builds multiple decision trees and combines their predictions. Each tree is trained on a random subset of the data and features, which helps to reduce overfitting and improve generalization.

## Implementation

### Feature Extraction Implementation

The feature extraction module (`feature_extraction.py`) provides functions to extract various features from images:

```python
# Extract Hu moments
hu_moments = extract_hu_moments(image)

# Extract Zernike moments
zernike_moments = extract_zernike_moments(image, radius=50, degree=10)

# Extract geometric features
geometric_features = extract_geometric_features(image)

# Extract all features
features = extract_features(image, feature_types=['hu', 'zernike', 'geometric'])
```

### Classification Implementation

The classification module (`classification.py`) provides functions for training and evaluating classifiers:

```python
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split_dataset(features, labels)

# Train a classifier
classifier, scaler = train_classifier(X_train, y_train, classifier_type='mlp')

# Evaluate the classifier
accuracy, y_pred, report, conf_matrix = evaluate_classifier(classifier, X_test, y_test, scaler)

# Classify a new image
label, probability = classify_image(new_image, classifier, scaler)
```

### Visualization Implementation

The visualization module (`visualization.py`) provides functions for visualizing the results:

```python
# Plot confusion matrix
fig, ax = plot_confusion_matrix(y_test, y_pred, class_names)

# Plot feature importance (for Random Forest)
fig, ax = plot_feature_importance(classifier)

# Visualize dataset using dimensionality reduction
fig, ax = visualize_dataset(features, labels, class_names, method='pca')

# Visualize misclassified images
fig = visualize_classification_results(images, y_true, y_pred, class_names)
```

## Usage

### Command-Line Interface

The machine learning module can be used through the command-line interface:

```bash
# Train a classifier
python main.py ml --task train --dataset data/images_Kimia --classifier mlp --features all --output output/machine_learning

# Classify an image
python main.py ml --task classify --image data/images_Kimia/camel-1.bmp --model output/machine_learning/model.pkl

# Visualize dataset
python main.py ml --task visualize --dataset data/images_Kimia --features all --output output/machine_learning
```

### Examples

#### Example 1: Training a classifier

```bash
python main.py ml --task train --dataset data/images_Kimia --classifier mlp --features all --output output/machine_learning
```

This command will:
1. Load the Kimia dataset from `data/images_Kimia`
2. Extract Hu moments, Zernike moments, and geometric features from each image
3. Split the dataset into training and testing sets
4. Train an MLP classifier on the training set
5. Evaluate the classifier on the testing set
6. Save the trained model and results to `output/machine_learning`

#### Example 2: Classifying an image

```bash
python main.py ml --task classify --image data/images_Kimia/camel-1.bmp --model output/machine_learning/model.pkl
```

This command will:
1. Load the trained model from `output/machine_learning/model.pkl`
2. Load the image from `data/images_Kimia/camel-1.bmp`
3. Extract features from the image
4. Classify the image using the trained model
5. Display the predicted class and probability

## Results

The machine learning module achieves good classification accuracy on the Kimia dataset. The following results were obtained using an MLP classifier with Hu moments, Zernike moments, and geometric features:

- Accuracy: 95.8%
- Confusion matrix:
  - Most confusions occur between similar shapes (e.g., apple and bone)
  - Perfect classification for camel class

The visualization tools help to understand the classification results and identify potential improvements.

---

For more information, see the source code in the `src/image_processing/machine_learning` directory.
