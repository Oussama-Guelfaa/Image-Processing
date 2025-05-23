#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for machine_learning.

This module tests the machine learning functionality for image processing.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.image_processing.machine_learning.feature_extraction import (
    load_kimia_dataset,
    extract_dataset_features,
    extract_features
)
from src.image_processing.machine_learning.classification import (
    train_test_split_dataset,
    train_classifier,
    evaluate_classifier,
    cross_validate
)
from src.image_processing.machine_learning.visualization import (
    plot_confusion_matrix,
    visualize_dataset,
    visualize_classification_results
)

# Create output directory
output_dir = 'output/machine_learning'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
print("Loading Kimia dataset...")
images, labels, class_names = load_kimia_dataset('data/images_Kimia')
print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")

# Extract features
print("Extracting features...")
feature_types = ['hu', 'geometric']  # Using Hu moments and geometric features
features = extract_dataset_features(images, feature_types=feature_types)
print(f"Extracted {features.shape[1]} features per image")

# Visualize dataset
print("Visualizing dataset with PCA...")
fig, ax = visualize_dataset(features, labels, class_names=class_names, method='pca')
plt.savefig(os.path.join(output_dir, 'dataset_pca.png'))
plt.close(fig)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split_dataset(
    features, labels, test_size=0.25)
print(f"Split dataset into {len(X_train)} training and {len(X_test)} testing samples")

# Train classifier
print("Training MLP classifier...")
classifier, scaler = train_classifier(X_train, y_train, classifier_type='mlp')

# Evaluate classifier
print("Evaluating classifier...")
accuracy, y_pred, report, conf_matrix = evaluate_classifier(classifier, X_test, y_test, scaler)
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

# Plot confusion matrix
print("Plotting confusion matrix...")
fig, ax = plot_confusion_matrix(y_test, y_pred, class_names=class_names)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close(fig)

# Visualize misclassified images
print("Visualizing misclassified images...")
test_images = [images[i] for i in range(len(images)) if i in np.where(np.array(labels) == y_test)[0]]
fig = visualize_classification_results(test_images, y_test, y_pred, class_names)
if fig is not None:
    plt.savefig(os.path.join(output_dir, 'misclassified_images.png'))
    plt.close(fig)

print("Done! Results saved to", output_dir)
