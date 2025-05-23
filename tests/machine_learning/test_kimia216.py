#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for machine_learning.

This module tests the machine learning functionality for image processing.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage import io

# Create output directory
output_dir = 'output/machine_learning_kimia216'
os.makedirs(output_dir, exist_ok=True)

def load_kimia_dataset(data_dir='data/images_Kimia216', max_images_per_class=None):
    """Load the Kimia dataset from the specified directory."""
    # Get all image files
    image_files = glob.glob(os.path.join(data_dir, '*.bmp'))

    # Extract class names from filenames
    class_names = sorted(list(set([os.path.basename(f).split('-')[0] for f in image_files])))

    # Create a dictionary to store images for each class
    class_images = {cls: [] for cls in class_names}

    # Load images and organize by class
    for image_file in image_files:
        class_name = os.path.basename(image_file).split('-')[0]
        class_images[class_name].append(image_file)

    # Limit the number of images per class if specified
    if max_images_per_class is not None:
        for cls in class_names:
            class_images[cls] = class_images[cls][:max_images_per_class]

    # Create lists for images and labels
    images = []
    labels = []

    # Load images and assign labels
    for i, cls in enumerate(class_names):
        for image_file in class_images[cls]:
            # Load image
            image = io.imread(image_file)
            # Convert to binary if needed
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            # Threshold to ensure binary image
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            # Add to lists
            images.append(image)
            labels.append(i)

    return images, labels, class_names

def extract_hu_moments(image):
    """Extract Hu moments from an image."""
    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments
    moments = cv2.moments(binary)

    # Calculate Hu moments
    hu_moments = cv2.HuMoments(moments)

    # Log transform to improve feature distribution
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return hu_moments.flatten()

def extract_geometric_features(image):
    """Extract geometric features from an image."""
    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours found, return zeros
    if not contours:
        return np.zeros(5)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calculate circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0

    # Calculate convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Return features
    return np.array([area, perimeter, circularity, aspect_ratio, solidity])

def extract_hog_features(image, pixels_per_cell=(8, 8)):
    """Extract HOG features from an image."""
    from skimage.feature import hog
    # Resize image for HOG
    resized = cv2.resize(image, (64, 64))
    # Extract HOG features
    hog_features = hog(resized, orientations=9, pixels_per_cell=pixels_per_cell,
                      cells_per_block=(2, 2), visualize=False)
    return hog_features

def extract_features(image, feature_types=None):
    """Extract features from an image."""
    if feature_types is None:
        feature_types = ['hu', 'geometric', 'hog']

    features = []

    if 'hu' in feature_types:
        hu_features = extract_hu_moments(image)
        features.append(hu_features)

    if 'geometric' in feature_types:
        geometric_features = extract_geometric_features(image)
        features.append(geometric_features)

    if 'hog' in feature_types:
        hog_features = extract_hog_features(image)
        features.append(hog_features)

    # Concatenate all features
    return np.concatenate(features)

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(15, 15), cmap='Blues'):
    """Plot a confusion matrix for classification results."""
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    if class_names is not None:
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    ax.set_title("Confusion Matrix")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    plt.tight_layout()

    return fig, ax

# Main execution
print("Loading Kimia216 dataset...")
images, labels, class_names = load_kimia_dataset('data/images_Kimia216')
print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")

# Extract features
print("Extracting features...")
features = []
for image in images:
    # Extract both Hu moments and geometric features
    image_features = extract_features(image, feature_types=['hu', 'geometric'])
    features.append(image_features)
features = np.array(features)
print(f"Extracted {features.shape[1]} features per image")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=42, stratify=labels)
print(f"Split dataset into {len(X_train)} training and {len(X_test)} testing samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
print("Training Voting Ensemble classifier...")
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Create individual classifiers
rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                   alpha=0.0001, max_iter=1000, random_state=42)

# Create voting classifier
classifier = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
    voting='soft'
)

# Train the ensemble
classifier.fit(X_train_scaled, y_train)

# Evaluate classifier
print("Evaluating classifier...")
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(f"Classification Report:\n{report}")

# Plot confusion matrix
print("Plotting confusion matrix...")
fig, ax = plot_confusion_matrix(y_test, y_pred, class_names=class_names)
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close(fig)

# SVM doesn't provide feature importances

# Visualize misclassified images
print("Visualizing misclassified images...")
misclassified = np.where(y_test != y_pred)[0]
if len(misclassified) > 0:
    print(f"Found {len(misclassified)} misclassified images")
    # Create a figure to show misclassifications
    n_cols = min(5, len(misclassified))
    n_rows = (len(misclassified) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # For each misclassified image
    for i, idx in enumerate(misclassified):
        if i < len(axes):
            # Get the true and predicted labels
            true_label = y_test[idx]
            pred_label = y_pred[idx]

            # Set the title
            axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
            axes[i].axis('off')

    # Hide unused axes
    for i in range(len(misclassified), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassified_images.png'))
    plt.close(fig)
else:
    print("No misclassified images found.")

print("Done! Results saved to", output_dir)
