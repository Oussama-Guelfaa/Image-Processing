#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine_learning

Machine learning techniques for image processing and analysis.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import glob
import numpy as np
import cv2
from skimage import io, measure, img_as_ubyte
from skimage.feature import hog
from skimage.measure import moments, moments_hu, moments_normalized
import mahotas as mh
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_kimia_dataset(data_dir='data/images_Kimia', max_images_per_class=None):
    """
    Load the Kimia dataset from the specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the Kimia dataset.
    max_images_per_class : int, optional
        Maximum number of images to load per class. If None, load all images.

    Returns
    -------
    images : list
        List of images.
    labels : list
        List of labels corresponding to the images.
    class_names : list
        List of unique class names.
    """
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
    """
    Extract Hu moments from an image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    features : ndarray
        Hu moments features.
    """
    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate moments
    moments = cv2.moments(binary)

    # Calculate Hu moments
    hu_moments = cv2.HuMoments(moments)

    # Log transform to improve feature distribution
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return hu_moments.flatten()

def extract_zernike_moments(image, radius=50, degree=10):
    """
    Extract Zernike moments from an image.

    Parameters
    ----------
    image : ndarray
        Input image.
    radius : int, optional
        Radius for Zernike moments calculation.
    degree : int, optional
        Degree for Zernike moments calculation.

    Returns
    -------
    features : ndarray
        Zernike moments features.
    """
    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Calculate Zernike moments
    zernike = mh.features.zernike_moments(binary, radius, degree=degree)

    return zernike

def extract_geometric_features(image):
    """
    Extract geometric features from an image.

    Parameters
    ----------
    image : ndarray
        Input image.

    Returns
    -------
    features : ndarray
        Geometric features.
    """
    # Ensure binary image
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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

    # Calculate minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Calculate convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # Return features
    return np.array([area, perimeter, circularity, aspect_ratio, solidity])

def extract_hog_features(image, pixels_per_cell=(8, 8)):
    """
    Extract HOG features from an image.

    Parameters
    ----------
    image : ndarray
        Input image.
    pixels_per_cell : tuple, optional
        Size (in pixels) of a cell.

    Returns
    -------
    features : ndarray
        HOG features.
    """
    # Resize image for HOG
    resized = cv2.resize(image, (64, 64))

    # Extract HOG features
    hog_features = hog(resized, orientations=9, pixels_per_cell=pixels_per_cell,
                      cells_per_block=(2, 2), visualize=False)

    return hog_features

def extract_features(image, feature_types=None):
    """
    Extract features from an image.

    Parameters
    ----------
    image : ndarray
        Input image.
    feature_types : list, optional
        List of feature types to extract. If None, extract all features.
        Options: 'hu', 'zernike', 'geometric', 'hog'

    Returns
    -------
    features : ndarray
        Extracted features.
    """
    if feature_types is None:
        feature_types = ['hu', 'zernike', 'geometric', 'hog']

    features = []

    if 'hu' in feature_types:
        hu_features = extract_hu_moments(image)
        features.append(hu_features)

    if 'zernike' in feature_types:
        zernike_features = extract_zernike_moments(image)
        features.append(zernike_features)

    if 'geometric' in feature_types:
        geometric_features = extract_geometric_features(image)
        features.append(geometric_features)

    if 'hog' in feature_types:
        hog_features = extract_hog_features(image)
        features.append(hog_features)

    # Concatenate all features
    return np.concatenate(features)

def extract_dataset_features(images, feature_types=None, verbose=True):
    """
    Extract features from a dataset of images.

    Parameters
    ----------
    images : list
        List of images.
    feature_types : list, optional
        List of feature types to extract. If None, extract all features.
    verbose : bool, optional
        Whether to display progress bar.

    Returns
    -------
    features : ndarray
        Extracted features for all images.
    """
    if verbose:
        iterator = tqdm(images, desc="Extracting features")
    else:
        iterator = images

    features = []
    for image in iterator:
        image_features = extract_features(image, feature_types)
        features.append(image_features)

    return np.array(features)

def main():
    """
    Main function to demonstrate feature extraction capabilities.
    This function can be run directly to test the feature extraction module.
    """
    # Create output directory
    os.makedirs('output/machine_learning', exist_ok=True)

    print("Feature Extraction Module Demo")
    print("==============================")

    # Load dataset
    print("\nLoading Kimia dataset...")
    data_dir = 'data/images_Kimia'
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Using sample image instead.")
        # Create a simple binary image for demonstration
        sample_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(sample_image, (50, 50), 30, 255, -1)
        images = [sample_image]
        labels = [0]
        class_names = ['Circle']
    else:
        images, labels, class_names = load_kimia_dataset(data_dir, max_images_per_class=5)
        print(f"Loaded {len(images)} images from {len(class_names)} classes")

    # Display a sample image
    if len(images) > 0:
        plt.figure(figsize=(6, 6))
        plt.imshow(images[0], cmap='gray')
        plt.title(f"Sample Image: {class_names[labels[0]]}")
        plt.axis('off')
        plt.savefig('output/machine_learning/sample_image.png')
        plt.close()
        print(f"Sample image saved to output/machine_learning/sample_image.png")

    # Extract features
    print("\nExtracting features...")
    feature_types = ['hu', 'zernike', 'geometric']

    # Extract features for each type and display information
    for feature_type in feature_types:
        print(f"\nExtracting {feature_type} features:")
        features = extract_dataset_features(images, feature_types=[feature_type], verbose=False)
        print(f"- Shape: {features.shape}")
        print(f"- Mean: {np.mean(features):.4f}")
        print(f"- Std: {np.std(features):.4f}")
        print(f"- Min: {np.min(features):.4f}")
        print(f"- Max: {np.max(features):.4f}")

    # Extract all features
    print("\nExtracting all features:")
    all_features = extract_dataset_features(images, feature_types=feature_types, verbose=False)
    print(f"- Shape: {all_features.shape}")
    print(f"- Total features per image: {all_features.shape[1]}")

    print("\nFeature extraction completed successfully!")

if __name__ == "__main__":
    main()
