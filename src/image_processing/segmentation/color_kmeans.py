#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation

Image segmentation techniques including K-means clustering and other segmentation algorithms.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Color image segmentation using K-means in 3D

This script demonstrates the application of k-means clustering for segmenting
a color image based on its RGB components. The image is transformed into a
vector of size N × 3 (where N is the number of pixels) and then clustered
using the k-means algorithm.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from skimage import io
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Default image to use for segmentation
DEFAULT_IMAGE = "Tv16.png"  # Color image of muscle cells

def load_image(image_path=None):
    """
    Load the color image for segmentation.

    Args:
        image_path (str, optional): Path to the image file. If None, uses the default image.

    Returns:
        ndarray: Color image
    """
    # Load the image
    if image_path is None:
        # Use the default image from data directory
        image_path = get_data_path(DEFAULT_IMAGE)
        image_name = DEFAULT_IMAGE
    else:
        # Check if the path is absolute or relative
        if os.path.isabs(image_path):
            # Absolute path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_name = os.path.basename(image_path)
        else:
            # Try as a relative path
            if os.path.exists(image_path):
                image_name = os.path.basename(image_path)
            else:
                # Try as a file in the data directory
                data_path = get_data_path(image_path)
                if os.path.exists(data_path):
                    image_path = data_path
                    image_name = image_path
                else:
                    raise FileNotFoundError(f"Image not found: {image_path}")

    # Load the image
    image = io.imread(image_path)
    print(f"Image loaded: {image_name}")
    print(f"Image shape: {image.shape}")

    return image

def apply_kmeans_segmentation(image, n_clusters=3):
    """
    Apply K-means clustering to segment a color image.

    Args:
        image (ndarray): Color image
        n_clusters (int): Number of clusters (default: 3)

    Returns:
        tuple: (labels, centers, reshaped_labels)
            - labels: Cluster labels for each pixel
            - centers: Cluster centers (colors)
            - reshaped_labels: Labels reshaped to the original image dimensions
    """
    # Get image dimensions
    nLines, nCols, nChannels = image.shape

    # Reshape the image to a 2D array of pixels
    # Each row is a pixel with RGB values
    data = np.reshape(image, (nLines*nCols, nChannels))

    # Convert to float32 for better precision
    data = data.astype(np.float32)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)

    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Reshape the labels to the original image dimensions
    reshaped_labels = np.reshape(labels, (nLines, nCols))

    return labels, centers, reshaped_labels

def visualize_color_space(data, labels, centers, sample_size=5000):
    """
    Visualize the color distribution in 3D space.

    Args:
        data (ndarray): Pixel data (N x 3)
        labels (ndarray): Cluster labels
        centers (ndarray): Cluster centers
        sample_size (int): Number of points to sample for visualization
    """
    # Sample a subset of points for visualization
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sampled_data = data[indices]
        sampled_labels = labels[indices]
    else:
        sampled_data = data
        sampled_labels = labels

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the sampled points
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    for i in range(len(centers)):
        ax.scatter(
            sampled_data[sampled_labels == i, 0],
            sampled_data[sampled_labels == i, 1],
            sampled_data[sampled_labels == i, 2],
            c=colors[i % len(colors)],
            s=10,
            alpha=0.5,
            label=f'Cluster {i+1}'
        )

    # Plot the cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c='black',
        s=200,
        marker='*',
        label='Cluster Centers'
    )

    # Set labels and title
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('3D Color Space Clustering')

    # Add a legend
    ax.legend()

    plt.tight_layout()
    plt.show()

def visualize_segmented_image(original_image, labels, centers, n_clusters):
    """
    Visualize the original image and the segmented result.

    Args:
        original_image (ndarray): Original color image
        labels (ndarray): Cluster labels
        centers (ndarray): Cluster centers (colors)
        n_clusters (int): Number of clusters
    """
    # Create a segmented image by replacing each pixel with its cluster center
    segmented_image = np.zeros_like(original_image)
    nLines, nCols = labels.shape

    # Assign the color of each cluster center to the corresponding pixels
    for i in range(nLines):
        for j in range(nCols):
            cluster_idx = labels[i, j]
            segmented_image[i, j] = centers[cluster_idx]

    # Convert to uint8 for display
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the segmented image
    axes[1].imshow(segmented_image)
    axes[1].set_title(f'Segmented Image (K={n_clusters})')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def main(image_path=None, n_clusters=3, output_path=None):
    """
    Main function to demonstrate color image segmentation using K-means.

    Args:
        image_path (str, optional): Path to the image file. If None, uses command line args or default.
        n_clusters (int, optional): Number of clusters for K-means. Default is 3.
        output_path (str, optional): Path to save the segmented image. If None, doesn't save.
    """
    # If no arguments were provided, parse from command line
    if image_path is None and n_clusters == 3 and output_path is None:
        import argparse
        parser = argparse.ArgumentParser(description='Color K-means segmentation')
        parser.add_argument('--image', type=str, default=None, help='Path to the image file')
        parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for K-means')
        parser.add_argument('--output', type=str, default=None, help='Path to save the segmented image')
        args, _ = parser.parse_known_args()

        # Update variables with command line arguments
        image_path = args.image
        n_clusters = args.clusters
        output_path = args.output

    # Load the image
    try:
        image = load_image(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Print info
    print(f"Using {n_clusters} clusters for K-means segmentation")

    # Apply K-means segmentation
    labels, centers, reshaped_labels = apply_kmeans_segmentation(image, n_clusters)

    # Create segmented image
    nLines, nCols, nChannels = image.shape
    segmented_image = np.zeros_like(image)
    for i in range(nLines):
        for j in range(nCols):
            cluster_idx = labels[i * nCols + j]
            segmented_image[i, j] = centers[cluster_idx]
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

    # Save the segmented image if output path is provided
    if output_path:
        try:
            io.imsave(output_path, segmented_image)
            print(f"Segmented image saved to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    # Get image dimensions and reshape data for visualization
    nLines, nCols, nChannels = image.shape
    data = np.reshape(image, (nLines*nCols, nChannels)).astype(np.float32)

    # Visualize the color distribution in 3D space
    visualize_color_space(data, labels, centers)

    # Visualize the segmented image
    visualize_segmented_image(image, reshaped_labels, centers, n_clusters)

    # Create a figure showing all results in one view
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Segmented image
    segmented_image = np.zeros_like(image)
    # Create a segmented image by mapping each pixel to its cluster center
    for i in range(nLines):
        for j in range(nCols):
            cluster_idx = labels[i * nCols + j]
            segmented_image[i, j] = centers[cluster_idx]
    segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)

    plt.subplot(2, 2, 2)
    plt.imshow(segmented_image)
    plt.title(f'Segmented Image (K={n_clusters})')
    plt.axis('off')

    # Cluster map (each cluster shown with a different color)
    plt.subplot(2, 2, 3)
    plt.imshow(reshaped_labels, cmap='viridis')
    plt.title('Cluster Map')
    plt.axis('off')
    plt.colorbar(ticks=range(n_clusters), label='Cluster ID')

    # Histogram of cluster distribution
    plt.subplot(2, 2, 4)
    plt.hist(labels, bins=n_clusters, rwidth=0.8, align='mid')
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Pixels')
    plt.xticks(range(n_clusters))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
else:
    # When imported from main.py, get the arguments from sys.argv
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'color-kmeans':
        # Extract arguments from sys.argv
        image_path = None
        n_clusters = 3
        output_path = None

        # Parse command line arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--image' and i + 1 < len(sys.argv):
                image_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--clusters' and i + 1 < len(sys.argv):
                try:
                    n_clusters = int(sys.argv[i + 1])
                except ValueError:
                    print(f"Error: Invalid number of clusters: {sys.argv[i + 1]}")
                i += 2
            elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        # Run the main function with the parsed arguments
        main(image_path, n_clusters, output_path)
