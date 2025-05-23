#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Region Growing Segmentation

Implementation of the region growing segmentation algorithm.

Author: Oussama GUELFAA
Date: 23-05-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from collections import deque
import os
import sys
import argparse


# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# Define utility functions if running as standalone script
def get_project_root():
    """Get the absolute path to the project root directory."""
    return project_root

def get_data_path(filename):
    """Get the absolute path to a file in the data directory."""
    return os.path.join(get_project_root(), 'data', filename)

def ensure_output_dir(output_dir):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Try to import from the package if available
try:
    from src.utils.path_utils import get_data_path, get_project_root
except ImportError:
    # Already defined above, so we can continue
    pass

# Default image to use for segmentation
DEFAULT_IMAGE = "jambe.tif"

def load_image(image_path=None):
    """
    Load an image for segmentation.

    Args:
        image_path (str, optional): Path to the image file. If None, uses the default image.

    Returns:
        ndarray: Grayscale image as float array
    """
    # Try different default images in case the primary one has issues
    default_images = [
        DEFAULT_IMAGE,
        "blood.jpg",
        "osteoblaste.jpg",
        "cornee.png"
    ]

    if image_path is None:
        # Try to load one of the default images
        for img_name in default_images:
            try:
                image_path = get_data_path(img_name)
                if os.path.exists(image_path):
                    print(f"Using default image: {img_name}")
                    break
            except Exception as e:
                print(f"Error loading {img_name}: {e}")
                continue
    else:
        # Use the provided image path
        if not os.path.isabs(image_path):
            # If the path is relative, try to find it in the data directory
            data_path = get_data_path(image_path)
            if os.path.exists(data_path):
                image_path = data_path
        print(f"Using image: {image_path}")

    # Load the image and convert to grayscale if needed
    try:
        image = io.imread(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Generating a synthetic test image instead...")
        # Create a synthetic test image
        image = np.zeros((256, 256), dtype=np.float32)
        # Add a circle in the middle
        center = (128, 128)
        radius = 64
        y, x = np.ogrid[:256, :256]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        image[mask] = 1.0
        # Add some noise
        image += 0.1 * np.random.randn(256, 256)
        # Clip values to [0, 1]
        image = np.clip(image, 0, 1)
        return image

    if len(image.shape) > 2:
        image = rgb2gray(image)

    # Convert to float for processing
    image = img_as_float(image)

    return image

def predicate_intensity_diff(image, i, j, seed, threshold=20):
    """
    Predicate function based on intensity difference between pixel and seed.

    Args:
        image (ndarray): Input image
        i (int): Pixel row coordinate
        j (int): Pixel column coordinate
        seed (tuple): Seed pixel coordinates (row, col)
        threshold (float, optional): Maximum allowed intensity difference

    Returns:
        bool: True if pixel should be included in the region
    """
    # Get pixel intensities
    pixel_value = image[i, j]
    seed_value = image[seed[0], seed[1]]

    # Check if intensity difference is below threshold
    return abs(float(pixel_value) - float(seed_value)) <= threshold/255.0

def predicate_region_mean(image, i, j, region_mask, threshold=20):
    """
    Predicate function based on intensity difference between pixel and region mean.

    Args:
        image (ndarray): Input image
        i (int): Pixel row coordinate
        j (int): Pixel column coordinate
        region_mask (ndarray): Boolean mask of current region
        threshold (float, optional): Maximum allowed intensity difference

    Returns:
        bool: True if pixel should be included in the region
    """
    # Get pixel intensity
    pixel_value = image[i, j]

    # Calculate region mean (only for pixels in the region)
    if np.sum(region_mask) > 0:
        region_mean = np.mean(image[region_mask])
    else:
        # If region is empty, use the pixel value itself
        region_mean = pixel_value

    # Check if intensity difference is below threshold
    return abs(float(pixel_value) - float(region_mean)) <= threshold/255.0

def predicate_adaptive_threshold(image, i, j, region_mask, T0=20):
    """
    Predicate function with adaptive threshold based on region statistics.

    Args:
        image (ndarray): Input image
        i (int): Pixel row coordinate
        j (int): Pixel column coordinate
        region_mask (ndarray): Boolean mask of current region
        T0 (float, optional): Base threshold value

    Returns:
        bool: True if pixel should be included in the region
    """
    # Get pixel intensity
    pixel_value = image[i, j]

    # Calculate region statistics (only for pixels in the region)
    if np.sum(region_mask) > 0:
        region_mean = np.mean(image[region_mask])
        region_std = np.std(image[region_mask])

        # Avoid division by zero
        if region_mean == 0:
            region_mean = 1e-10

        # Calculate adaptive threshold
        T = (1 - region_std/region_mean) * T0/255.0

        # Ensure T is positive
        T = max(T, 0.01/255.0)
    else:
        # If region is empty, use a fixed threshold
        T = T0/255.0
        region_mean = pixel_value

    # Check if intensity difference is below threshold
    return abs(float(pixel_value) - float(region_mean)) <= T

def region_growing(image, seed, predicate_func=predicate_intensity_diff, **kwargs):
    """
    Perform region growing segmentation starting from a seed point.

    Args:
        image (ndarray): Input image
        seed (tuple): Seed pixel coordinates (row, col)
        predicate_func (function): Function that determines if a pixel belongs to the region
        **kwargs: Additional arguments for the predicate function

    Returns:
        ndarray: Binary mask of the segmented region
    """
    # Get image dimensions
    rows, cols = image.shape

    # Initialize visited matrix and result
    visited = np.zeros_like(image, dtype=bool)

    # Mark the seed as visited and part of the region
    visited[seed[0], seed[1]] = True

    # Initialize queue with seed
    queue = deque([seed])

    # Process pixels in the queue
    while queue:
        # Get next pixel from queue
        current = queue.popleft()
        i, j = current

        # Check 8-connected neighbors
        for ni in range(max(0, i-1), min(rows, i+2)):
            for nj in range(max(0, j-1), min(cols, j+2)):
                # Skip if already visited
                if visited[ni, nj]:
                    continue

                # Check which predicate function we're using and call it with appropriate arguments
                include_pixel = False

                if predicate_func == predicate_intensity_diff:
                    # Extract threshold from kwargs if present
                    threshold = kwargs.get('threshold', 20)
                    include_pixel = predicate_func(image, ni, nj, seed, threshold=threshold)
                elif predicate_func == predicate_region_mean:
                    # Extract threshold from kwargs if present
                    threshold = kwargs.get('threshold', 20)
                    include_pixel = predicate_func(image, ni, nj, visited, threshold=threshold)
                elif predicate_func == predicate_adaptive_threshold:
                    # Extract T0 from kwargs if present
                    T0 = kwargs.get('threshold', 20)
                    include_pixel = predicate_func(image, ni, nj, visited, T0=T0)
                else:
                    # Generic case - try to determine arguments from function signature
                    try:
                        if 'seed' in predicate_func.__code__.co_varnames and 'region_mask' in predicate_func.__code__.co_varnames:
                            include_pixel = predicate_func(image, ni, nj, seed, region_mask=visited, **kwargs)
                        elif 'seed' in predicate_func.__code__.co_varnames:
                            include_pixel = predicate_func(image, ni, nj, seed, **kwargs)
                        elif 'region_mask' in predicate_func.__code__.co_varnames:
                            include_pixel = predicate_func(image, ni, nj, region_mask=visited, **kwargs)
                        else:
                            include_pixel = predicate_func(image, ni, nj, **kwargs)
                    except Exception as e:
                        print(f"Error calling predicate function: {e}")
                        include_pixel = False

                if include_pixel:
                    # Mark as visited and part of the region
                    visited[ni, nj] = True
                    # Add to queue for further processing
                    queue.append((ni, nj))

    return visited

def onclick(event, image):
    """
    Mouse click event handler for selecting seed point.

    Args:
        event: Mouse click event
        image: Image being processed

    Returns:
        tuple: Seed coordinates (row, col)
    """
    if event.xdata is not None and event.ydata is not None:
        col = int(event.xdata)
        row = int(event.ydata)
        print(f"Seed selected at: ({row}, {col})")
        return (row, col)
    return None

def interactive_region_growing(image, predicate_funcs=None, thresholds=None):
    """
    Interactive region growing with manual seed selection.

    Args:
        image (ndarray): Input image
        predicate_funcs (list, optional): List of predicate functions to use
        thresholds (list, optional): List of thresholds for each predicate function

    Returns:
        list: List of segmentation results for each predicate function
    """
    if predicate_funcs is None:
        predicate_funcs = [predicate_intensity_diff, predicate_region_mean, predicate_adaptive_threshold]

    if thresholds is None:
        thresholds = [20, 20, 20]  # Default thresholds

    # Create figure for seed selection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.set_title('Click on a point to select seed')
    plt.tight_layout()

    # Wait for mouse click to select seed
    seed = None
    def on_click(event):
        nonlocal seed
        if event.xdata is not None and event.ydata is not None:
            seed = (int(event.ydata), int(event.xdata))
            plt.close(fig)

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    if seed is None:
        print("No seed selected. Using default seed at center of image.")
        seed = (image.shape[0] // 2, image.shape[1] // 2)

    print(f"Using seed at: {seed}")

    # Apply region growing with each predicate function
    results = []
    for i, (func, threshold) in enumerate(zip(predicate_funcs, thresholds)):
        print(f"Applying region growing with {func.__name__} (threshold={threshold})...")
        result = region_growing(image, seed, func, threshold=threshold)
        results.append(result)

    return results, seed

def main(image_path=None, output_dir=None):
    """
    Main function to run region growing segmentation.

    Args:
        image_path (str, optional): Path to the input image
        output_dir (str, optional): Directory to save output images
    """
    # Load the image
    image = load_image(image_path)

    # Define predicate functions and thresholds
    predicate_funcs = [
        predicate_intensity_diff,
        predicate_region_mean,
        predicate_adaptive_threshold
    ]

    thresholds = [20, 20, 20]  # Default thresholds

    # Run interactive region growing
    results, seed = interactive_region_growing(image, predicate_funcs, thresholds)

    # Visualize results
    fig = plt.figure(figsize=(15, 10))

    # Original image with seed point
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.plot(seed[1], seed[0], 'ro', markersize=10)
    ax1.set_title('Original Image with Seed Point')
    ax1.axis('off')

    # Results for each predicate function
    for i, (result, func) in enumerate(zip(results, predicate_funcs)):
        ax = fig.add_subplot(2, 2, i+2)
        ax.imshow(result, cmap='gray')
        ax.set_title(f'Segmentation with {func.__name__}')
        ax.axis('off')

    plt.tight_layout()

    # Save results if output directory is specified
    if output_dir:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save original image with seed
        fig_original = plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.plot(seed[1], seed[0], 'ro', markersize=10)
        plt.title('Original Image with Seed Point')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'original_with_seed.png'))
        plt.close(fig_original)

        # Save each segmentation result
        for i, (result, func) in enumerate(zip(results, predicate_funcs)):
            fig_result = plt.figure(figsize=(8, 8))
            plt.imshow(result, cmap='gray')
            plt.title(f'Segmentation with {func.__name__}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'segmentation_{func.__name__}.png'))
            plt.close(fig_result)

        # Save combined figure
        plt.savefig(os.path.join(output_dir, 'region_growing_results.png'))
        print(f"Results saved to {output_dir}")

    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Region Growing Segmentation')
    parser.add_argument('--image', type=str, default=None, help='Path to the input image')
    parser.add_argument('--output', type=str, default='output/segmentation/region_growing',
                        help='Directory to save output images')
    args = parser.parse_args()

    # Ensure output directory exists
    if args.output:
        ensure_output_dir(args.output)

    # Run the main function
    main(args.image, args.output)
