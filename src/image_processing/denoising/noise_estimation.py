#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising

Techniques for removing noise from images, including various filtering methods.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import io, img_as_float
import os

def extract_roi(image, roi=None, interactive=False):
    """
    Extract a Region of Interest (ROI) from an image.
    
    Args:
        image (ndarray): Input image
        roi (tuple): ROI coordinates (x_min, y_min, x_max, y_max) (default: None)
        interactive (bool): Whether to use interactive selection (default: False)
        
    Returns:
        ndarray: Extracted ROI
    """
    if interactive:
        # Use interactive selection
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image, cmap='gray')
        ax.set_title('Select ROI (uniform intensity region) and press Enter')
        
        # Initialize ROI coordinates
        roi_coords = [0, 0, 0, 0]
        
        def line_select_callback(eclick, erelease):
            """Callback for line selection."""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            roi_coords[0] = min(x1, x2)
            roi_coords[1] = min(y1, y2)
            roi_coords[2] = max(x1, x2)
            roi_coords[3] = max(y1, y2)
        
        def toggle_selector(event):
            """Toggle selector on/off."""
            if event.key == 'enter':
                plt.close()
        
        # Create the RectangleSelector
        rect_selector = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1],  # Left mouse button only
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        plt.connect('key_press_event', toggle_selector)
        plt.show()
        
        # Extract the ROI using the selected coordinates
        x_min, y_min, x_max, y_max = roi_coords
    elif roi is not None:
        # Use provided ROI coordinates
        x_min, y_min, x_max, y_max = roi
    else:
        # Use default ROI (center of the image)
        h, w = image.shape
        x_min, y_min = w // 4, h // 4
        x_max, y_max = 3 * w // 4, 3 * h // 4
    
    # Extract the ROI
    roi_image = image[y_min:y_max, x_min:x_max]
    
    # Visualize the selected ROI
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with ROI highlighted
    axes[0].imshow(image, cmap='gray')
    axes[0].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  fill=False, edgecolor='red', linewidth=2))
    axes[0].set_title('Original Image with ROI')
    axes[0].axis('off')
    
    # Extracted ROI
    axes[1].imshow(roi_image, cmap='gray')
    axes[1].set_title('Extracted ROI')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return roi_image, (x_min, y_min, x_max, y_max)

def visualize_roi_histogram(roi_image, bins=256, title="ROI Histogram"):
    """
    Visualize the histogram of a Region of Interest (ROI).
    
    Args:
        roi_image (ndarray): ROI image
        bins (int): Number of bins for the histogram (default: 256)
        title (str): Title for the histogram plot (default: "ROI Histogram")
    """
    # Compute the histogram
    hist, bin_edges = np.histogram(roi_image, bins=bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Visualize the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist, width=1/bins, alpha=0.7)
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()
    
    return hist, bin_centers

def estimate_noise_parameters(roi_image, noise_type='gaussian'):
    """
    Estimate noise parameters from a Region of Interest (ROI).
    
    Args:
        roi_image (ndarray): ROI image
        noise_type (str): Type of noise to estimate ('gaussian', 'uniform', 'salt_pepper', 'exponential')
        
    Returns:
        dict: Estimated noise parameters
    """
    # Compute basic statistics
    mean_val = np.mean(roi_image)
    std_val = np.std(roi_image)
    min_val = np.min(roi_image)
    max_val = np.max(roi_image)
    
    # Estimate parameters based on noise type
    if noise_type == 'gaussian':
        # For Gaussian noise, estimate mean and standard deviation
        params = {
            'mean': mean_val,
            'std': std_val
        }
    elif noise_type == 'uniform':
        # For uniform noise, estimate lower and upper bounds
        params = {
            'a': min_val,
            'b': max_val
        }
    elif noise_type == 'salt_pepper':
        # For salt and pepper noise, estimate thresholds
        # This is a simplified estimation
        hist, bin_edges = np.histogram(roi_image, bins=256, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks in the histogram
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bin_centers[i], hist[i]))
        
        # Sort peaks by height
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Use the top peaks as thresholds
        if len(peaks) >= 2:
            a, b = sorted([peaks[0][0], peaks[1][0]])
        else:
            a, b = 0.3, 0.7  # Default values
        
        params = {
            'a': a,
            'b': b
        }
    elif noise_type == 'exponential':
        # For exponential noise, estimate the scale parameter
        # The mean of an exponential distribution is 1/a
        a = 1 / mean_val if mean_val > 0 else 1
        
        params = {
            'a': a
        }
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Print the estimated parameters
    print(f"Estimated {noise_type} noise parameters:")
    for param, value in params.items():
        print(f"  {param}: {value:.4f}")
    
    return params

def analyze_noisy_image(original_image, noisy_image, roi=None, noise_type='gaussian'):
    """
    Analyze a noisy image by extracting a ROI and estimating noise parameters.
    
    Args:
        original_image (ndarray): Original clean image
        noisy_image (ndarray): Noisy image
        roi (tuple): ROI coordinates (x_min, y_min, x_max, y_max) (default: None)
        noise_type (str): Type of noise to estimate (default: 'gaussian')
        
    Returns:
        dict: Estimated noise parameters
    """
    # Extract ROI from both images
    if roi is None:
        # Extract ROI interactively from the original image
        _, roi = extract_roi(original_image, interactive=True)
    
    # Extract the same ROI from both images
    original_roi = original_image[roi[1]:roi[3], roi[0]:roi[2]]
    noisy_roi = noisy_image[roi[1]:roi[3], roi[0]:roi[2]]
    
    # Compute the noise by subtracting the original from the noisy image
    noise = noisy_roi - original_roi
    
    # Visualize the noise
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_roi, cmap='gray')
    axes[0].set_title('Original ROI')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_roi, cmap='gray')
    axes[1].set_title('Noisy ROI')
    axes[1].axis('off')
    
    axes[2].imshow(noise, cmap='gray')
    axes[2].set_title('Noise (Difference)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the histogram of the noise
    visualize_roi_histogram(noise, title=f"{noise_type.capitalize()} Noise Histogram")
    
    # Estimate noise parameters
    params = estimate_noise_parameters(noise, noise_type)
    
    return params

if __name__ == "__main__":
    # Load an image
    image_path = os.path.join("data", "jambe.tif")
    image = img_as_float(io.imread(image_path))
    
    # Extract ROI
    roi_image, roi_coords = extract_roi(image, interactive=True)
    
    # Visualize ROI histogram
    visualize_roi_histogram(roi_image)
    
    # Estimate noise parameters
    estimate_noise_parameters(roi_image, noise_type='gaussian')
