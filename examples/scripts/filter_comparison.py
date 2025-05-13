#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison of different filters for salt and pepper noise removal.

This script demonstrates the application of different filters on an image with
salt and pepper noise, similar to the figure shown in the TP.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.morphology import disk
from skimage.filters.rank import mean, median, minimum, maximum
from skimage.util import random_noise

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the denoising module
from src.image_processing.denoising import (
    generate_salt_pepper_noise,
    add_noise_to_image,
    adaptive_median_filter
)

def apply_max_filter(image, kernel_size=3):
    """
    Apply a maximum filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    from skimage import img_as_ubyte, img_as_float
    image_uint8 = img_as_ubyte(image)

    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)

    # Apply maximum filter
    filtered_uint8 = maximum(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)

    return filtered

def apply_min_filter(image, kernel_size=3):
    """
    Apply a minimum filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    from skimage import img_as_ubyte, img_as_float
    image_uint8 = img_as_ubyte(image)

    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)

    # Apply minimum filter
    filtered_uint8 = minimum(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)

    return filtered

def apply_mean_filter(image, kernel_size=3):
    """
    Apply a mean filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    from skimage import img_as_ubyte, img_as_float
    image_uint8 = img_as_ubyte(image)

    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)

    # Apply mean filter
    filtered_uint8 = mean(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)

    return filtered

def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    from skimage import img_as_ubyte, img_as_float
    image_uint8 = img_as_ubyte(image)

    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)

    # Apply median filter
    filtered_uint8 = median(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)

    return filtered

def main():
    """Main function."""
    # Load the image
    image_path = os.path.join("data", "jambe.png")
    image = img_as_float(io.imread(image_path, as_gray=True))

    # Add salt and pepper noise
    noise_level = 0.1  # 10% of pixels affected
    noisy_image = random_noise(image, mode='s&p', amount=noise_level, salt_vs_pepper=0.5)

    # Apply different filters
    kernel_size = 5  # Increase kernel size for better filtering
    max_filtered = apply_max_filter(noisy_image, kernel_size)
    min_filtered = apply_min_filter(noisy_image, kernel_size)
    mean_filtered = apply_mean_filter(noisy_image, kernel_size)
    median_filtered = apply_median_filter(noisy_image, kernel_size)
    adaptive_median_filtered = adaptive_median_filter(noisy_image, max_window_size=7)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Create a figure similar to the one in the TP
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle("Figure 5.7: Different filters applied to the noisy image. The median filter is particularly adapted\nin the case of salt and pepper noise (impulse noise), but still destroy the structures observed in the images.", fontsize=10, y=0.98)

    # Create a 3x2 grid for the subplots
    gs = fig.add_gridspec(3, 2, hspace=0.2, wspace=0.1)

    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('(a) Original image.')
    ax1.axis('off')

    # Noisy image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(noisy_image, cmap='gray')
    ax2.set_title('(b) Noisy image (salt and pepper).')
    ax2.axis('off')

    # Maximum filter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(max_filtered, cmap='gray')
    ax3.set_title('(c) Maximum filter.')
    ax3.axis('off')

    # Minimum filter
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(min_filtered, cmap='gray')
    ax4.set_title('(d) Minimum filter.')
    ax4.axis('off')

    # Mean filter
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(mean_filtered, cmap='gray')
    ax5.set_title('(e) Mean filter.')
    ax5.axis('off')

    # Median filter
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(median_filtered, cmap='gray')
    ax6.set_title('(f) Median filter.')
    ax6.axis('off')

    # Adjust layout
    plt.subplots_adjust(top=0.92)
    plt.savefig('output/filter_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a second figure to compare median and adaptive median filters
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original image')
    axes[0].axis('off')

    # Median filter
    axes[1].imshow(median_filtered, cmap='gray')
    axes[1].set_title('Median filter')
    axes[1].axis('off')

    # Adaptive median filter
    axes[2].imshow(adaptive_median_filtered, cmap='gray')
    axes[2].set_title('Adaptive median filter')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('output/median_vs_adaptive.png', dpi=300)
    plt.show()

    print("Filter comparison completed. Results saved to output directory.")

if __name__ == "__main__":
    main()
