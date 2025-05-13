#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive denoising script for VS Code.

This script demonstrates the denoising functionality in VS Code's interactive window.
Run this script with the "Run Current File in Interactive Window" command in VS Code.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for interactive display
import matplotlib
# Use the 'Agg' backend which doesn't require a GUI
matplotlib.use('Agg')
# Don't use interactive mode with Agg backend
# plt.ion()

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the denoising module
from src.image_processing.denoising import (
    generate_salt_pepper_noise,
    add_noise_to_image,
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    apply_bilateral_filter,
    adaptive_median_filter,
    compare_denoising_methods
)

# Import utility functions
from src.image_processing.denoising.noise_generation import load_image

def main():
    """Main function."""
    # Load the image
    image_path = os.path.join("data", "jambe.png")
    image = load_image(image_path)

    print("Loaded image:", image_path)
    print("Image shape:", image.shape)
    print("Image min/max values:", image.min(), image.max())

    # Save the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.savefig('output/original_image.png')
    plt.close()

    # Add salt and pepper noise
    noise_level = 0.05  # 5% of pixels affected
    noisy_image = add_noise_to_image(image, 'salt_pepper', a=0.01, b=0.99, p=noise_level)

    # Save the noisy image
    plt.figure(figsize=(8, 8))
    plt.imshow(noisy_image, cmap='gray')
    plt.title("Noisy Image (Salt and Pepper)")
    plt.axis('off')
    plt.savefig('output/noisy_image.png')
    plt.close()

    # Apply median filter
    print("\nApplying median filter...")
    median_filtered = apply_median_filter(noisy_image, kernel_size=3)

    # Save the filtered image
    plt.figure(figsize=(8, 8))
    plt.imshow(median_filtered, cmap='gray')
    plt.title("Median Filtered Image")
    plt.axis('off')
    plt.savefig('output/median_filtered_image.png')
    plt.close()

    # Apply adaptive median filter
    print("\nApplying adaptive median filter...")
    adaptive_filtered = adaptive_median_filter(noisy_image, max_window_size=5)

    # Save the filtered image
    plt.figure(figsize=(8, 8))
    plt.imshow(adaptive_filtered, cmap='gray')
    plt.title("Adaptive Median Filtered Image")
    plt.axis('off')
    plt.savefig('output/adaptive_median_filtered_image.png')
    plt.close()

    # Compare all methods
    print("\nComparing all denoising methods...")
    compare_denoising_methods(image, noisy_image, save_path="output/interactive_denoising.png")

    print("\nDenoising demonstration completed.")
    print("The figures should be displayed in VS Code's interactive window.")
    print("If you don't see them, try running this script with 'Run Current File in Interactive Window'.")

if __name__ == "__main__":
    main()
