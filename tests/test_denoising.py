#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the denoising module.

Author: Oussama GUELFAA
Date: 01-05-2025
"""

import sys
import os
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the denoising module
from src.image_processing.denoising import (
    generate_uniform_noise,
    generate_gaussian_noise,
    generate_salt_pepper_noise,
    generate_exponential_noise,
    add_noise_to_image,
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    apply_bilateral_filter,
    apply_nlm_filter,
    compare_denoising_methods
)

def test_noise_generation():
    """Test the noise generation functions."""
    print("Testing noise generation...")

    # Generate noise samples
    shape = (256, 256)
    uniform_noise = generate_uniform_noise(shape, a=-0.5, b=0.5)
    gaussian_noise = generate_gaussian_noise(shape, mean=0, std=0.1)
    salt_pepper_noise = generate_salt_pepper_noise(shape, a=0.3, b=0.7)
    exponential_noise = generate_exponential_noise(shape, a=5)

    # Visualize the noise samples
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(uniform_noise, cmap='gray')
    axes[0, 0].set_title('Uniform Noise')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gaussian_noise, cmap='gray')
    axes[0, 1].set_title('Gaussian Noise')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(salt_pepper_noise, cmap='gray')
    axes[1, 0].set_title('Salt and Pepper Noise')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(exponential_noise, cmap='gray')
    axes[1, 1].set_title('Exponential Noise')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('output/noise_samples.png')
    plt.close()  # Close the figure instead of showing it

    print("Noise generation test completed.")

def test_denoising_filters():
    """Test the denoising filters."""
    print("Testing denoising filters...")

    # Load the test image
    try:
        image_path = os.path.join("data", "jambe.tif")
        image = img_as_float(io.imread(image_path, as_gray=True))
    except Exception as e:
        print(f"Error loading jambe.tif: {e}")
        print("Falling back to Tv16.png")
        image_path = os.path.join("data", "Tv16.png")
        image = img_as_float(io.imread(image_path, as_gray=True))

    # Add Gaussian noise
    noisy_image = add_noise_to_image(image, 'gaussian', mean=0, std=0.1)

    # Apply different denoising filters
    denoised_mean = apply_mean_filter(noisy_image, kernel_size=3)
    denoised_median = apply_median_filter(noisy_image, kernel_size=3)
    denoised_gaussian = apply_gaussian_filter(noisy_image, sigma=1.0)
    denoised_bilateral = apply_bilateral_filter(noisy_image, sigma_spatial=2, sigma_color=0.1)

    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title('Noisy Image (Gaussian)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised_mean, cmap='gray')
    axes[0, 2].set_title('Mean Filter')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(denoised_median, cmap='gray')
    axes[1, 0].set_title('Median Filter')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_gaussian, cmap='gray')
    axes[1, 1].set_title('Gaussian Filter')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_bilateral, cmap='gray')
    axes[1, 2].set_title('Bilateral Filter')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('output/denoising_results.png')
    plt.close()  # Close the figure instead of showing it

    print("Denoising filters test completed.")

def test_compare_denoising_methods():
    """Test the compare_denoising_methods function."""
    print("Testing compare_denoising_methods...")

    # Load the test image
    try:
        image_path = os.path.join("data", "jambe.tif")
        image = img_as_float(io.imread(image_path, as_gray=True))
    except Exception as e:
        print(f"Error loading jambe.tif: {e}")
        print("Falling back to Tv16.png")
        image_path = os.path.join("data", "Tv16.png")
        image = img_as_float(io.imread(image_path, as_gray=True))

    # Add Gaussian noise
    noisy_image = add_noise_to_image(image, 'gaussian', mean=0, std=0.1)

    # Compare denoising methods
    denoised_images = compare_denoising_methods(image, noisy_image, save_path='output/denoising_comparison.png')

    print("Compare denoising methods test completed.")

def main():
    """Main function."""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Run the tests
    test_noise_generation()
    test_denoising_filters()
    test_compare_denoising_methods()

if __name__ == "__main__":
    main()
