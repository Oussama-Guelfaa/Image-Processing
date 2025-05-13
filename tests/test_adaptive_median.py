#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the adaptive median filter.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the denoising module
from src.image_processing.denoising import (
    generate_salt_pepper_noise,
    add_noise_to_image,
    apply_mean_filter,
    apply_median_filter,
    adaptive_median_filter,
    fast_adaptive_median_filter
)

def test_adaptive_median_filter():
    """Test the adaptive median filter."""
    print("Testing adaptive median filter...")

    # Load the test image - using jambe.png (converted from jambe.tif)
    image_path = os.path.join("data", "jambe.png")
    image = img_as_float(io.imread(image_path, as_gray=True))

    # Add salt and pepper noise
    noise_level = 0.1  # 10% of pixels affected
    noisy_image = add_noise_to_image(image, 'salt_pepper', a=0.3, b=0.7, p=noise_level)

    # Apply different filters
    mean_filtered = apply_mean_filter(noisy_image, kernel_size=3)
    median_filtered = apply_median_filter(noisy_image, kernel_size=3)
    adaptive_median_filtered = adaptive_median_filter(noisy_image, max_window_size=7)
    fast_adaptive_median_filtered = fast_adaptive_median_filter(noisy_image, max_window_size=7)

    # Calculate PSNR
    psnr_noisy = psnr(image, noisy_image)
    psnr_mean = psnr(image, mean_filtered)
    psnr_median = psnr(image, median_filtered)
    psnr_adaptive = psnr(image, adaptive_median_filtered)
    psnr_fast_adaptive = psnr(image, fast_adaptive_median_filtered)

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title(f'Salt & Pepper Noise\nPSNR: {psnr_noisy:.2f} dB')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mean_filtered, cmap='gray')
    axes[0, 2].set_title(f'Mean Filter\nPSNR: {psnr_mean:.2f} dB')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(median_filtered, cmap='gray')
    axes[1, 0].set_title(f'Median Filter\nPSNR: {psnr_median:.2f} dB')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(adaptive_median_filtered, cmap='gray')
    axes[1, 1].set_title(f'Adaptive Median Filter\nPSNR: {psnr_adaptive:.2f} dB')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(fast_adaptive_median_filtered, cmap='gray')
    axes[1, 2].set_title(f'Fast Adaptive Median Filter\nPSNR: {psnr_fast_adaptive:.2f} dB')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('output/adaptive_median_results.png')
    plt.close()

    # Print results
    print("\nFilter Performance Comparison:")
    print(f"{'Filter':<25} {'PSNR (dB)':<10}")
    print("-" * 35)
    print(f"{'Noisy Image':<25} {psnr_noisy:<10.2f}")
    print(f"{'Mean Filter':<25} {psnr_mean:<10.2f}")
    print(f"{'Median Filter':<25} {psnr_median:<10.2f}")
    print(f"{'Adaptive Median Filter':<25} {psnr_adaptive:<10.2f}")
    print(f"{'Fast Adaptive Median Filter':<25} {psnr_fast_adaptive:<10.2f}")

    print("\nAdaptive median filter test completed.")

def test_noise_levels():
    """Test the adaptive median filter with different noise levels."""
    print("Testing adaptive median filter with different noise levels...")

    # Load the test image - using jambe.png (converted from jambe.tif)
    image_path = os.path.join("data", "jambe.png")
    image = img_as_float(io.imread(image_path, as_gray=True))

    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Test different noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Store PSNR values
    psnr_values = {
        'Noisy': [],
        'Mean': [],
        'Median': [],
        'Adaptive Median': [],
        'Fast Adaptive Median': []
    }

    for noise_level in noise_levels:
        # Add salt and pepper noise
        noisy_image = add_noise_to_image(image, 'salt_pepper', a=0.3, b=0.7, p=noise_level)

        # Apply different filters
        mean_filtered = apply_mean_filter(noisy_image, kernel_size=3)
        median_filtered = apply_median_filter(noisy_image, kernel_size=3)
        adaptive_median_filtered = adaptive_median_filter(noisy_image, max_window_size=7)
        fast_adaptive_median_filtered = fast_adaptive_median_filter(noisy_image, max_window_size=7)

        # Calculate PSNR
        psnr_values['Noisy'].append(psnr(image, noisy_image))
        psnr_values['Mean'].append(psnr(image, mean_filtered))
        psnr_values['Median'].append(psnr(image, median_filtered))
        psnr_values['Adaptive Median'].append(psnr(image, adaptive_median_filtered))
        psnr_values['Fast Adaptive Median'].append(psnr(image, fast_adaptive_median_filtered))

    # Plot PSNR vs. noise level
    plt.figure(figsize=(10, 6))

    plt.plot(noise_levels, psnr_values['Noisy'], 'k-o', label='Noisy Image')
    plt.plot(noise_levels, psnr_values['Mean'], 'b-o', label='Mean Filter')
    plt.plot(noise_levels, psnr_values['Median'], 'g-o', label='Median Filter')
    plt.plot(noise_levels, psnr_values['Adaptive Median'], 'r-o', label='Adaptive Median Filter')
    plt.plot(noise_levels, psnr_values['Fast Adaptive Median'], 'm-o', label='Fast Adaptive Median Filter')

    plt.xlabel('Noise Level (p)')
    plt.ylabel('PSNR (dB)')
    plt.title('Filter Performance vs. Noise Level')
    plt.legend()
    plt.grid(True)

    plt.savefig('output/adaptive_median_noise_levels.png')
    plt.close()

    print("Noise level test completed.")

def main():
    """Main function."""
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)

    # Run the tests
    test_adaptive_median_filter()
    test_noise_levels()

if __name__ == "__main__":
    main()
