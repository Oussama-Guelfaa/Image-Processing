#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for image denoising techniques.

This module implements various denoising methods:
1. Mean filter
2. Median filter
3. Gaussian filter
4. Bilateral filter
5. Non-local means filter

Author: Oussama GUELFAA
Date: 01-05-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import denoise_bilateral, denoise_nl_means
from skimage.filters import gaussian, median
from scipy.ndimage import uniform_filter
import os
import time

from .noise_generation import load_image, add_noise_to_image

def apply_mean_filter(image, kernel_size=3):
    """
    Apply a mean filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Denoised image
    """
    # Apply uniform filter (mean filter)
    denoised = uniform_filter(image, size=kernel_size)

    return denoised

def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter to an image.

    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)

    Returns:
        ndarray: Denoised image
    """
    # Apply median filter
    from skimage.morphology import disk
    from skimage.filters import rank

    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)

    # Apply median filter using rank.median
    # Convert to uint8 for rank filters
    from skimage import img_as_ubyte, img_as_float
    image_uint8 = img_as_ubyte(image)
    denoised_uint8 = rank.median(image_uint8, selem)
    denoised = img_as_float(denoised_uint8)

    return denoised

def apply_gaussian_filter(image, sigma=1.0):
    """
    Apply a Gaussian filter to an image.

    Args:
        image (ndarray): Input image
        sigma (float): Standard deviation of the Gaussian kernel (default: 1.0)

    Returns:
        ndarray: Denoised image
    """
    # Apply Gaussian filter
    denoised = gaussian(image, sigma=sigma)

    return denoised

def apply_bilateral_filter(image, sigma_spatial=2, sigma_color=0.1):
    """
    Apply a bilateral filter to an image.

    Args:
        image (ndarray): Input image
        sigma_spatial (float): Standard deviation of the spatial kernel (default: 2)
        sigma_color (float): Standard deviation of the color kernel (default: 0.1)

    Returns:
        ndarray: Denoised image
    """
    # Apply bilateral filter
    denoised = denoise_bilateral(image, sigma_spatial=sigma_spatial, sigma_color=sigma_color)

    return denoised

def apply_nlm_filter(image, patch_size=5, patch_distance=6, h=0.1):
    """
    Apply a non-local means filter to an image.

    Args:
        image (ndarray): Input image
        patch_size (int): Size of patches used for denoising (default: 5)
        patch_distance (int): Maximum distance to search for similar patches (default: 6)
        h (float): Cut-off distance (in gray levels) (default: 0.1)

    Returns:
        ndarray: Denoised image
    """
    # Apply non-local means filter
    denoised = denoise_nl_means(image, patch_size=patch_size, patch_distance=patch_distance, h=h)

    return denoised

def compare_denoising_methods(image, noisy_image, save_path=None):
    """
    Compare different denoising methods on a noisy image.

    Args:
        image (ndarray): Original clean image
        noisy_image (ndarray): Noisy image
        save_path (str): Path to save the comparison image (default: None)

    Returns:
        tuple: Denoised images using different methods
    """
    # Import the adaptive median filter
    from .adaptive_median import adaptive_median_filter, fast_adaptive_median_filter

    # Apply different denoising methods
    start_time = time.time()
    denoised_mean = apply_mean_filter(noisy_image, kernel_size=3)
    time_mean = time.time() - start_time

    start_time = time.time()
    denoised_median = apply_median_filter(noisy_image, kernel_size=3)
    time_median = time.time() - start_time

    start_time = time.time()
    denoised_gaussian = apply_gaussian_filter(noisy_image, sigma=1.0)
    time_gaussian = time.time() - start_time

    start_time = time.time()
    denoised_bilateral = apply_bilateral_filter(noisy_image, sigma_spatial=2, sigma_color=0.1)
    time_bilateral = time.time() - start_time

    start_time = time.time()
    denoised_adaptive = adaptive_median_filter(noisy_image, max_window_size=7)
    time_adaptive = time.time() - start_time

    start_time = time.time()
    denoised_fast_adaptive = fast_adaptive_median_filter(noisy_image, max_window_size=7)
    time_fast_adaptive = time.time() - start_time

    start_time = time.time()
    denoised_nlm = apply_nlm_filter(noisy_image, patch_size=5, patch_distance=6, h=0.1)
    time_nlm = time.time() - start_time

    # Calculate PSNR for each method
    def calculate_psnr(original, denoised):
        mse = np.mean((original - denoised) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    psnr_noisy = calculate_psnr(image, noisy_image)
    psnr_mean = calculate_psnr(image, denoised_mean)
    psnr_median = calculate_psnr(image, denoised_median)
    psnr_gaussian = calculate_psnr(image, denoised_gaussian)
    psnr_bilateral = calculate_psnr(image, denoised_bilateral)
    psnr_adaptive = calculate_psnr(image, denoised_adaptive)
    psnr_fast_adaptive = calculate_psnr(image, denoised_fast_adaptive)
    psnr_nlm = calculate_psnr(image, denoised_nlm)

    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title(f'Noisy Image\nPSNR: {psnr_noisy:.2f} dB')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(denoised_mean, cmap='gray')
    axes[0, 2].set_title(f'Mean Filter\nPSNR: {psnr_mean:.2f} dB\nTime: {time_mean:.3f} s')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(denoised_median, cmap='gray')
    axes[1, 0].set_title(f'Median Filter\nPSNR: {psnr_median:.2f} dB\nTime: {time_median:.3f} s')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_gaussian, cmap='gray')
    axes[1, 1].set_title(f'Gaussian Filter\nPSNR: {psnr_gaussian:.2f} dB\nTime: {time_gaussian:.3f} s')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(denoised_bilateral, cmap='gray')
    axes[1, 2].set_title(f'Bilateral Filter\nPSNR: {psnr_bilateral:.2f} dB\nTime: {time_bilateral:.3f} s')
    axes[1, 2].axis('off')

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Denoising comparison saved to: {save_path}")

    plt.show()

    # Create a separate figure for adaptive filters and NLM due to their longer processing time
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(denoised_adaptive, cmap='gray')
    axes[0].set_title(f'Adaptive Median Filter\nPSNR: {psnr_adaptive:.2f} dB\nTime: {time_adaptive:.3f} s')
    axes[0].axis('off')

    axes[1].imshow(denoised_fast_adaptive, cmap='gray')
    axes[1].set_title(f'Fast Adaptive Median Filter\nPSNR: {psnr_fast_adaptive:.2f} dB\nTime: {time_fast_adaptive:.3f} s')
    axes[1].axis('off')

    axes[2].imshow(denoised_nlm, cmap='gray')
    axes[2].set_title(f'Non-Local Means Filter\nPSNR: {psnr_nlm:.2f} dB\nTime: {time_nlm:.3f} s')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        base_name, ext = os.path.splitext(save_path)
        advanced_path = f"{base_name}_advanced{ext}"
        plt.savefig(advanced_path, dpi=300)
        print(f"Advanced denoising results saved to: {advanced_path}")

    plt.show()

    # Print a summary of the results
    print("\nDenoising Results Summary:")
    print(f"{'Method':<20} {'PSNR (dB)':<10} {'Time (s)':<10}")
    print("-" * 40)
    print(f"{'Noisy Image':<20} {psnr_noisy:<10.2f} {'N/A':<10}")
    print(f"{'Mean Filter':<20} {psnr_mean:<10.2f} {time_mean:<10.3f}")
    print(f"{'Median Filter':<20} {psnr_median:<10.2f} {time_median:<10.3f}")
    print(f"{'Gaussian Filter':<20} {psnr_gaussian:<10.2f} {time_gaussian:<10.3f}")
    print(f"{'Bilateral Filter':<20} {psnr_bilateral:<10.2f} {time_bilateral:<10.3f}")
    print(f"{'Adaptive Median':<20} {psnr_adaptive:<10.2f} {time_adaptive:<10.3f}")
    print(f"{'Fast Adaptive Median':<20} {psnr_fast_adaptive:<10.2f} {time_fast_adaptive:<10.3f}")
    print(f"{'NLM Filter':<20} {psnr_nlm:<10.2f} {time_nlm:<10.3f}")

    return (denoised_mean, denoised_median, denoised_gaussian, denoised_bilateral,
            denoised_adaptive, denoised_fast_adaptive, denoised_nlm)

def test_denoising(image_path=None, noise_type='gaussian', **noise_params):
    """
    Test denoising methods on an image with added noise.

    Args:
        image_path (str): Path to the image file (default: None)
        noise_type (str): Type of noise to add (default: 'gaussian')
        **noise_params: Additional parameters for the noise generation

    Returns:
        tuple: Original image, noisy image, and denoised images
    """
    # Load the image
    image = load_image(image_path)

    # Set default noise parameters if not provided
    if noise_type == 'gaussian' and 'std' not in noise_params:
        noise_params['std'] = 0.1
    elif noise_type == 'uniform' and 'a' not in noise_params:
        noise_params['a'] = -0.2
        noise_params['b'] = 0.2
    elif noise_type == 'salt_pepper' and 'a' not in noise_params:
        noise_params['a'] = 0.01
        noise_params['b'] = 0.99
    elif noise_type == 'exponential' and 'a' not in noise_params:
        noise_params['a'] = 10

    # Add noise to the image
    noisy_image = add_noise_to_image(image, noise_type, **noise_params)

    # Compare denoising methods
    denoised_images = compare_denoising_methods(image, noisy_image, save_path=f"output/denoising_{noise_type}.png")

    return image, noisy_image, denoised_images

if __name__ == "__main__":
    # Test denoising with different noise types
    test_denoising(image_path="data/jambe.tif", noise_type='gaussian', std=0.1)
    test_denoising(image_path="data/jambe.tif", noise_type='salt_pepper', a=0.01, b=0.99)
    test_denoising(image_path="data/jambe.tif", noise_type='exponential', a=10)
