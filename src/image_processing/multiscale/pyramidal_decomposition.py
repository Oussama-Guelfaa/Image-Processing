#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiscale

Multiscale analysis techniques including pyramidal decomposition and scale-space decomposition.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.transform import resize, rescale
from skimage.filters import gaussian
import os

def gaussian_pyramid(image, levels=4, sigma=1):
    """
    Create a Gaussian pyramid by successively blurring and downsampling the image.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    levels : int
        Number of levels in the pyramid.
    sigma : float
        Standard deviation for Gaussian filter.

    Returns
    -------
    pyramid : list
        List of images forming the Gaussian pyramid.
    """
    pyramid = [image]
    current_image = image.copy()

    for i in range(levels - 1):
        # Apply Gaussian filter
        blurred = gaussian(current_image, sigma=sigma)

        # Downsample by factor of 2
        downsampled = resize(blurred, (blurred.shape[0] // 2, blurred.shape[1] // 2))

        # Add to pyramid
        pyramid.append(downsampled)

        # Update current image for next iteration
        current_image = downsampled

    return pyramid

def laplacian_pyramid(image, levels=4, sigma=1):
    """
    Create a Laplacian pyramid by computing differences between
    successive levels of the Gaussian pyramid.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    levels : int
        Number of levels in the pyramid.
    sigma : float
        Standard deviation for Gaussian filter.

    Returns
    -------
    gaussian_pyr : list
        List of images forming the Gaussian pyramid.
    laplacian_pyr : list
        List of images forming the Laplacian pyramid.
    """
    # Generate Gaussian pyramid
    gaussian_pyr = gaussian_pyramid(image, levels, sigma)
    laplacian_pyr = []

    # Compute Laplacian pyramid (difference between Gaussian levels)
    for i in range(len(gaussian_pyr) - 1):
        # Get current and next level from Gaussian pyramid
        current_level = gaussian_pyr[i]
        next_level = gaussian_pyr[i + 1]

        # Upsample next level to match current level's size
        upsampled = resize(next_level, current_level.shape)

        # Compute difference (Laplacian)
        laplacian = current_level - upsampled

        # Add to Laplacian pyramid
        laplacian_pyr.append(laplacian)

    # Add the last level of Gaussian pyramid to Laplacian pyramid
    laplacian_pyr.append(gaussian_pyr[-1])

    return gaussian_pyr, laplacian_pyr

def reconstruct_from_laplacian_pyramid(laplacian_pyr, interp='bilinear'):
    """
    Reconstruct an image from its Laplacian pyramid.

    Parameters
    ----------
    laplacian_pyr : list
        List of images forming the Laplacian pyramid.
    interp : str
        Interpolation method for resizing.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image.
    """
    # Start with the smallest level (coarsest)
    reconstructed = laplacian_pyr[-1].copy()

    # Iterate from second-to-last to first level
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        # Upsample current reconstruction
        upsampled = resize(reconstructed, laplacian_pyr[i].shape, order=1)

        # Add Laplacian detail
        reconstructed = upsampled + laplacian_pyr[i]

    return reconstructed

def reconstruct_from_gaussian_pyramid(gaussian_pyr, interp='bilinear'):
    """
    Reconstruct an image from the last level of Gaussian pyramid without details.

    Parameters
    ----------
    gaussian_pyr : list
        List of images forming the Gaussian pyramid.
    interp : str
        Interpolation method for resizing.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image.
    """
    # Start with the smallest level (coarsest)
    reconstructed = gaussian_pyr[-1].copy()

    # Iterate from second-to-last to first level
    for i in range(len(gaussian_pyr) - 2, -1, -1):
        # Upsample current reconstruction to match the size of the next level
        reconstructed = resize(reconstructed, gaussian_pyr[i].shape, order=1)

    return reconstructed

def calculate_reconstruction_error(original, reconstructed):
    """
    Calculate the mean squared error between original and reconstructed images.

    Parameters
    ----------
    original : ndarray
        Original image.
    reconstructed : ndarray
        Reconstructed image.

    Returns
    -------
    error : float
        Mean squared error.
    """
    # Ensure images have the same shape
    if original.shape != reconstructed.shape:
        reconstructed = resize(reconstructed, original.shape)

    # Calculate mean squared error
    mse = np.mean((original - reconstructed) ** 2)

    return mse

def visualize_pyramid(pyramid, title="Pyramid Visualization"):
    """
    Visualize a pyramid (Gaussian or Laplacian).

    Parameters
    ----------
    pyramid : list
        List of images forming the pyramid.
    title : str
        Title for the figure.
    """
    n_levels = len(pyramid)
    fig, axes = plt.subplots(1, n_levels, figsize=(15, 5))

    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(pyramid):
        axes[i].imshow(level, cmap='gray')
        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def load_image(image_path=None):
    """
    Load an image for processing.

    Parameters
    ----------
    image_path : str, optional
        Path to the image file. If None, a default image is loaded.

    Returns
    -------
    image : ndarray
        Loaded image as float array.
    """
    if image_path is None:
        # Default to cerveau.jpg in the data folder
        image_path = 'data/cerveau.jpg'

    # Load and convert to grayscale float image
    image = img_as_float(io.imread(image_path, as_gray=True))

    return image

def main():
    """
    Main function to demonstrate pyramidal decomposition and reconstruction.
    """
    # Load image
    image = load_image()

    # Create output directory if it doesn't exist
    os.makedirs('output/multiscale', exist_ok=True)

    # Display original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('output/multiscale/original_image.png')
    plt.show()

    # Create Gaussian and Laplacian pyramids
    levels = 4
    gaussian_pyr, laplacian_pyr = laplacian_pyramid(image, levels=levels)

    # Visualize Gaussian pyramid
    plt.figure(figsize=(15, 5))
    for i, level in enumerate(gaussian_pyr):
        plt.subplot(1, len(gaussian_pyr), i+1)
        plt.imshow(level, cmap='gray')
        plt.title(f'Level {i}')
        plt.axis('off')
    plt.suptitle("Gaussian Pyramid")
    plt.tight_layout()
    plt.savefig('output/multiscale/gaussian_pyramid.png')
    visualize_pyramid(gaussian_pyr, "Gaussian Pyramid")

    # Visualize Laplacian pyramid
    plt.figure(figsize=(15, 5))
    for i, level in enumerate(laplacian_pyr):
        plt.subplot(1, len(laplacian_pyr), i+1)
        plt.imshow(level, cmap='gray')
        plt.title(f'Level {i}')
        plt.axis('off')
    plt.suptitle("Laplacian Pyramid")
    plt.tight_layout()
    plt.savefig('output/multiscale/laplacian_pyramid.png')
    visualize_pyramid(laplacian_pyr, "Laplacian Pyramid")

    # Reconstruct from Laplacian pyramid (with details)
    reconstructed_with_details = reconstruct_from_laplacian_pyramid(laplacian_pyr)

    # Reconstruct from Gaussian pyramid (without details)
    reconstructed_without_details = reconstruct_from_gaussian_pyramid(gaussian_pyr)

    # Calculate reconstruction errors
    error_with_details = calculate_reconstruction_error(image, reconstructed_with_details)
    error_without_details = calculate_reconstruction_error(image, reconstructed_without_details)

    print(f"Reconstruction error with details: {error_with_details:.8f}")
    print(f"Reconstruction error without details: {error_without_details:.8f}")

    # Display reconstructed images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(reconstructed_without_details, cmap='gray')
    axes[1].set_title(f'Reconstructed without Details\nMSE: {error_without_details:.8f}')
    axes[1].axis('off')

    axes[2].imshow(reconstructed_with_details, cmap='gray')
    axes[2].set_title(f'Reconstructed with Details\nMSE: {error_with_details:.8f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('output/multiscale/reconstruction_comparison.png')
    plt.show()

    # Save individual reconstructed images
    io.imsave('output/multiscale/reconstructed_with_details.png', img_as_ubyte(reconstructed_with_details))
    io.imsave('output/multiscale/reconstructed_without_details.png', img_as_ubyte(reconstructed_without_details))

if __name__ == "__main__":
    main()
