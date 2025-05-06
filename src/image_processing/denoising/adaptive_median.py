#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Median Filter implementation.

This module implements the adaptive median filter algorithm as described by
Gonzalez and Woods. The algorithm is designed to handle salt-and-pepper noise
while preserving image details.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
from skimage.util import view_as_windows
from skimage import img_as_float, img_as_ubyte

def is_impulse_noise(image, i, j, s, med_value):
    """
    Check if a pixel is an impulse noise.

    Args:
        image (ndarray): Input image
        i (int): Row index
        j (int): Column index
        s (int): Window size
        med_value (float): Median value in the window

    Returns:
        bool: True if the pixel is an impulse noise, False otherwise
    """
    # Get the neighborhood of size s centered at (i, j)
    half_s = s // 2
    i_start = max(0, i - half_s)
    i_end = min(image.shape[0], i + half_s + 1)
    j_start = max(0, j - half_s)
    j_end = min(image.shape[1], j + half_s + 1)

    neighborhood = image[i_start:i_end, j_start:j_end]

    # Calculate min and max values in the neighborhood
    min_val = np.min(neighborhood)
    max_val = np.max(neighborhood)

    # Check if the pixel value is equal to min or max
    pixel_val = image[i, j]

    # If the pixel value is equal to min or max, and different from the median,
    # it is likely an impulse noise
    return (pixel_val == min_val or pixel_val == max_val) and abs(pixel_val - med_value) > 0.01

def adaptive_median_filter(image, max_window_size=7):
    """
    Apply adaptive median filter to an image.

    The adaptive median filter algorithm works as follows:
    1. For each pixel, start with a small window size (3x3)
    2. Calculate the median, min, and max values in the window
    3. If the median is not between min and max, increase the window size
    4. If the pixel value is not an impulse noise, keep it unchanged
    5. Otherwise, replace it with the median value

    Args:
        image (ndarray): Input image (grayscale)
        max_window_size (int): Maximum window size (default: 7)

    Returns:
        ndarray: Filtered image
    """
    # Convert to float for processing
    image_float = img_as_float(image)

    # Create a copy of the image for the output
    filtered_image = np.copy(image_float)

    # Get image dimensions
    height, width = image_float.shape

    # Pad the image to handle border pixels
    pad_size = max_window_size // 2
    padded_image = np.pad(image_float, pad_size, mode='reflect')

    # Process each pixel
    for i in range(height):
        for j in range(width):
            # Get the pixel value
            pixel_val = image_float[i, j]

            # Start with the smallest window size (3x3)
            s = 3

            # Flag to indicate if the pixel has been processed
            processed = False

            while s <= max_window_size and not processed:
                # Get the neighborhood of size s centered at (i, j)
                half_s = s // 2
                i_pad = i + pad_size
                j_pad = j + pad_size

                # Extract the neighborhood from the padded image
                neighborhood = padded_image[i_pad-half_s:i_pad+half_s+1, j_pad-half_s:j_pad+half_s+1]

                # Calculate median, min, and max values
                med_val = np.median(neighborhood)
                min_val = np.min(neighborhood)
                max_val = np.max(neighborhood)

                # Check if the pixel is an impulse noise (salt or pepper)
                if pixel_val == min_val or pixel_val == max_val:
                    # Replace with median value
                    filtered_image[i, j] = med_val
                    processed = True
                else:
                    # Keep the original value
                    processed = True

                # Increase window size if not processed
                if not processed:
                    s += 2

    return filtered_image

def fast_adaptive_median_filter(image, max_window_size=7):
    """
    Apply adaptive median filter to an image using a faster implementation.

    This implementation uses a more efficient approach to detect and remove salt and pepper noise.

    Args:
        image (ndarray): Input image (grayscale)
        max_window_size (int): Maximum window size (default: 7)

    Returns:
        ndarray: Filtered image
    """
    # Convert to float for processing
    image_float = img_as_float(image)

    # Create a copy of the image for the output
    filtered_image = np.copy(image_float)

    # Get image dimensions
    height, width = image_float.shape

    # Pad the image to handle border pixels
    pad_size = max_window_size // 2
    padded_image = np.pad(image_float, pad_size, mode='reflect')

    # Create a mask for pixels that need to be processed
    # Initially, all pixels are candidates for processing
    to_process = np.ones((height, width), dtype=bool)

    # Process with increasing window sizes
    for s in range(3, max_window_size + 1, 2):
        # Skip if all pixels have been processed
        if not np.any(to_process):
            break

        half_s = s // 2

        # Process only the remaining pixels
        for i in range(height):
            for j in range(width):
                if not to_process[i, j]:
                    continue

                # Get the neighborhood from the padded image
                i_pad = i + pad_size
                j_pad = j + pad_size
                neighborhood = padded_image[i_pad-half_s:i_pad+half_s+1, j_pad-half_s:j_pad+half_s+1]

                # Calculate median, min, and max
                med_val = np.median(neighborhood)
                min_val = np.min(neighborhood)
                max_val = np.max(neighborhood)

                # Get the pixel value
                pixel_val = image_float[i, j]

                # Check if the pixel is an impulse noise (salt or pepper)
                if pixel_val == min_val or pixel_val == max_val:
                    # Replace with median value
                    filtered_image[i, j] = med_val
                    # Mark as processed
                    to_process[i, j] = False
                else:
                    # Keep the original value and mark as processed
                    to_process[i, j] = False

    return filtered_image
