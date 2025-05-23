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
from skimage.morphology import disk, dilation, erosion
import os

def morphological_multiscale(image, levels=4):
    """
    Morphological multiscale decomposition using dilation and erosion.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    levels : int
        Number of levels in the decomposition.

    Returns
    -------
    pyramid_dilations : list
        List of images forming the pyramid of dilations.
    pyramid_erosions : list
        List of images forming the pyramid of erosions.
    """
    pyramid_dilations = []
    pyramid_erosions = []

    for r in range(1, levels + 1):
        # Create disk structuring element with increasing radius
        se = disk(r)

        # Apply dilation
        dilated = dilation(image, footprint=se)
        pyramid_dilations.append(dilated)

        # Apply erosion
        eroded = erosion(image, footprint=se)
        pyramid_erosions.append(eroded)

    return pyramid_dilations, pyramid_erosions

def kramer_bruckner_filter(image, radius=5):
    """
    Elementary Kramer/Bruckner filter (toggle filter).

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    radius : int
        Radius of the structuring element (disk).

    Returns
    -------
    filtered : ndarray
        Filtered image.
    """
    # Create disk structuring element
    se = disk(radius)

    # Apply dilation and erosion
    dilated = dilation(image, footprint=se)
    eroded = erosion(image, footprint=se)

    # Calculate difference between image and operations
    diff_dilation = np.abs(dilated - image)
    diff_erosion = np.abs(image - eroded)

    # Apply the filter rule: choose dilation or erosion based on which is closer to original
    filtered = np.where(diff_dilation <= diff_erosion, dilated, eroded)

    return filtered

def kramer_bruckner_multiscale(image, levels=3, radius=5):
    """
    Kramer and Bruckner multiscale decomposition.

    Parameters
    ----------
    image : ndarray
        Input image (grayscale).
    levels : int
        Number of iterations of the filter.
    radius : int
        Radius of the structuring element (disk).

    Returns
    -------
    kb_filters : list
        List of images after applying KB filter iteratively.
    """
    kb_filters = [image]  # Start with original image

    current_image = image.copy()
    for i in range(levels):
        # Apply KB filter to current image
        filtered = kramer_bruckner_filter(current_image, radius)
        kb_filters.append(filtered)
        current_image = filtered

    return kb_filters

def visualize_morphological_pyramid(pyramid_dilations, pyramid_erosions, title="Morphological Multiscale Decomposition"):
    """
    Visualize the morphological multiscale decomposition.

    Parameters
    ----------
    pyramid_dilations : list
        List of dilated images.
    pyramid_erosions : list
        List of eroded images.
    title : str
        Title for the figure.
    """
    levels = len(pyramid_dilations)

    # Create a figure with two rows (dilations and erosions)
    fig, axes = plt.subplots(2, levels, figsize=(4*levels, 8))

    # Plot dilations in the first row
    for i, dilated in enumerate(pyramid_dilations):
        axes[0, i].imshow(dilated, cmap='gray')
        axes[0, i].set_title(f'Dilation scale {i+1}')
        axes[0, i].axis('off')

    # Plot erosions in the second row
    for i, eroded in enumerate(pyramid_erosions):
        axes[1, i].imshow(eroded, cmap='gray')
        axes[1, i].set_title(f'Erosion scale {i+1}')
        axes[1, i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig

def visualize_kb_filters(kb_filters, title="Kramer and Bruckner Multiscale Decomposition"):
    """
    Visualize the Kramer and Bruckner multiscale decomposition.

    Parameters
    ----------
    kb_filters : list
        List of KB filtered images.
    title : str
        Title for the figure.
    """
    levels = len(kb_filters)

    # Create a figure with one row
    fig, axes = plt.subplots(1, levels, figsize=(4*levels, 4))

    if levels == 1:
        axes = [axes]

    # Plot KB filters
    for i, filtered in enumerate(kb_filters):
        axes[i].imshow(filtered, cmap='gray')
        axes[i].set_title(f'$MK^{i}_B$')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig

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
    Main function to demonstrate scale-space decomposition and multiscale filtering.
    """
    # Create output directory
    os.makedirs('output/multiscale/scale_space', exist_ok=True)

    # Load image
    image = load_image()

    # Display original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('output/multiscale/scale_space/original_image.png')
    plt.close()

    # 1. Morphological Multiscale Decomposition
    print("Performing Morphological Multiscale Decomposition...")
    pyramid_dilations, pyramid_erosions = morphological_multiscale(image, levels=4)

    # Visualize morphological pyramids
    fig_morpho = visualize_morphological_pyramid(pyramid_dilations, pyramid_erosions)
    plt.savefig('output/multiscale/scale_space/morphological_decomposition.png')
    plt.close(fig_morpho)

    # 2. Kramer and Bruckner Multiscale Decomposition
    print("Performing Kramer and Bruckner Multiscale Decomposition...")

    # Test with different values of n (iterations)
    for n in [1, 3, 5]:
        kb_filters = kramer_bruckner_multiscale(image, levels=n, radius=5)

        # Visualize KB filters
        fig_kb = visualize_kb_filters(kb_filters, f"Kramer and Bruckner Decomposition (n={n}, r=5)")
        plt.savefig(f'output/multiscale/scale_space/kb_decomposition_n{n}.png')
        plt.close(fig_kb)

    print("Scale-space decomposition and multiscale filtering completed successfully!")
    print("Results saved to output/multiscale/scale_space/")

if __name__ == "__main__":
    main()
