#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiscale

Multiscale analysis techniques including pyramidal decomposition and scale-space decomposition.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import argparse
import os
import matplotlib.pyplot as plt
from skimage import io, img_as_float

from .pyramidal_decomposition import (
    gaussian_pyramid,
    laplacian_pyramid,
    reconstruct_from_laplacian_pyramid,
    reconstruct_from_gaussian_pyramid,
    calculate_reconstruction_error
)

from .scale_space import (
    morphological_multiscale,
    kramer_bruckner_multiscale,
    visualize_morphological_pyramid,
    visualize_kb_filters,
    load_image
)

def visualize_gaussian_pyramid(gaussian_pyr, title="Gaussian Pyramid"):
    """
    Visualize a Gaussian pyramid.

    Parameters
    ----------
    gaussian_pyr : list
        List of images forming the Gaussian pyramid.
    title : str
        Title for the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    n_levels = len(gaussian_pyr)
    fig, axes = plt.subplots(1, n_levels, figsize=(15, 5))

    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(gaussian_pyr):
        axes[i].imshow(level, cmap='gray')
        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig

def visualize_laplacian_pyramid(laplacian_pyr, title="Laplacian Pyramid"):
    """
    Visualize a Laplacian pyramid.

    Parameters
    ----------
    laplacian_pyr : list
        List of images forming the Laplacian pyramid.
    title : str
        Title for the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    n_levels = len(laplacian_pyr)
    fig, axes = plt.subplots(1, n_levels, figsize=(15, 5))

    if n_levels == 1:
        axes = [axes]

    for i, level in enumerate(laplacian_pyr):
        # For better visualization, we normalize the Laplacian levels
        if i < n_levels - 1:  # All levels except the last one
            # Normalize to [0, 1] for visualization
            level_normalized = (level - level.min()) / (level.max() - level.min() + 1e-8)
            axes[i].imshow(level_normalized, cmap='gray')
        else:  # Last level (Gaussian residual)
            axes[i].imshow(level, cmap='gray')

        axes[i].set_title(f'Level {i}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()

    return fig

def visualize_reconstruction_comparison(original, without_details, with_details):
    """
    Visualize comparison between original image and reconstructions.

    Parameters
    ----------
    original : ndarray
        Original image.
    without_details : ndarray
        Image reconstructed without details.
    with_details : ndarray
        Image reconstructed with details.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    # Calculate reconstruction errors
    error_with_details = calculate_reconstruction_error(original, with_details)
    error_without_details = calculate_reconstruction_error(original, without_details)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(without_details, cmap='gray')
    axes[1].set_title(f'Reconstructed without Details\nMSE: {error_without_details:.8f}')
    axes[1].axis('off')

    axes[2].imshow(with_details, cmap='gray')
    axes[2].set_title(f'Reconstructed with Details\nMSE: {error_with_details:.8f}')
    axes[2].axis('off')

    plt.tight_layout()

    return fig

def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Multiscale analysis for image processing'
    )

    parser.add_argument(
        '--image',
        type=str,
        default='data/cerveau.jpg',
        help='Path to the input image (default: data/cerveau.jpg)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/multiscale',
        help='Directory to save output images (default: output/multiscale)'
    )

    parser.add_argument(
        '--method',
        choices=['pyramidal', 'scale_space', 'both'],
        default='both',
        help='Analysis method to use (default: both)'
    )

    # Pyramidal decomposition parameters
    parser.add_argument(
        '--levels',
        type=int,
        default=4,
        help='Number of levels for pyramidal decomposition (default: 4)'
    )

    parser.add_argument(
        '--sigma',
        type=float,
        default=1.0,
        help='Sigma for Gaussian filtering in pyramidal decomposition (default: 1.0)'
    )

    # Scale-space decomposition parameters
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Radius of the structuring element (disk) for scale-space decomposition (default: 5)'
    )

    parser.add_argument(
        '--kb-iterations',
        type=int,
        default=3,
        help='Number of iterations for Kramer-Bruckner filter (default: 3)'
    )

    return parser.parse_args()

def main(args=None):
    """
    Main function for the command-line interface.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line arguments. If None, arguments are parsed from sys.argv.
    """
    # Parse command-line arguments if not provided
    if args is None:
        args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image
    try:
        image = img_as_float(io.imread(args.image, as_gray=True))
    except FileNotFoundError:
        print(f"Error: Image file '{args.image}' not found.")
        return

    # Display original image
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, 'original_image.png'))
    plt.close()

    # Apply selected methods
    if args.method in ['pyramidal', 'both']:
        # Create output subdirectory for pyramidal decomposition
        os.makedirs(os.path.join(args.output_dir, 'pyramidal'), exist_ok=True)

        # Pyramidal Decomposition
        print("Performing Pyramidal Decomposition...")

        # Create Gaussian and Laplacian pyramids
        gaussian_pyr, laplacian_pyr = laplacian_pyramid(image, levels=args.levels, sigma=args.sigma)

        # Visualize Gaussian pyramid
        fig_gaussian = visualize_gaussian_pyramid(gaussian_pyr)
        plt.savefig(os.path.join(args.output_dir, 'pyramidal', 'gaussian_pyramid.png'))
        plt.close(fig_gaussian)

        # Visualize Laplacian pyramid
        fig_laplacian = visualize_laplacian_pyramid(laplacian_pyr)
        plt.savefig(os.path.join(args.output_dir, 'pyramidal', 'laplacian_pyramid.png'))
        plt.close(fig_laplacian)

        # Reconstruct from Laplacian pyramid (with details)
        reconstructed_with_details = reconstruct_from_laplacian_pyramid(laplacian_pyr)

        # Reconstruct from Gaussian pyramid (without details)
        reconstructed_without_details = reconstruct_from_gaussian_pyramid(gaussian_pyr)

        # Calculate reconstruction errors
        error_with_details = calculate_reconstruction_error(image, reconstructed_with_details)
        error_without_details = calculate_reconstruction_error(image, reconstructed_without_details)

        print(f"Reconstruction error with details: {error_with_details:.8f}")
        print(f"Reconstruction error without details: {error_without_details:.8f}")

        # Visualize reconstruction comparison
        fig_comparison = visualize_reconstruction_comparison(
            image, reconstructed_without_details, reconstructed_with_details
        )
        plt.savefig(os.path.join(args.output_dir, 'pyramidal', 'reconstruction_comparison.png'))
        plt.close(fig_comparison)

    if args.method in ['scale_space', 'both']:
        # Create output subdirectory for scale-space decomposition
        os.makedirs(os.path.join(args.output_dir, 'scale_space'), exist_ok=True)

        # Scale-Space Decomposition
        print("Performing Scale-Space Decomposition...")

        # 1. Morphological Multiscale Decomposition
        print("  - Morphological Multiscale Decomposition...")
        pyramid_dilations, pyramid_erosions = morphological_multiscale(image, levels=args.levels)

        # Visualize morphological pyramids
        fig_morpho = visualize_morphological_pyramid(pyramid_dilations, pyramid_erosions)
        plt.savefig(os.path.join(args.output_dir, 'scale_space', 'morphological_decomposition.png'))
        plt.close(fig_morpho)

        # 2. Kramer and Bruckner Multiscale Decomposition
        print("  - Kramer and Bruckner Multiscale Decomposition...")
        kb_filters = kramer_bruckner_multiscale(image, levels=args.kb_iterations, radius=args.radius)

        # Visualize KB filters
        fig_kb = visualize_kb_filters(kb_filters, f"Kramer and Bruckner Decomposition (n={args.kb_iterations}, r={args.radius})")
        plt.savefig(os.path.join(args.output_dir, 'scale_space', f'kb_decomposition_n{args.kb_iterations}.png'))
        plt.close(fig_kb)

    print(f"Multiscale analysis completed successfully!")
    print(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
