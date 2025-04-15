#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functionality for the image processing tools.
This module contains the main logic that was previously in main.py.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt

def process_intensity_command(args):
    """Process the intensity command."""
    image_info = f" on {args.image}" if args.image else ""
    output_info = f" and saving to {args.output}" if args.output else ""

    if args.type == 'gamma':
        print(f"Applying gamma correction with gamma={args.gamma}{image_info}{output_info}")
    elif args.type == 'contrast':
        print(f"Applying contrast stretching with E={args.E}{image_info}{output_info}")
    else:  # both
        print(f"Applying both gamma correction (gamma={args.gamma}) and contrast stretching (E={args.E}){image_info}{output_info}")

    # Import the intensity transformations module
    from src.image_processing import intensity_transformations

    # Load the image
    image = intensity_transformations.load_image() if args.image is None else \
            intensity_transformations.img_as_float(intensity_transformations.io.imread(args.image, as_gray=True))

    # Apply the transformations based on the type
    if args.type == 'gamma' or args.type == 'both':
        # Test gamma correction
        intensity_transformations.test_gamma_correction(image, [args.gamma])

    if args.type == 'contrast' or args.type == 'both':
        # Test contrast stretching
        intensity_transformations.test_contrast_stretching(image, [args.E])

    # Save the output if requested
    if args.output and args.type != 'both':
        result = None
        if args.type == 'gamma':
            result = intensity_transformations.apply_gamma_correction(image, args.gamma)
        elif args.type == 'contrast':
            result = intensity_transformations.apply_contrast_stretching(image, args.E)

        if result is not None:
            # Convert to uint8 for saving
            result_uint8 = intensity_transformations.img_as_ubyte(result)
            intensity_transformations.io.imsave(args.output, result_uint8)
            print(f"Transformed image saved to: {args.output}")

def process_histogram_command(args):
    """Process the histogram command."""
    image_info = f" on {args.image}" if args.image else ""
    output_info = f" and saving to {args.output}" if args.output else ""
    bins_info = f" with {args.bins} bins"

    if args.method == 'builtin':
        print(f"Applying histogram equalization using built-in method{image_info}{bins_info}{output_info}")
    elif args.method == 'custom':
        print(f"Applying histogram equalization using custom method{image_info}{bins_info}{output_info}")
    else:  # both
        print(f"Applying histogram equalization using both methods{image_info}{bins_info}{output_info}")

    # Import the histogram equalization module
    from src.image_processing import histogram_equalization

    # Load the image
    image = histogram_equalization.load_image() if args.image is None else \
            histogram_equalization.img_as_float(histogram_equalization.io.imread(args.image, as_gray=True))

    # Compute and visualize the histogram of the original image
    histogram_equalization.visualize_histogram(image, bins=args.bins, title="Histogramme de l'image originale")

    # Apply histogram equalization based on the method
    if args.method == 'builtin':
        # Apply built-in histogram equalization
        equalized = histogram_equalization.equalize_histogram_builtin(image)
        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        axes[1].imshow(equalized, cmap='gray')
        axes[1].set_title('Égalisation (builtin)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        # Visualize the histogram of the equalized image
        histogram_equalization.visualize_histogram(equalized, bins=args.bins, title="Histogramme après égalisation (builtin)")
        # Save the output if requested
        if args.output:
            result_uint8 = histogram_equalization.img_as_ubyte(equalized)
            histogram_equalization.io.imsave(args.output, result_uint8)
            print(f"Equalized image saved to: {args.output}")

    elif args.method == 'custom':
        # Apply custom histogram equalization
        equalized = histogram_equalization.equalize_histogram_custom(image, bins=args.bins)
        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        axes[1].imshow(equalized, cmap='gray')
        axes[1].set_title('Égalisation (custom)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
        # Visualize the histogram of the equalized image
        histogram_equalization.visualize_histogram(equalized, bins=args.bins, title="Histogramme après égalisation (custom)")
        # Visualize the LUT
        histogram_equalization.visualize_equalization_lut(image, bins=args.bins)
        # Save the output if requested
        if args.output:
            result_uint8 = histogram_equalization.img_as_ubyte(equalized)
            histogram_equalization.io.imsave(args.output, result_uint8)
            print(f"Equalized image saved to: {args.output}")

    else:  # both
        # Test both methods
        equalized_builtin, equalized_custom = histogram_equalization.test_histogram_equalization(image)

def process_matching_command(args):
    """Process the matching command."""
    image_info = f" on {args.image}" if args.image else ""
    output_info = f" and saving to {args.output}" if args.output else ""
    bins_info = f" with {args.bins} bins"
    peaks_info = f" (peaks at {args.peak1:.2f} and {args.peak2:.2f})"

    if args.method == 'builtin':
        print(f"Applying histogram matching using built-in method{image_info}{bins_info}{peaks_info}{output_info}")
    elif args.method == 'custom':
        print(f"Applying histogram matching using custom method{image_info}{bins_info}{peaks_info}{output_info}")
    else:  # both
        print(f"Applying histogram matching using both methods{image_info}{bins_info}{peaks_info}{output_info}")

    # Import the histogram matching module
    from src.image_processing import histogram_matching

    # Load the image
    image = histogram_matching.load_image() if args.image is None else \
            histogram_matching.img_as_float(histogram_matching.io.imread(args.image, as_gray=True))

    # Visualize the histogram of the image
    print("Visualisation de l'histogramme de l'image...")
    histogram_matching.visualize_histogram(image, bins=args.bins, title=f"Histogramme de l'image originale")

    # Create a bimodal histogram as reference
    print("Création d'un histogramme bimodal de référence...")
    reference_hist, _ = histogram_matching.create_bimodal_histogram(
        bins=args.bins,
        peak1=args.peak1,
        peak2=args.peak2,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        weight1=args.weight1,
        weight2=args.weight2
    )

    # Visualize the reference histogram
    histogram_matching.visualize_bimodal_histogram(
        bins=args.bins,
        peak1=args.peak1,
        peak2=args.peak2,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        weight1=args.weight1,
        weight2=args.weight2,
        title="Histogramme bimodal de référence"
    )

    # Apply histogram matching based on the method
    if args.method == 'builtin':
        # Apply built-in histogram matching
        print("Application de l'appariement d'histogramme avec les fonctions intégrées...")
        matched = histogram_matching.match_histogram_builtin(image, reference_hist, bins=args.bins)

        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        axes[1].imshow(matched, cmap='gray')
        axes[1].set_title('Appariement (builtin)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

        # Visualize the histogram of the matched image
        histogram_matching.visualize_histogram(matched, bins=args.bins, title="Histogramme après appariement (builtin)")

        # Save the output if requested
        if args.output:
            result_uint8 = histogram_matching.img_as_ubyte(matched)
            histogram_matching.io.imsave(args.output, result_uint8)
            print(f"Matched image saved to: {args.output}")

    elif args.method == 'custom':
        # Apply custom histogram matching
        print("Application de l'appariement d'histogramme avec notre implémentation personnalisée...")
        matched = histogram_matching.match_histogram_custom(image, reference_hist, bins=args.bins)

        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        axes[1].imshow(matched, cmap='gray')
        axes[1].set_title('Appariement (custom)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

        # Visualize the histogram of the matched image
        histogram_matching.visualize_histogram(matched, bins=args.bins, title="Histogramme après appariement (custom)")

        # Save the output if requested
        if args.output:
            result_uint8 = histogram_matching.img_as_ubyte(matched)
            histogram_matching.io.imsave(args.output, result_uint8)
            print(f"Matched image saved to: {args.output}")

    else:  # both
        # Apply both methods and compare
        print("Application de l'appariement d'histogramme avec les deux méthodes...")
        equalized, matched_custom, matched_builtin = histogram_matching.visualize_matching_results(image, reference_hist, bins=args.bins)

def process_command(args):
    """Process the command based on the arguments."""
    if args.command == 'intensity':
        process_intensity_command(args)
    elif args.command == 'histogram':
        process_histogram_command(args)
    elif args.command == 'matching':
        process_matching_command(args)
    else:
        print(f"Unknown command: {args.command}")
