#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core

Module for image processing operations.

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

def process_damage_command(args):
    """Process the damage command."""
    # Import the damage modeling module
    from src.image_processing import damage_modeling

    # Use positional argument for image if provided
    if hasattr(args, 'image_path') and args.image_path and not args.image:
        args.image = args.image_path

    # Load the image
    image = damage_modeling.load_image() if args.image is None else \
            damage_modeling.img_as_float(damage_modeling.io.imread(args.image, as_gray=True))

    # Generate the PSF
    if args.psf == 'gaussian':
        psf = damage_modeling.generate_gaussian_psf(size=64, sigma=args.sigma)
        psf_title = f"Gaussian PSF (sigma={args.sigma})"
    else:  # motion
        psf = damage_modeling.generate_motion_blur_psf(size=64, length=args.length, angle=args.angle)
        psf_title = f"Motion Blur PSF (length={args.length}, angle={args.angle})"

    # Visualize the PSF
    damage_modeling.visualize_psf(psf, title=psf_title)

    # Apply damage to the image
    damaged = damage_modeling.apply_damage(image, psf, noise_level=args.noise)

    # Visualize the original and damaged images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(damaged, cmap='gray')
    axes[1].set_title('Damaged Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Save the output if requested
    if args.output:
        result_uint8 = damage_modeling.img_as_ubyte(damaged)
        damage_modeling.io.imsave(args.output, result_uint8)
        print(f"Damaged image saved to: {args.output}")

def process_restore_command(args):
    """Process the restore command."""
    # Import the damage modeling module
    from src.image_processing import damage_modeling

    # Use positional argument for image if provided
    if hasattr(args, 'image_path') and args.image_path and not args.image:
        args.image = args.image_path

    # Load the image
    damaged_image = damage_modeling.load_image() if args.image is None else \
                   damage_modeling.img_as_float(damage_modeling.io.imread(args.image, as_gray=True))

    # Generate the PSF
    if args.psf == 'gaussian':
        psf = damage_modeling.generate_gaussian_psf(size=64, sigma=args.sigma)
        psf_title = f"Gaussian PSF (sigma={args.sigma})"
    else:  # motion
        psf = damage_modeling.generate_motion_blur_psf(size=64, length=args.length, angle=args.angle)
        psf_title = f"Motion Blur PSF (length={args.length}, angle={args.angle})"

    # Visualize the PSF
    damage_modeling.visualize_psf(psf, title=psf_title)

    if args.method == 'compare':
        # Compare different restoration methods
        print("Comparing different restoration methods...")

        # Apply different restoration methods
        restored_inverse = damage_modeling.inverse_filter(damaged_image, psf, epsilon=1e-3)
        restored_wiener_low = damage_modeling.wiener_filter(damaged_image, psf, K=0.001)
        restored_wiener_med = damage_modeling.wiener_filter(damaged_image, psf, K=0.01)
        restored_wiener_high = damage_modeling.wiener_filter(damaged_image, psf, K=0.1)

        # Visualize the results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(damaged_image, cmap='gray')
        axes[0, 0].set_title('Damaged Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(restored_inverse, cmap='gray')
        axes[0, 1].set_title('Inverse Filter')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(restored_wiener_low, cmap='gray')
        axes[0, 2].set_title('Wiener Filter (K=0.001)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(restored_wiener_med, cmap='gray')
        axes[1, 0].set_title('Wiener Filter (K=0.01)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(restored_wiener_high, cmap='gray')
        axes[1, 1].set_title('Wiener Filter (K=0.1)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            # Save the medium Wiener filter result
            result_uint8 = damage_modeling.img_as_ubyte(restored_wiener_med)
            damage_modeling.io.imsave(args.output, result_uint8)
            print(f"Restored image (Wiener K=0.01) saved to: {args.output}")

            # Save other results with suffixes
            base_name, ext = os.path.splitext(args.output)
            damage_modeling.io.imsave(f"{base_name}_inverse{ext}", damage_modeling.img_as_ubyte(restored_inverse))
            damage_modeling.io.imsave(f"{base_name}_wiener_low{ext}", damage_modeling.img_as_ubyte(restored_wiener_low))
            damage_modeling.io.imsave(f"{base_name}_wiener_high{ext}", damage_modeling.img_as_ubyte(restored_wiener_high))
            print(f"All restoration results saved with different suffixes.")

    elif args.method == 'inverse':
        # Apply inverse filter
        print("Applying inverse filter...")
        restored = damage_modeling.inverse_filter(damaged_image, psf, epsilon=1e-3)

        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(damaged_image, cmap='gray')
        axes[0].set_title('Damaged Image')
        axes[0].axis('off')
        axes[1].imshow(restored, cmap='gray')
        axes[1].set_title('Restored Image (Inverse Filter)')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = damage_modeling.img_as_ubyte(restored)
            damage_modeling.io.imsave(args.output, result_uint8)
            print(f"Restored image saved to: {args.output}")

    else:  # wiener
        # Apply Wiener filter
        print(f"Applying Wiener filter with K={args.k}...")
        restored = damage_modeling.wiener_filter(damaged_image, psf, K=args.k)

        # Visualize the result
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(damaged_image, cmap='gray')
        axes[0].set_title('Damaged Image')
        axes[0].axis('off')
        axes[1].imshow(restored, cmap='gray')
        axes[1].set_title(f'Restored Image (Wiener Filter, K={args.k})')
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = damage_modeling.img_as_ubyte(restored)
            damage_modeling.io.imsave(args.output, result_uint8)
            print(f"Restored image saved to: {args.output}")

def process_checkerboard_command(args):
    """Process the checkerboard command."""
    # Import the damage modeling module
    from src.image_processing import damage_modeling

    # Generate the checkerboard
    checkerboard = damage_modeling.generate_checkerboard(size=args.size, square_size=args.square_size)

    # Visualize the checkerboard
    plt.figure(figsize=(5, 5))
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Checkerboard Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save the output if requested
    if args.output:
        result_uint8 = damage_modeling.img_as_ubyte(checkerboard)
        damage_modeling.io.imsave(args.output, result_uint8)
        print(f"Checkerboard image saved to: {args.output}")

def process_fourier_command(args):
    """Process the fourier command."""
    print(f"Running Fourier transform analysis on {args.image}")
    # Import the Fourier transform module
    from src.image_processing.fourier import transformer_fourier
    # The module will run automatically when imported

def process_filter_command(args):
    """Process the filter command."""
    print(f"Applying {args.type} filter to {args.image} with cutoff {args.cutoff}")
    # Import the filtering module
    from src.image_processing.filtering import filtering_hp_lp
    # The module will run automatically when imported

def process_denoising_command(args):
    """Process the denoising command."""
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float, img_as_ubyte

    noise_info = f" with {args.noise} noise" if args.noise else ""
    method_info = f" using {args.method} method" if args.method != 'all' else " using all methods"
    image_info = f" on {args.image}" if args.image else ""
    output_info = f" and saving to {args.output}" if args.output else ""

    print(f"Applying denoising{noise_info}{method_info}{image_info}{output_info}")

    # Import the denoising module
    from src.image_processing.denoising import (
        generate_uniform_noise,
        generate_gaussian_noise,
        generate_salt_pepper_noise,
        generate_exponential_noise,
        add_noise_to_image,
        extract_roi,
        estimate_noise_parameters,
        visualize_roi_histogram,
        apply_mean_filter,
        apply_median_filter,
        apply_gaussian_filter,
        apply_bilateral_filter,
        apply_nlm_filter,
        adaptive_median_filter,
        fast_adaptive_median_filter,
        compare_denoising_methods
    )

    # Load the image
    image = img_as_float(io.imread(args.image, as_gray=True))

    # Set noise parameters based on the noise type
    noise_params = {}
    if args.noise == 'gaussian':
        noise_params = {'mean': 0, 'std': args.noise_param}
    elif args.noise == 'uniform':
        noise_params = {'a': -args.noise_param, 'b': args.noise_param}
    elif args.noise == 'salt_pepper':
        # For salt and pepper, we use a different approach to control noise level
        a = args.noise_param / 2
        b = 1 - args.noise_param / 2
        noise_params = {'a': a, 'b': b}
    elif args.noise == 'exponential':
        noise_params = {'a': 1 / args.noise_param if args.noise_param > 0 else 10}

    # Add noise to the image
    noisy_image = add_noise_to_image(image, args.noise, **noise_params)

    # Apply denoising based on the method
    if args.method == 'mean':
        # Apply mean filter
        denoised = apply_mean_filter(noisy_image, kernel_size=args.kernel_size)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Mean Filter, k={args.kernel_size})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'median':
        # Apply median filter
        denoised = apply_median_filter(noisy_image, kernel_size=args.kernel_size)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Median Filter, k={args.kernel_size})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'gaussian':
        # Apply Gaussian filter
        denoised = apply_gaussian_filter(noisy_image, sigma=args.sigma)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Gaussian Filter, sigma={args.sigma})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'bilateral':
        # Apply bilateral filter
        denoised = apply_bilateral_filter(noisy_image, sigma_spatial=args.kernel_size, sigma_color=args.sigma)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Bilateral Filter, s={args.kernel_size}, c={args.sigma})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'nlm':
        # Apply non-local means filter
        denoised = apply_nlm_filter(noisy_image, patch_size=args.kernel_size, h=args.sigma)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (NLM Filter, p={args.kernel_size}, h={args.sigma})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'adaptive_median':
        # Apply adaptive median filter
        denoised = adaptive_median_filter(noisy_image, max_window_size=args.max_window_size)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Adaptive Median Filter, max_size={args.max_window_size})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    elif args.method == 'fast_adaptive_median':
        # Apply fast adaptive median filter
        denoised = fast_adaptive_median_filter(noisy_image, max_window_size=args.max_window_size)

        # Visualize the result
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(noisy_image, cmap='gray')
        axes[1].set_title(f'Noisy Image ({args.noise})')
        axes[1].axis('off')

        axes[2].imshow(denoised, cmap='gray')
        axes[2].set_title(f'Denoised Image (Fast Adaptive Median Filter, max_size={args.max_window_size})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Save the output if requested
        if args.output:
            result_uint8 = img_as_ubyte(denoised)
            io.imsave(args.output, result_uint8)
            print(f"Denoised image saved to: {args.output}")

    else:  # all
        # Compare all denoising methods
        denoised_images = compare_denoising_methods(image, noisy_image, save_path=args.output)

        # Extract ROI and estimate noise parameters
        print("\nExtracting ROI and estimating noise parameters...")
        roi_image, roi_coords = extract_roi(noisy_image, interactive=True)
        visualize_roi_histogram(roi_image, title=f"Histogram of ROI ({args.noise} noise)")
        noise_params = estimate_noise_parameters(roi_image, noise_type=args.noise)

def process_segmentation_command(args):
    """Process the segmentation command."""
    print(f"Running image segmentation using {args.method} method")
    # Import the segmentation module
    from src.image_processing import segmentation
    # The module will run automatically when imported

def process_kmeans_sim_command(args):
    """Process the kmeans-sim command."""
    print("Running K-means clustering simulation in 2D")
    # Import the kmeans simulation module
    from src.image_processing import kmeans_simulation
    # The module will run automatically when imported

def process_color_kmeans_command(args):
    """Process the color-kmeans command."""
    image_info = f" on {args.image}" if args.image else ""
    output_info = f" and saving to {args.output}" if args.output else ""
    print(f"Running color image segmentation using K-means with {args.clusters} clusters{image_info}{output_info}")
    # Import the color kmeans module
    from src.image_processing import color_kmeans
    # The module will run automatically when imported

def process_registration_command(args):
    """Process the registration command."""
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from skimage import io, color

    source_info = f" source: {args.source}" if args.source else ""
    target_info = f" target: {args.target}" if args.target else ""
    output_info = f" output: {args.output}" if args.output else ""
    superimpose_info = " with superimposition" if args.superimpose else ""

    print(f"Running image registration using {args.method} method{source_info}{target_info}{output_info}{superimpose_info}")

    # Import the registration module
    from src.image_processing.registration import (
        estimate_rigid_transform,
        apply_rigid_transform,
        icp_registration,
        detect_corners,
        visualize_point_pairs,
        visualize_registration_result,
        superimpose,
        select_corresponding_points,
        rigid_registration
    )

    # Load images
    source_path = "data/Brain1.bmp" if args.source is None else args.source
    target_path = "data/Brain2.bmp" if args.target is None else args.target

    try:
        source_image = io.imread(source_path)
        target_image = io.imread(target_path)

        # Convert to grayscale if needed
        if len(source_image.shape) == 3:
            source_gray = color.rgb2gray(source_image)
        else:
            source_gray = source_image

        if len(target_image.shape) == 3:
            target_gray = color.rgb2gray(target_image)
        else:
            target_gray = target_image

        # Display original images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(source_gray, cmap='gray')
        plt.title("Source Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(target_gray, cmap='gray')
        plt.title("Target Image")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('output/images/registration_original_images.png')
        plt.show()

        # Apply registration based on the method
        if args.method == 'manual':
            # Manual point selection
            print("\nSelecting corresponding points...")
            print("Please select points on both images and press 'q' when finished.")

            source_points, target_points = select_corresponding_points(
                source_gray, target_gray,
                "Select Points on Source Image",
                "Select Points on Target Image"
            )

            if len(source_points) < 2 or len(target_points) < 2:
                print("Error: At least 2 corresponding points are required for registration")
                return

            print(f"Selected {len(source_points)} corresponding points")

            # Estimate rigid transformation
            print("\nEstimating rigid transformation...")
            T = rigid_registration(source_points, target_points)
            print("Transformation matrix:")
            print(T)

            # Apply transformation to source image
            print("\nApplying transformation to source image...")
            rows, cols = target_gray.shape
            registered_image = cv2.warpAffine(source_gray, T, (cols, rows))

        elif args.method == 'automatic':
            # Automatic corner detection
            print("\nDetecting corners automatically...")
            source_corners = detect_corners(source_gray, max_corners=10)
            target_corners = detect_corners(target_gray, max_corners=10)

            # Visualize detected corners
            visualize_point_pairs(source_gray, target_gray, source_corners, target_corners,
                                title="Automatically Detected Corners")
            plt.savefig('output/images/registration_auto_corners.png')
            plt.show()

            # Apply ICP with detected corners
            print("\nApplying ICP algorithm with detected corners...")
            R, t, transformed_corners, error = icp_registration(
                source_corners, target_corners, max_iterations=50
            )
            print(f"ICP completed with error: {error:.6f}")

            # Convert to transformation matrix
            T = np.zeros((2, 3))
            T[0:2, 0:2] = R
            T[0:2, 2] = t

            # Apply transformation to source image
            print("\nApplying transformation to source image...")
            rows, cols = target_gray.shape
            registered_image = cv2.warpAffine(source_gray, T, (cols, rows))

        else:  # icp
            # Define control points (these would normally be selected by the user)
            print("\nUsing predefined control points...")
            source_points = np.array([[100, 100], [150, 100], [100, 150], [150, 150]])
            target_points = np.array([[110, 110], [160, 105], [105, 160], [155, 155]])  # Slightly shifted

            # Visualize point pairs
            visualize_point_pairs(source_gray, target_gray, source_points, target_points)
            plt.savefig('output/images/registration_icp_points.png')
            plt.show()

            # Apply ICP
            print("\nApplying ICP algorithm...")
            R, t, transformed_points, error = icp_registration(
                source_points, target_points, max_iterations=20
            )
            print(f"ICP completed with error: {error:.6f}")

            # Convert to transformation matrix
            T = np.zeros((2, 3))
            T[0:2, 0:2] = R
            T[0:2, 2] = t

            # Apply transformation to source image
            print("\nApplying transformation to source image...")
            rows, cols = target_gray.shape
            registered_image = cv2.warpAffine(source_gray, T, (cols, rows))

        # Visualize registration result
        visualize_registration_result(source_gray, target_gray, registered_image,
                                    title=f"Image Registration with {args.method.capitalize()} Method")
        plt.savefig(f'output/images/registration_{args.method}_result.png')
        plt.show()

        # Create and save superimposed image if requested
        if args.superimpose:
            print("\nCreating superimposed image...")
            superimpose_path = args.superimpose_output if args.superimpose_output else f'output/images/registration_{args.method}_superimposed.png'
            superimposed = superimpose(registered_image, target_gray, superimpose_path, show=True)

        # Save the registered image if requested
        if args.output:
            io.imsave(args.output, (registered_image * 255).astype(np.uint8))
            print(f"Registered image saved to: {args.output}")

        print("\nRegistration completed successfully!")

    except Exception as e:
        print(f"Error during registration: {str(e)}")

def process_machine_learning_command(args):
    """Process the machine learning command."""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    # Import machine learning modules
    from src.image_processing.machine_learning import (
        extract_features,
        load_kimia_dataset,
        extract_dataset_features,
        train_test_split_dataset,
        train_classifier,
        evaluate_classifier,
        classify_image,
        cross_validate,
        plot_confusion_matrix,
        plot_feature_importance,
        visualize_classification_results,
        visualize_dataset
    )
    from src.image_processing.machine_learning.utils import (
        create_output_directory,
        save_model,
        load_model,
        save_results,
        plot_learning_curve,
        plot_validation_curve
    )

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process based on the task
    if args.task == 'extract':
        print(f"Extracting features from dataset: {args.dataset}")

        # Load dataset
        images, labels, class_names = load_kimia_dataset(args.dataset)
        print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")

        # Determine feature types
        if args.features == 'all':
            feature_types = ['hu', 'zernike', 'geometric']
        else:
            feature_types = [args.features]

        # Extract features
        features = extract_dataset_features(images, feature_types=feature_types)
        print(f"Extracted {features.shape[1]} features per image")

        # Visualize dataset
        print("Visualizing dataset...")
        fig, ax = visualize_dataset(features, labels, class_names=class_names, method='pca')
        plt.savefig(os.path.join(args.output, 'dataset_pca.png'))
        plt.show()

        fig, ax = visualize_dataset(features, labels, class_names=class_names, method='tsne')
        plt.savefig(os.path.join(args.output, 'dataset_tsne.png'))
        plt.show()

    elif args.task == 'train':
        print(f"Training classifier on dataset: {args.dataset}")

        # Load dataset
        images, labels, class_names = load_kimia_dataset(args.dataset)
        print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")

        # Determine feature types
        if args.features == 'all':
            feature_types = ['hu', 'zernike', 'geometric']
        else:
            feature_types = [args.features]

        # Extract features
        print(f"Extracting features: {feature_types}")
        features = extract_dataset_features(images, feature_types=feature_types)
        print(f"Extracted {features.shape[1]} features per image")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split_dataset(
            features, labels, test_size=args.test_size)
        print(f"Split dataset into {len(X_train)} training and {len(X_test)} testing samples")

        # Train classifier
        print(f"Training {args.classifier} classifier...")
        classifier, scaler = train_classifier(X_train, y_train, classifier_type=args.classifier)

        # Evaluate classifier
        print("Evaluating classifier...")
        accuracy, y_pred, report, conf_matrix = evaluate_classifier(classifier, X_test, y_test, scaler)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")

        # Plot confusion matrix
        print("Plotting confusion matrix...")
        fig, ax = plot_confusion_matrix(y_test, y_pred, class_names=class_names)
        plt.savefig(os.path.join(args.output, 'confusion_matrix.png'))
        plt.show()

        # Plot feature importance if applicable
        if args.classifier == 'rf':
            print("Plotting feature importance...")
            fig, ax = plot_feature_importance(classifier)
            plt.savefig(os.path.join(args.output, 'feature_importance.png'))
            plt.show()

        # Visualize misclassified images
        print("Visualizing misclassified images...")
        # Get indices of misclassified images
        misclassified = np.where(y_test != y_pred)[0]
        if len(misclassified) > 0:
            print(f"Found {len(misclassified)} misclassified images")
            # Create a figure to show misclassifications
            n_cols = min(5, len(misclassified))
            n_rows = (len(misclassified) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            # For each misclassified image
            for i, idx in enumerate(misclassified):
                if i < len(axes):
                    # Get the true and predicted labels
                    true_label = y_test[idx]
                    pred_label = y_pred[idx]

                    # Set the title
                    axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}')
                    axes[i].axis('off')

            # Hide unused axes
            for i in range(len(misclassified), len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(args.output, 'misclassified_images.png'))
            plt.show()
        else:
            print("No misclassified images found.")

        # Save model
        print("Saving model...")
        save_model(classifier, scaler, args.output, 'model.pkl')

        # Save results
        results = {
            'Accuracy': accuracy,
            'Classification Report': report,
            'Confusion Matrix': conf_matrix
        }
        save_results(results, args.output, 'results.txt')

        # Perform cross-validation if requested
        if args.cross_validate:
            print("Performing cross-validation...")
            cv_scores, mean_score, std_score = cross_validate(
                features, labels, classifier_type=args.classifier)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean score: {mean_score:.4f} (±{std_score:.4f})")

            # Plot learning curve
            print("Plotting learning curve...")
            if args.classifier == 'mlp':
                clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            elif args.classifier == 'svm':
                clf = SVC(gamma='scale', probability=True, random_state=42)
            else:  # rf
                clf = RandomForestClassifier(n_estimators=100, random_state=42)

            fig = plot_learning_curve(clf, features, labels, cv=5,
                                    output_dir=args.output, filename='learning_curve.png')
            plt.show()

    elif args.task == 'classify':
        if args.image is None or args.model is None:
            print("Error: --image and --model are required for classify task")
            return

        print(f"Classifying image: {args.image}")

        # Load model
        print(f"Loading model from: {args.model}")
        classifier, scaler = load_model(args.model)

        # Load image
        print(f"Loading image: {args.image}")
        image = io.imread(args.image, as_gray=True)

        # Get the feature types used for training
        # We need to use the same feature types that were used for training
        # Use all feature types to match the training
        feature_types = ['hu', 'zernike', 'geometric']

        # Classify image
        print(f"Classifying image using {feature_types} features...")
        label, probability = classify_image(image, classifier, scaler, feature_types)

        # Load dataset to get class names
        _, _, class_names = load_kimia_dataset(args.dataset)

        # Display result
        print(f"Predicted class: {class_names[label]}")
        if probability is not None:
            print(f"Probability: {probability:.4f}")

        # Display image with prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {class_names[label]}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'classified_image.png'))
        plt.show()

    elif args.task == 'evaluate':
        if args.model is None:
            print("Error: --model is required for evaluate task")
            return

        print(f"Evaluating model: {args.model}")

        # Load model
        print(f"Loading model from: {args.model}")
        classifier, scaler = load_model(args.model)

        # Load dataset
        print(f"Loading dataset: {args.dataset}")
        images, labels, class_names = load_kimia_dataset(args.dataset)

        # Determine feature types
        if args.features == 'all':
            feature_types = ['hu', 'zernike', 'geometric']
        else:
            feature_types = [args.features]

        # Extract features
        print(f"Extracting features: {feature_types}")
        features = extract_dataset_features(images, feature_types=feature_types)

        # Scale features if scaler is available
        if scaler is not None:
            features = scaler.transform(features)

        # Make predictions
        print("Making predictions...")
        y_pred = classifier.predict(features)

        # Evaluate
        accuracy = accuracy_score(labels, y_pred)
        report = classification_report(labels, y_pred)
        conf_matrix = confusion_matrix(labels, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}")

        # Plot confusion matrix
        print("Plotting confusion matrix...")
        fig, ax = plot_confusion_matrix(labels, y_pred, class_names=class_names)
        plt.savefig(os.path.join(args.output, 'confusion_matrix.png'))
        plt.show()

        # Visualize misclassified images
        print("Visualizing misclassified images...")
        fig = visualize_classification_results(images, labels, y_pred, class_names)
        if fig is not None:
            plt.savefig(os.path.join(args.output, 'misclassified_images.png'))
            plt.show()

        # Save results
        results = {
            'Accuracy': accuracy,
            'Classification Report': report,
            'Confusion Matrix': conf_matrix
        }
        save_results(results, args.output, 'results.txt')

    elif args.task == 'visualize':
        print(f"Visualizing dataset: {args.dataset}")

        # Load dataset
        images, labels, class_names = load_kimia_dataset(args.dataset)
        print(f"Loaded {len(images)} images from {len(class_names)} classes: {class_names}")

        # Determine feature types
        if args.features == 'all':
            feature_types = ['hu', 'zernike', 'geometric']
        else:
            feature_types = [args.features]

        # Extract features
        print(f"Extracting features: {feature_types}")
        features = extract_dataset_features(images, feature_types=feature_types)

        # Visualize dataset
        print("Visualizing dataset with PCA...")
        fig, ax = visualize_dataset(features, labels, class_names=class_names, method='pca')
        plt.savefig(os.path.join(args.output, 'dataset_pca.png'))
        plt.show()

        print("Visualizing dataset with t-SNE...")
        fig, ax = visualize_dataset(features, labels, class_names=class_names, method='tsne')
        plt.savefig(os.path.join(args.output, 'dataset_tsne.png'))
        plt.show()

        # Display sample images from each class
        print("Displaying sample images from each class...")
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=(15, 5))

        for i, class_name in enumerate(class_names):
            # Find images of this class
            class_indices = [j for j, label in enumerate(labels) if label == i]
            if class_indices:
                # Display the first image of this class
                axes[i].imshow(images[class_indices[0]], cmap='gray')
                axes[i].set_title(class_name)
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'sample_images.png'))
        plt.show()

def process_multiscale_command(args):
    """Process the multiscale command."""
    # Import the multiscale module
    from src.image_processing.multiscale import (
        gaussian_pyramid,
        laplacian_pyramid,
        reconstruct_from_laplacian_pyramid,
        reconstruct_from_gaussian_pyramid,
        calculate_reconstruction_error
    )
    from src.image_processing.multiscale.pyramidal_decomposition import (
        visualize_pyramid,
        load_image
    )
    from skimage import io, img_as_ubyte

    # Use positional argument for image if provided
    if hasattr(args, 'image_path') and args.image_path and not args.image:
        args.image = args.image_path

    # Load the image
    image = load_image(args.image)

    # Display original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # Create Gaussian and Laplacian pyramids
    print(f"Creating pyramids with {args.levels} levels and sigma={args.sigma}...")
    gaussian_pyr, laplacian_pyr = laplacian_pyramid(image, levels=args.levels, sigma=args.sigma)

    # Visualize Gaussian pyramid
    print("Visualizing Gaussian pyramid...")
    visualize_pyramid(gaussian_pyr, "Gaussian Pyramid")

    # Visualize Laplacian pyramid
    print("Visualizing Laplacian pyramid...")
    visualize_pyramid(laplacian_pyr, "Laplacian Pyramid")

    # Reconstruct from Laplacian pyramid (with details)
    print("Reconstructing image from Laplacian pyramid (with details)...")
    reconstructed_with_details = reconstruct_from_laplacian_pyramid(laplacian_pyr)

    # Calculate reconstruction error with details
    error_with_details = calculate_reconstruction_error(image, reconstructed_with_details)
    print(f"Reconstruction error with details: {error_with_details:.8f}")

    if args.compare:
        # Reconstruct from Gaussian pyramid (without details)
        print("Reconstructing image from Gaussian pyramid (without details)...")
        reconstructed_without_details = reconstruct_from_gaussian_pyramid(gaussian_pyr)

        # Calculate reconstruction error without details
        error_without_details = calculate_reconstruction_error(image, reconstructed_without_details)
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
        plt.show()

        # Save output if requested
        if args.output:
            from skimage import img_as_ubyte
            import os

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save reconstructed images
            base_name, ext = os.path.splitext(args.output)
            io.imsave(f"{base_name}_with_details{ext}", img_as_ubyte(reconstructed_with_details))
            io.imsave(f"{base_name}_without_details{ext}", img_as_ubyte(reconstructed_without_details))
            print(f"Reconstructed images saved to: {base_name}_with_details{ext} and {base_name}_without_details{ext}")
    else:
        # Display reconstructed image with details only
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(reconstructed_with_details, cmap='gray')
        axes[1].set_title(f'Reconstructed with Details\nMSE: {error_with_details:.8f}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Save output if requested
        if args.output:
            from skimage import img_as_ubyte
            import os

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save reconstructed image
            io.imsave(args.output, img_as_ubyte(reconstructed_with_details))
            print(f"Reconstructed image saved to: {args.output}")

def process_command(args):
    """Process the command based on the arguments."""
    if args.command == 'intensity':
        process_intensity_command(args)
    elif args.command == 'histogram':
        process_histogram_command(args)
    elif args.command == 'matching':
        process_matching_command(args)
    elif args.command == 'damage':
        process_damage_command(args)
    elif args.command == 'restore':
        process_restore_command(args)
    elif args.command == 'checkerboard':
        process_checkerboard_command(args)
    elif args.command == 'fourier':
        process_fourier_command(args)
    elif args.command == 'filter':
        process_filter_command(args)
    elif args.command == 'denoise':
        process_denoising_command(args)
    elif args.command == 'segmentation':
        process_segmentation_command(args)
    elif args.command == 'kmeans-sim':
        process_kmeans_sim_command(args)
    elif args.command == 'color-kmeans':
        process_color_kmeans_command(args)
    elif args.command == 'registration':
        process_registration_command(args)
    elif args.command == 'ml':
        process_machine_learning_command(args)
    elif args.command == 'multiscale':
        process_multiscale_command(args)
    else:
        print(f"Unknown command: {args.command}")
