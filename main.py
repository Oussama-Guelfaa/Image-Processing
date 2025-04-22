"""
Main entry point for the image processing project.

This script provides a simple command-line interface to access the various
image processing and visualization functionalities of the project.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt

def main():
    """Main function that parses command line arguments and runs the appropriate module."""
    parser = argparse.ArgumentParser(description='Image Processing Project')

    # Add subparsers for different functionalities
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Fourier transform parser
    fourier_parser = subparsers.add_parser('fourier', help='Run Fourier transform analysis')
    fourier_parser.add_argument('--image', default='data/cornee.png', help='Path to the image file')

    # Filtering parser
    filter_parser = subparsers.add_parser('filter', help='Apply filters to an image')
    filter_parser.add_argument('--image', default='data/cornee.png', help='Path to the image file')
    filter_parser.add_argument('--type', choices=['lowpass', 'highpass'], default='lowpass',
                              help='Type of filter to apply')
    filter_parser.add_argument('--cutoff', type=int, default=30, help='Cutoff frequency for the filter')

    # Graph visualization parser
    graph_parser = subparsers.add_parser('graph', help='Generate a graph visualization')

    # Segmentation parser
    segmentation_parser = subparsers.add_parser('segmentation', help='Run image segmentation')
    segmentation_parser.add_argument('--method', choices=['threshold', 'kmeans', 'auto', 'otsu', 'all'], default='all',
                                  help='Segmentation method to use')

    # K-means simulation parser
    kmeans_parser = subparsers.add_parser('kmeans-sim', help='Run K-means clustering simulation in 2D')

    # Color K-means segmentation parser
    color_kmeans_parser = subparsers.add_parser('color-kmeans', help='Run color image segmentation using K-means in 3D')
    color_kmeans_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: Tv16.png)')
    color_kmeans_parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for K-means')
    color_kmeans_parser.add_argument('--output', type=str, default=None, help='Path to save the segmented image')

    # Intensity transformations parser
    intensity_parser = subparsers.add_parser('intensity', help='Apply intensity transformations (LUT) to an image')
    intensity_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: osteoblast.jpg)')
    intensity_parser.add_argument('--type', choices=['gamma', 'contrast', 'both'], default='both',
                                help='Type of transformation to apply')
    intensity_parser.add_argument('--gamma', type=float, default=2.0, help='Gamma value for gamma correction')
    intensity_parser.add_argument('--E', type=float, default=4.0, help='E parameter for contrast stretching')
    intensity_parser.add_argument('--output', type=str, default=None, help='Path to save the transformed image')

    # Histogram equalization parser
    histogram_parser = subparsers.add_parser('histogram', help='Apply histogram equalization to an image')
    histogram_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: osteoblaste.jpg)')
    histogram_parser.add_argument('--method', choices=['builtin', 'custom', 'both'], default='both',
                                help='Method to use for histogram equalization')
    histogram_parser.add_argument('--bins', type=int, default=256, help='Number of bins for the histogram')
    histogram_parser.add_argument('--output', type=str, default=None, help='Path to save the equalized image')

    # Histogram matching parser
    matching_parser = subparsers.add_parser('matching', help='Apply histogram matching to an image')
    matching_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: phobos.jpg)')
    matching_parser.add_argument('--method', choices=['builtin', 'custom', 'both'], default='both',
                                help='Method to use for histogram matching')
    matching_parser.add_argument('--bins', type=int, default=256, help='Number of bins for the histogram')
    matching_parser.add_argument('--peak1', type=float, default=0.3, help='Position of the first peak in the bimodal histogram')
    matching_parser.add_argument('--peak2', type=float, default=0.7, help='Position of the second peak in the bimodal histogram')
    matching_parser.add_argument('--sigma1', type=float, default=0.05, help='Standard deviation of the first peak')
    matching_parser.add_argument('--sigma2', type=float, default=0.05, help='Standard deviation of the second peak')
    matching_parser.add_argument('--weight1', type=float, default=0.6, help='Weight of the first peak')
    matching_parser.add_argument('--weight2', type=float, default=0.4, help='Weight of the second peak')
    matching_parser.add_argument('--output', type=str, default=None, help='Path to save the matched image')

    # Damage modeling parser
    damage_parser = subparsers.add_parser('damage', help='Apply damage to an image using convolution and noise')
    damage_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: jupiter.jpg)')
    damage_parser.add_argument('--psf', choices=['gaussian', 'motion'], default='gaussian',
                             help='Type of Point Spread Function to use')
    damage_parser.add_argument('--sigma', type=float, default=3.0, help='Sigma parameter for Gaussian PSF')
    damage_parser.add_argument('--length', type=int, default=15, help='Length parameter for motion blur PSF')
    damage_parser.add_argument('--angle', type=float, default=45.0, help='Angle parameter for motion blur PSF (in degrees)')
    damage_parser.add_argument('--noise', type=float, default=0.01, help='Noise level for the damage')
    damage_parser.add_argument('--output', type=str, default=None, help='Path to save the damaged image')

    # Restoration parser
    restore_parser = subparsers.add_parser('restore', help='Restore a damaged image using deconvolution')
    restore_parser.add_argument('--image', type=str, default=None, help='Path to the damaged image file (default: jupiter.jpg)')
    restore_parser.add_argument('--psf', choices=['gaussian', 'motion'], default='gaussian',
                              help='Type of Point Spread Function used for the damage')
    restore_parser.add_argument('--sigma', type=float, default=3.0, help='Sigma parameter for Gaussian PSF')
    restore_parser.add_argument('--length', type=int, default=15, help='Length parameter for motion blur PSF')
    restore_parser.add_argument('--angle', type=float, default=45.0, help='Angle parameter for motion blur PSF (in degrees)')
    restore_parser.add_argument('--method', choices=['inverse', 'wiener', 'compare'], default='wiener',
                              help='Restoration method to use')
    restore_parser.add_argument('--k', type=float, default=0.01, help='K parameter for Wiener filter')
    restore_parser.add_argument('--output', type=str, default=None, help='Path to save the restored image')

    # Checkerboard parser
    checkerboard_parser = subparsers.add_parser('checkerboard', help='Generate a checkerboard image')
    checkerboard_parser.add_argument('--size', type=int, default=8, help='Number of squares in each dimension')
    checkerboard_parser.add_argument('--square_size', type=int, default=32, help='Size of each square in pixels')
    checkerboard_parser.add_argument('--output', type=str, default=None, help='Path to save the checkerboard image')

    # Image registration parser
    registration_parser = subparsers.add_parser('registration', help='Register two images')
    registration_parser.add_argument('--source', type=str, default=None, help='Path to the source image (default: Brain1.bmp)')
    registration_parser.add_argument('--target', type=str, default=None, help='Path to the target image (default: Brain2.bmp)')
    registration_parser.add_argument('--method', choices=['manual', 'automatic', 'icp'], default='manual',
                                  help='Registration method to use')
    registration_parser.add_argument('--output', type=str, default=None, help='Path to save the registered image')
    registration_parser.add_argument('--superimpose', action='store_true', help='Generate a superimposed image of the registration result')
    registration_parser.add_argument('--superimpose_output', type=str, default=None, help='Path to save the superimposed image')

    # Parse arguments
    args = parser.parse_args()

    # Run the appropriate module based on the command
    if args.command == 'fourier':
        print(f"Running Fourier transform analysis on {args.image}")
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.image_processing import transformer_fourier
        # The module will run automatically when imported

    elif args.command == 'filter':
        print(f"Applying {args.type} filter to {args.image} with cutoff {args.cutoff}")
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.image_processing import filtering_hp_lp
        # The module will run automatically when imported

    elif args.command == 'graph':
        print("Generating graph visualization")
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from src.visualization import draw
        # The module will run automatically when imported

    elif args.command == 'segmentation':
        print(f"Running image segmentation using {args.method} method")
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Import the segmentation module
        from src.image_processing import segmentation

        # The module will run automatically when imported, but we can also
        # customize the behavior based on the method argument
        if args.method != 'all':
            print(f"Note: The segmentation module currently shows all methods regardless of the --method argument.")

    elif args.command == 'kmeans-sim':
        print("Running K-means clustering simulation in 2D")
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Import the kmeans simulation module
        from src.image_processing import kmeans_simulation
        # The module will run automatically when imported

    elif args.command == 'color-kmeans':
        image_info = f" on {args.image}" if args.image else ""
        output_info = f" and saving to {args.output}" if args.output else ""
        print(f"Running color image segmentation using K-means with {args.clusters} clusters{image_info}{output_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Import the color kmeans module
        from src.image_processing import color_kmeans
        # The module will run automatically when imported

    elif args.command == 'intensity':
        image_info = f" on {args.image}" if args.image else ""
        output_info = f" and saving to {args.output}" if args.output else ""

        if args.type == 'gamma':
            print(f"Applying gamma correction with gamma={args.gamma}{image_info}{output_info}")
        elif args.type == 'contrast':
            print(f"Applying contrast stretching with E={args.E}{image_info}{output_info}")
        else:  # both
            print(f"Applying both gamma correction (gamma={args.gamma}) and contrast stretching (E={args.E}){image_info}{output_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    elif args.command == 'histogram':
        image_info = f" on {args.image}" if args.image else ""
        output_info = f" and saving to {args.output}" if args.output else ""
        bins_info = f" with {args.bins} bins"

        if args.method == 'builtin':
            print(f"Applying histogram equalization using built-in method{image_info}{bins_info}{output_info}")
        elif args.method == 'custom':
            print(f"Applying histogram equalization using custom method{image_info}{bins_info}{output_info}")
        else:  # both
            print(f"Applying histogram equalization using both methods{image_info}{bins_info}{output_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    elif args.command == 'matching':
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

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    elif args.command == 'damage':
        image_info = f" on {args.image}" if args.image else ""
        output_info = f" and saving to {args.output}" if args.output else ""
        psf_info = f" using {args.psf} PSF"
        noise_info = f" with noise level {args.noise}"

        print(f"Applying damage to image{image_info}{psf_info}{noise_info}{output_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Import the damage modeling module
        from src.image_processing import damage_modeling

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

    elif args.command == 'restore':
        image_info = f" on {args.image}" if args.image else ""
        output_info = f" and saving to {args.output}" if args.output else ""
        psf_info = f" using {args.psf} PSF"
        method_info = f" with {args.method} method"

        print(f"Restoring image{image_info}{psf_info}{method_info}{output_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        # Import the damage modeling module
        from src.image_processing import damage_modeling

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

    elif args.command == 'checkerboard':
        print(f"Generating checkerboard image with {args.size}x{args.size} squares of size {args.square_size}px")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

    elif args.command == 'registration':
        source_info = f" source: {args.source}" if args.source else ""
        target_info = f" target: {args.target}" if args.target else ""
        output_info = f" output: {args.output}" if args.output else ""
        superimpose_info = " with superimposition" if args.superimpose else ""

        print(f"Running image registration using {args.method} method{source_info}{target_info}{output_info}{superimpose_info}")

        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        from skimage import io, color
        import numpy as np
        import cv2

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

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
