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

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
