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

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
