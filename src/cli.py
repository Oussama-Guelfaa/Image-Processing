#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for the image processing tools.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import argparse

def main():
    """
    Main entry point for the command-line interface.
    """
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Image Processing Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using named arguments
  imgproc intensity --method gamma --gamma 0.5 --image path/to/image.jpg
  imgproc histogram --method custom --bins 256 --image path/to/image.jpg
  imgproc matching --method custom --peak1 0.3 --peak2 0.7 --image path/to/image.jpg

  # Using positional arguments for image path
  imgproc intensity --method gamma --gamma 0.5 path/to/image.jpg
  imgproc histogram --method custom --bins 256 path/to/image.jpg
  imgproc matching --method custom --peak1 0.3 --peak2 0.7 path/to/image.jpg
        """
    )

    # Add version argument
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Intensity transformations command
    intensity_parser = subparsers.add_parser('intensity', help='Apply intensity transformations')
    intensity_parser.add_argument('--method', choices=['gamma', 'contrast', 'both'], default='both',
                        help='Method to use for intensity transformation')
    intensity_parser.add_argument('--gamma', type=float, default=0.5,
                        help='Gamma value for gamma correction')
    intensity_parser.add_argument('--E', type=float, default=4.0,
                        help='E parameter for contrast stretching')
    intensity_parser.add_argument('--m', type=float, default=0.5,
                        help='m parameter for contrast stretching')
    intensity_parser.add_argument('--image', type=str, default=None,
                        help='Path to the input image (default: use sample image)')
    intensity_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output image')
    intensity_parser.add_argument('image_path', type=str, nargs='?', default=None,
                        help='Path to the input image (positional argument)')

    # Damage modeling command
    damage_parser = subparsers.add_parser('damage', help='Apply damage to an image using convolution and noise')
    damage_parser.add_argument('--psf', choices=['gaussian', 'motion'], default='gaussian',
                        help='Type of Point Spread Function to use')
    damage_parser.add_argument('--sigma', type=float, default=3.0,
                        help='Sigma parameter for Gaussian PSF')
    damage_parser.add_argument('--length', type=int, default=15,
                        help='Length parameter for motion blur PSF')
    damage_parser.add_argument('--angle', type=float, default=45.0,
                        help='Angle parameter for motion blur PSF (in degrees)')
    damage_parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level for the damage')
    damage_parser.add_argument('--image', type=str, default=None,
                        help='Path to the input image (default: use sample image)')
    damage_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the damaged image')
    damage_parser.add_argument('image_path', type=str, nargs='?', default=None,
                        help='Path to the input image (positional argument)')

    # Restoration command
    restore_parser = subparsers.add_parser('restore', help='Restore a damaged image using deconvolution')
    restore_parser.add_argument('--method', choices=['inverse', 'wiener', 'compare'], default='wiener',
                        help='Restoration method to use')
    restore_parser.add_argument('--psf', choices=['gaussian', 'motion'], default='gaussian',
                        help='Type of Point Spread Function used for the damage')
    restore_parser.add_argument('--sigma', type=float, default=3.0,
                        help='Sigma parameter for Gaussian PSF')
    restore_parser.add_argument('--length', type=int, default=15,
                        help='Length parameter for motion blur PSF')
    restore_parser.add_argument('--angle', type=float, default=45.0,
                        help='Angle parameter for motion blur PSF (in degrees)')
    restore_parser.add_argument('--k', type=float, default=0.01,
                        help='K parameter for Wiener filter')
    restore_parser.add_argument('--image', type=str, default=None,
                        help='Path to the damaged image file (default: use sample image)')
    restore_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the restored image')
    restore_parser.add_argument('image_path', type=str, nargs='?', default=None,
                        help='Path to the input image (positional argument)')

    # Checkerboard command
    checkerboard_parser = subparsers.add_parser('checkerboard', help='Generate a checkerboard image')
    checkerboard_parser.add_argument('--size', type=int, default=8,
                        help='Number of squares in each dimension')
    checkerboard_parser.add_argument('--square_size', type=int, default=32,
                        help='Size of each square in pixels')
    checkerboard_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the checkerboard image')

    # Histogram equalization command
    histogram_parser = subparsers.add_parser('histogram', help='Apply histogram equalization')
    histogram_parser.add_argument('--method', choices=['builtin', 'custom', 'both'], default='both',
                        help='Method to use for histogram equalization')
    histogram_parser.add_argument('--bins', type=int, default=256,
                        help='Number of bins for the histogram')
    histogram_parser.add_argument('--image', type=str, default=None,
                        help='Path to the input image (default: use sample image)')
    histogram_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output image')
    histogram_parser.add_argument('image_path', type=str, nargs='?', default=None,
                        help='Path to the input image (positional argument)')

    # Histogram matching command
    matching_parser = subparsers.add_parser('matching', help='Apply histogram matching')
    matching_parser.add_argument('--method', choices=['builtin', 'custom', 'both'], default='both',
                        help='Method to use for histogram matching')
    matching_parser.add_argument('--bins', type=int, default=256,
                        help='Number of bins for the histogram')
    matching_parser.add_argument('--peak1', type=float, default=0.25,
                        help='Position of the first peak in the bimodal histogram')
    matching_parser.add_argument('--peak2', type=float, default=0.75,
                        help='Position of the second peak in the bimodal histogram')
    matching_parser.add_argument('--sigma1', type=float, default=0.05,
                        help='Standard deviation of the first peak')
    matching_parser.add_argument('--sigma2', type=float, default=0.05,
                        help='Standard deviation of the second peak')
    matching_parser.add_argument('--weight1', type=float, default=0.5,
                        help='Weight of the first peak')
    matching_parser.add_argument('--weight2', type=float, default=0.5,
                        help='Weight of the second peak')
    matching_parser.add_argument('--image', type=str, default=None,
                        help='Path to the input image (default: use sample image)')
    matching_parser.add_argument('--output', type=str, default=None,
                        help='Path to save the output image')
    matching_parser.add_argument('image_path', type=str, nargs='?', default=None,
                        help='Path to the input image (positional argument)')

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, show help
    if args.command is None:
        parser.print_help()
        return

    # Import the core module
    from src import core

    # Execute the command
    # Use positional argument for image if provided
    if args.command in ['intensity', 'histogram', 'matching'] and hasattr(args, 'image_path') and args.image_path and not args.image:
        args.image = args.image_path

    # Convert CLI arguments to the format expected by core.py
    if args.command == 'intensity':
        # Convert --method to --type for intensity command
        args.type = args.method

    # Process the command
    core.process_command(args)

if __name__ == "__main__":
    main()
