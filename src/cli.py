#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for the image processing tools.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import sys
import os
import argparse
from pathlib import Path

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
  imgproc intensity --method gamma --gamma 0.5 --image path/to/image.jpg
  imgproc histogram --method custom --bins 256 --image path/to/image.jpg
  imgproc matching --method custom --peak1 0.3 --peak2 0.7 --image path/to/image.jpg
  imgproc segmentation --method kmeans --k 3 --image path/to/image.jpg
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

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, show help
    if args.command is None:
        parser.print_help()
        return

    # Import the main module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import main

    # Execute the command
    # Map the arguments to match main.py's expected format
    if args.command == 'intensity':
        # Convert --method to --type for intensity command
        if args.method == 'gamma':
            sys.argv = ["main.py", "intensity", "--type", "gamma", "--gamma", str(args.gamma)]
            if args.image:
                sys.argv.extend(["--image", args.image])
            if args.output:
                sys.argv.extend(["--output", args.output])
        elif args.method == 'contrast':
            sys.argv = ["main.py", "intensity", "--type", "contrast", "--E", str(args.E)]
            if args.image:
                sys.argv.extend(["--image", args.image])
            if args.output:
                sys.argv.extend(["--output", args.output])
        else:  # both
            sys.argv = ["main.py", "intensity", "--type", "both", "--gamma", str(args.gamma), "--E", str(args.E)]
            if args.image:
                sys.argv.extend(["--image", args.image])
            if args.output:
                sys.argv.extend(["--output", args.output])
    elif args.command in ['histogram', 'matching']:
        # These commands already have the same parameter names
        sys.argv = ["main.py", args.command]
        if args.method:
            sys.argv.extend(["--method", args.method])
        if args.bins:
            sys.argv.extend(["--bins", str(args.bins)])
        if args.image:
            sys.argv.extend(["--image", args.image])
        if args.output:
            sys.argv.extend(["--output", args.output])

        # Add additional parameters for matching command
        if args.command == 'matching':
            if hasattr(args, 'peak1'):
                sys.argv.extend(["--peak1", str(args.peak1)])
            if hasattr(args, 'peak2'):
                sys.argv.extend(["--peak2", str(args.peak2)])
            if hasattr(args, 'sigma1'):
                sys.argv.extend(["--sigma1", str(args.sigma1)])
            if hasattr(args, 'sigma2'):
                sys.argv.extend(["--sigma2", str(args.sigma2)])
            if hasattr(args, 'weight1'):
                sys.argv.extend(["--weight1", str(args.weight1)])
            if hasattr(args, 'weight2'):
                sys.argv.extend(["--weight2", str(args.weight2)])
    else:
        # For other commands, just pass the command name
        sys.argv = ["main.py", args.command]

    # Run the main function
    main.main()

if __name__ == "__main__":
    main()
