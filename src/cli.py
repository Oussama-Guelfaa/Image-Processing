#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cli

Module for image processing operations.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import argparse
import sys
import os
import matplotlib.pyplot as plt

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
  imgproc denoise --method median --kernel_size 5 --image path/to/image.jpg
  imgproc registration --method manual --source path/to/source.jpg --target path/to/target.jpg

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

    # Fourier transform parser
    fourier_parser = subparsers.add_parser('fourier', help='Run Fourier transform analysis')
    fourier_parser.add_argument('--image', default='data/cornee.png', help='Path to the image file')
    fourier_parser.add_argument('--output', type=str, default=None, help='Path to save the output image')

    # Filtering parser
    filter_parser = subparsers.add_parser('filter', help='Apply filters to an image')
    filter_parser.add_argument('--image', default='data/cornee.png', help='Path to the image file')
    filter_parser.add_argument('--type', choices=['lowpass', 'highpass'], default='lowpass',
                              help='Type of filter to apply')
    filter_parser.add_argument('--cutoff', type=int, default=30, help='Cutoff frequency for the filter')
    filter_parser.add_argument('--output', type=str, default=None, help='Path to save the output image')

    # Denoising parser
    denoising_parser = subparsers.add_parser('denoise', help='Apply denoising techniques to an image')
    denoising_parser.add_argument('--image', default='data/jambe.tif', help='Path to the image file')
    denoising_parser.add_argument('--noise', choices=['uniform', 'gaussian', 'salt_pepper', 'exponential'],
                                default='gaussian', help='Type of noise to add')
    denoising_parser.add_argument('--method',
                                choices=['mean', 'median', 'gaussian', 'bilateral', 'nlm', 'adaptive_median', 'fast_adaptive_median', 'all'],
                                default='all', help='Denoising method to use')
    denoising_parser.add_argument('--noise_param', type=float, default=0.1,
                                help='Noise parameter (std for gaussian, a/b for uniform, etc.)')
    denoising_parser.add_argument('--kernel_size', type=int, default=3,
                                help='Kernel size for mean and median filters')
    denoising_parser.add_argument('--max_window_size', type=int, default=7,
                                help='Maximum window size for adaptive median filter')
    denoising_parser.add_argument('--sigma', type=float, default=1.0,
                                help='Sigma for Gaussian filter')
    denoising_parser.add_argument('--output', type=str, default=None,
                                help='Path to save the denoised image')

    # Segmentation parser
    segmentation_parser = subparsers.add_parser('segmentation', help='Run image segmentation')
    segmentation_parser.add_argument('--method', choices=['threshold', 'kmeans', 'auto', 'otsu', 'all'], default='all',
                                  help='Segmentation method to use')
    segmentation_parser.add_argument('--image', type=str, default=None, help='Path to the image file')
    segmentation_parser.add_argument('--output', type=str, default=None, help='Path to save the segmented image')

    # K-means simulation parser
    kmeans_parser = subparsers.add_parser('kmeans-sim', help='Run K-means clustering simulation in 2D')
    kmeans_parser.add_argument('--output', type=str, default=None, help='Path to save the simulation results')

    # Color K-means segmentation parser
    color_kmeans_parser = subparsers.add_parser('color-kmeans', help='Run color image segmentation using K-means in 3D')
    color_kmeans_parser.add_argument('--image', type=str, default=None, help='Path to the image file (default: Tv16.png)')
    color_kmeans_parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for K-means')
    color_kmeans_parser.add_argument('--output', type=str, default=None, help='Path to save the segmented image')

    # Image registration parser
    registration_parser = subparsers.add_parser('registration', help='Register two images')
    registration_parser.add_argument('--source', type=str, default=None, help='Path to the source image (default: Brain1.bmp)')
    registration_parser.add_argument('--target', type=str, default=None, help='Path to the target image (default: Brain2.bmp)')
    registration_parser.add_argument('--method', choices=['manual', 'automatic', 'icp'], default='manual',
                                  help='Registration method to use')
    registration_parser.add_argument('--output', type=str, default=None, help='Path to save the registered image')
    registration_parser.add_argument('--superimpose', action='store_true', help='Generate a superimposed image of the registration result')
    registration_parser.add_argument('--superimpose_output', type=str, default=None, help='Path to save the superimposed image')

    # Machine learning parser
    ml_parser = subparsers.add_parser('ml', help='Apply machine learning techniques to images')
    ml_parser.add_argument('--task', choices=['extract', 'train', 'classify', 'evaluate', 'visualize'], default='train',
                         help='Machine learning task to perform')
    ml_parser.add_argument('--dataset', type=str, default='data/images_Kimia216',
                         help='Path to the dataset directory (default: data/images_Kimia216)')
    ml_parser.add_argument('--classifier', choices=['mlp', 'svm', 'rf', 'ensemble'], default='ensemble',
                         help='Classifier to use (default: ensemble)')
    ml_parser.add_argument('--features', choices=['hu', 'zernike', 'geometric', 'hog', 'all'], default='all',
                         help='Features to extract (default: all)')
    ml_parser.add_argument('--test_size', type=float, default=0.25,
                         help='Proportion of the dataset to include in the test split (default: 0.25)')
    ml_parser.add_argument('--image', type=str, default=None,
                         help='Path to the image to classify (required for classify task)')
    ml_parser.add_argument('--model', type=str, default=None,
                         help='Path to the saved model (required for classify task)')
    ml_parser.add_argument('--output', type=str, default='output/machine_learning',
                         help='Path to save the output (default: output/machine_learning)')
    ml_parser.add_argument('--cross_validate', action='store_true',
                         help='Perform cross-validation')

    # Multiscale analysis parser
    multiscale_parser = subparsers.add_parser('multiscale', help='Apply multiscale analysis techniques to images')
    multiscale_parser.add_argument('--image', type=str, default=None,
                                help='Path to the image file (default: cerveau.jpg)')
    multiscale_parser.add_argument('--levels', type=int, default=4,
                                help='Number of levels in the pyramid (default: 4)')
    multiscale_parser.add_argument('--sigma', type=float, default=1.0,
                                help='Sigma for Gaussian filter (default: 1.0)')
    multiscale_parser.add_argument('--output', type=str, default=None,
                                help='Path to save the output images')
    multiscale_parser.add_argument('--compare', action='store_true',
                                help='Compare reconstruction with and without details')
    multiscale_parser.add_argument('image_path', type=str, nargs='?', default=None,
                                help='Path to the input image (positional argument)')

    # Parse arguments
    args = parser.parse_args()

    # If no command is provided, show help
    if args.command is None:
        parser.print_help()
        return

    # Process the command
    # First, handle the case where image_path is provided as a positional argument
    if hasattr(args, 'image_path') and args.image_path and not args.image:
        args.image = args.image_path

    # Convert CLI arguments to the format expected by the processing functions
    if args.command == 'intensity':
        args.type = args.method

    # Import the core module
    from src import core

    # Process the command
    core.process_command(args)

if __name__ == "__main__":
    main()
