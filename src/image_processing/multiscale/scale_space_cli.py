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

from .scale_space import (
    morphological_multiscale,
    kramer_bruckner_multiscale,
    visualize_morphological_pyramid,
    visualize_kb_filters
)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Scale-space decomposition and multiscale filtering'
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
        default='output/multiscale/scale_space',
        help='Directory to save output images (default: output/multiscale/scale_space)'
    )
    
    parser.add_argument(
        '--levels',
        type=int,
        default=4,
        help='Number of levels for morphological decomposition (default: 4)'
    )
    
    parser.add_argument(
        '--kb-iterations',
        type=int,
        default=3,
        help='Number of iterations for Kramer-Bruckner filter (default: 3)'
    )
    
    parser.add_argument(
        '--radius',
        type=int,
        default=5,
        help='Radius of the structuring element (disk) (default: 5)'
    )
    
    parser.add_argument(
        '--method',
        choices=['morphological', 'kb', 'both'],
        default='both',
        help='Decomposition method to use (default: both)'
    )
    
    return parser.parse_args()

def main():
    """
    Main function for the command-line interface.
    """
    # Parse command-line arguments
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
    if args.method in ['morphological', 'both']:
        # Morphological Multiscale Decomposition
        print("Performing Morphological Multiscale Decomposition...")
        pyramid_dilations, pyramid_erosions = morphological_multiscale(image, levels=args.levels)
        
        # Visualize morphological pyramids
        fig_morpho = visualize_morphological_pyramid(pyramid_dilations, pyramid_erosions)
        plt.savefig(os.path.join(args.output_dir, 'morphological_decomposition.png'))
        plt.close(fig_morpho)
    
    if args.method in ['kb', 'both']:
        # Kramer and Bruckner Multiscale Decomposition
        print("Performing Kramer and Bruckner Multiscale Decomposition...")
        kb_filters = kramer_bruckner_multiscale(image, levels=args.kb_iterations, radius=args.radius)
        
        # Visualize KB filters
        fig_kb = visualize_kb_filters(kb_filters, f"Kramer and Bruckner Decomposition (n={args.kb_iterations}, r={args.radius})")
        plt.savefig(os.path.join(args.output_dir, f'kb_decomposition_n{args.kb_iterations}.png'))
        plt.close(fig_kb)
    
    print(f"Scale-space decomposition and multiscale filtering completed successfully!")
    print(f"Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
