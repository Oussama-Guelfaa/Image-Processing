"""
Main entry point for the image processing project.

This script provides a simple command-line interface to access the various
image processing and visualization functionalities of the project.
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
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
