#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to organize the output directory structure.

This script creates a consistent directory structure for output files
and moves existing output files to their appropriate locations.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import shutil
import glob
import re

# Define the output directory structure
OUTPUT_STRUCTURE = {
    'convolution': ['kernels', 'results'],
    'damage_modeling': ['damaged', 'restored', 'psf'],
    'denoising': ['noisy', 'filtered', 'comparison'],
    'filtering': ['lowpass', 'highpass', 'derivative', 'wiener'],
    'fourier': ['spectrum', 'phase', 'filtered'],
    'histogram': ['equalization', 'matching', 'comparison'],
    'machine_learning': ['features', 'classification', 'results'],
    'multiscale': ['pyramidal', 'scale_space'],
    'registration': ['manual', 'automatic', 'icp', 'results'],
    'segmentation': ['kmeans', 'thresholding', 'region_growing', 'results'],
    'transformations': ['gamma', 'contrast', 'logarithmic', 'results']
}

def create_output_structure():
    """Create the output directory structure."""
    print("Creating output directory structure...")
    
    # Create the main output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Create subdirectories for each module
    for module, subdirs in OUTPUT_STRUCTURE.items():
        module_dir = os.path.join('output', module)
        if not os.path.exists(module_dir):
            os.makedirs(module_dir)
        
        # Create subdirectories for each module
        for subdir in subdirs:
            subdir_path = os.path.join(module_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
    
    print("Output directory structure created successfully.")

def organize_existing_files():
    """Organize existing output files into the new directory structure."""
    print("Organizing existing output files...")
    
    # Get all image files in the output directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.gif', '*.bmp']:
        image_files.extend(glob.glob(os.path.join('output', ext)))
    
    # Define patterns to match files with their appropriate directories
    patterns = {
        'convolution': r'(conv|kernel|filter)',
        'damage_modeling': r'(damage|restore|psf|blur)',
        'denoising': r'(denoise|noise|filter)',
        'filtering': r'(filter|lowpass|highpass|derivative|wiener)',
        'fourier': r'(fourier|spectrum|phase)',
        'histogram': r'(histogram|equalize|match)',
        'machine_learning': r'(machine_learning|feature|classify)',
        'multiscale': r'(multiscale|pyramid|scale_space)',
        'registration': r'(registration|transform|icp)',
        'segmentation': r'(segment|kmeans|threshold)',
        'transformations': r'(transform|gamma|contrast|log)'
    }
    
    # Move files to their appropriate directories
    for file_path in image_files:
        file_name = os.path.basename(file_path)
        file_name_lower = file_name.lower()
        
        # Determine which module the file belongs to
        target_module = None
        for module, pattern in patterns.items():
            if re.search(pattern, file_name_lower):
                target_module = module
                break
        
        # If no module matches, put it in a general 'images' directory
        if target_module is None:
            target_dir = os.path.join('output', 'images')
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
        else:
            # Determine which subdirectory the file belongs to
            target_subdir = 'results'  # Default subdirectory
            for subdir in OUTPUT_STRUCTURE[target_module]:
                if re.search(subdir, file_name_lower):
                    target_subdir = subdir
                    break
            
            target_dir = os.path.join('output', target_module, target_subdir)
        
        # Move the file to its target directory
        target_path = os.path.join(target_dir, file_name)
        if file_path != target_path:
            try:
                shutil.move(file_path, target_path)
                print(f"Moved {file_path} to {target_path}")
            except Exception as e:
                print(f"Error moving {file_path}: {e}")
    
    print("Existing output files organized successfully.")

def clean_empty_directories():
    """Remove empty directories in the output directory."""
    print("Cleaning empty directories...")
    
    for root, dirs, files in os.walk('output', topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f"Removed empty directory: {dir_path}")
    
    print("Empty directories cleaned successfully.")

def main():
    """Main function to organize the output directory structure."""
    create_output_structure()
    organize_existing_files()
    clean_empty_directories()
    print("Output directory organization completed successfully.")

if __name__ == "__main__":
    main()
