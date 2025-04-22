#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for simple manual point selection and registration.

This script demonstrates the manual selection of control points
for image registration using a simplified interface.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import cv2

from src.image_processing.simple_manual_selection import register_images_with_manual_points
from src.image_processing.image_registration import superimpose


def main():
    """Main function to test manual point selection and registration."""
    # Load images
    print("Loading images...")
    source_image_path = "data/Brain1.bmp"
    target_image_path = "data/Brain2.bmp"
    
    source_image = io.imread(source_image_path)
    target_image = io.imread(target_image_path)
    
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
    plt.savefig('simple_manual_original_images.png')
    plt.show()
    
    # Register images using manual point selection
    print("\nRegistering images using manual point selection...")
    print("Please select corresponding points on both images.")
    print("Click on distinctive features that you can identify in both images.")
    print("Press 'q' when you have finished selecting points on each image.")
    
    T, registered_image = register_images_with_manual_points(source_gray, target_gray)
    
    if T is None:
        print("Registration failed. Please try again with more points.")
        return
    
    print("\nTransformation matrix:")
    print(T)
    
    # Create and save superimposed image
    print("\nCreating superimposed image...")
    superimposed = superimpose(registered_image, target_gray, 'simple_manual_superimposed.png', show=True)
    
    print("\nSimple manual registration test completed successfully!")


if __name__ == "__main__":
    main()
