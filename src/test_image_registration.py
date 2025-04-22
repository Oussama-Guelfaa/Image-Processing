#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for image registration module.

This script demonstrates the use of the image registration module
on brain MRI images.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from src.image_processing.image_registration import (
    estimate_rigid_transform,
    apply_rigid_transform,
    icp_registration,
    detect_corners,
    visualize_point_pairs,
    visualize_registration_result,
    superimpose
)


def main():
    """Main function to test image registration."""
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

    # Visualize original images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(source_gray, cmap='gray')
    plt.title('Source Image (brain1)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(target_gray, cmap='gray')
    plt.title('Target Image (brain2)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('original_images.png')
    plt.show()

    # Define control points (as given in the tutorial)
    print("Using predefined control points...")
    A_points = np.array([[136, 100], [127, 153], [96, 156], [87, 99]])
    B_points = np.array([[144, 99], [109, 140], [79, 128], [100, 74]])

    # Visualize control points
    visualize_point_pairs(source_gray, A_points, target_gray, B_points,
                         title="Predefined Control Points")
    plt.savefig('control_points.png')

    # 1. Direct rigid transformation estimation
    print("\n1. Testing direct rigid transformation estimation...")
    R, t = estimate_rigid_transform(A_points, B_points)
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(t)

    # Apply transformation
    registered_image = apply_rigid_transform(source_gray, R, t)

    # Visualize result
    visualize_registration_result(source_gray, target_gray, registered_image,
                                 title="Image Registration with Direct Estimation")
    plt.savefig('direct_registration.png')

    # Create and save superimposed image
    print("Creating superimposed image for direct estimation...")
    superimposed_direct = superimpose(registered_image, target_gray, 'direct_superimposed.png')

    # 2. ICP registration with shuffled points
    print("\n2. Testing ICP registration with shuffled points...")
    # Shuffle source points to simulate incorrect pairing
    shuffled_A_points = A_points.copy()
    np.random.shuffle(shuffled_A_points)

    # Visualize shuffled points
    visualize_point_pairs(source_gray, shuffled_A_points, target_gray, B_points,
                         title="Shuffled Control Points")
    plt.savefig('shuffled_points.png')

    # Apply ICP
    R_icp, t_icp, transformed_points, error = icp_registration(
        shuffled_A_points, B_points, max_iterations=20
    )
    print(f"ICP completed with error: {error:.6f}")
    print("ICP Rotation matrix:")
    print(R_icp)
    print("ICP Translation vector:")
    print(t_icp)

    # Apply ICP transformation
    registered_image_icp = apply_rigid_transform(source_gray, R_icp, t_icp)

    # Visualize ICP result
    visualize_registration_result(source_gray, target_gray, registered_image_icp,
                                title="Image Registration with ICP (Shuffled Points)")
    plt.savefig('icp_registration.png')

    # Create and save superimposed image for ICP
    print("Creating superimposed image for ICP registration...")
    superimposed_icp = superimpose(registered_image_icp, target_gray, 'icp_superimposed.png')

    # 3. Automatic corner detection
    print("\n3. Testing registration with automatic corner detection...")
    source_corners = detect_corners(source_gray, max_corners=10)
    target_corners = detect_corners(target_gray, max_corners=10)

    # Ensure same number of points
    min_corners = min(len(source_corners), len(target_corners))
    source_corners = source_corners[:min_corners]
    target_corners = target_corners[:min_corners]

    # Visualize detected corners
    visualize_point_pairs(source_gray, source_corners, target_gray, target_corners,
                         title="Automatically Detected Corners")
    plt.savefig('automatic_corners.png')

    # Apply ICP with detected corners
    R_auto, t_auto, transformed_corners, error_auto = icp_registration(
        source_corners, target_corners, max_iterations=50
    )
    print(f"ICP with automatic corners completed with error: {error_auto:.6f}")

    # Apply transformation
    registered_image_auto = apply_rigid_transform(source_gray, R_auto, t_auto)

    # Visualize result
    visualize_registration_result(source_gray, target_gray, registered_image_auto,
                                title="Image Registration with Automatic Corner Detection and ICP")
    plt.savefig('automatic_registration.png')

    # Create and save superimposed image for automatic corner detection
    print("Creating superimposed image for automatic corner detection...")
    superimposed_auto = superimpose(registered_image_auto, target_gray, 'automatic_superimposed.png')

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    main()
