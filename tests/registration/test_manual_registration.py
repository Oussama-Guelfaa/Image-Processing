#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for manual point selection and registration.

This script demonstrates the manual selection of control points
for image registration.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import cv2

from src.image_processing.registration import (
    select_corresponding_points,
    rigid_registration,
    applyTransform,
    totuple,
    visualize_registration_result_opencv,
    superimpose
)


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
    plt.savefig('output/images/manual_original_images.png')
    plt.show()

    # Select corresponding points
    print("\nSelecting corresponding points...")
    source_points, target_points = select_corresponding_points(
        source_gray, target_gray,
        "Select Points on Source Image",
        "Select Points on Target Image"
    )

    print(f"Selected {len(source_points)} corresponding points")

    # Estimate rigid transformation
    print("\nEstimating rigid transformation...")
    T = rigid_registration(source_points, target_points)
    print("Transformation matrix:")
    print(T)

    # Apply transformation to source points
    transformed_points = applyTransform(source_points, T)

    # Visualize point pairs
    result_image = cv2.cvtColor((target_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Draw target points in blue
    for point in target_points:
        cv2.circle(result_image, totuple(point), 3, (255, 0, 0), -1)

    # Draw transformed source points in red
    for point in transformed_points:
        cv2.circle(result_image, totuple(point), 3, (0, 0, 255), -1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Control Points (Blue: Target, Red: Transformed Source)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('output/images/manual_control_points.png')
    plt.show()

    # Apply transformation to source image
    print("\nApplying transformation to source image...")
    rows, cols = target_gray.shape
    registered_image = cv2.warpAffine(source_gray, T, (cols, rows))

    # Visualize registration result
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(source_gray, cmap='gray')
    plt.title("Source Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(target_gray, cmap='gray')
    plt.title("Target Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(registered_image, cmap='gray')
    plt.title("Registered Image")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('output/images/manual_registration_result.png')
    plt.show()

    # Create and save superimposed image
    print("\nCreating superimposed image...")
    superimposed = superimpose(registered_image, target_gray, 'output/images/rigid_manual.png', show=True)

    print("\nManual registration test completed successfully!")


if __name__ == "__main__":
    main()
