#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for simple manual selection of control points using OpenCV.

This module provides a simplified interface for manually selecting 
corresponding points between two images for image registration.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


class PointSelector:
    """Class for selecting points on an image using OpenCV."""
    
    def __init__(self, image, window_name="Select Points"):
        """
        Initialize the point selector.
        
        Args:
            image (ndarray): Input image
            window_name (str): Name of the window
        """
        self.image = image.copy()
        self.window_name = window_name
        self.points = []
        self.display_image = self.image.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        print(f"Select points on '{window_name}' and press 'q' when finished")
    
    def on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback function.
        
        Args:
            event: Mouse event type
            x: x-coordinate of mouse position
            y: y-coordinate of mouse position
            flags: Additional flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point to list
            self.points.append((x, y))
            
            # Draw circle at clicked position
            cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
            
            # Update display
            cv2.imshow(self.window_name, self.display_image)
            print(f"Point added at ({x}, {y})")
    
    def select_points(self):
        """
        Run the point selection process.
        
        Returns:
            list: List of selected points as (x, y) tuples
        """
        # Display the image
        cv2.imshow(self.window_name, self.display_image)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Close window
        cv2.destroyWindow(self.window_name)
        
        return self.points


def select_corresponding_points(image1, image2):
    """
    Select corresponding points on two images.
    
    Args:
        image1 (ndarray): First image
        image2 (ndarray): Second image
        
    Returns:
        tuple: (points1, points2) where points1 and points2 are lists of corresponding points
    """
    # Ensure images are in the right format for OpenCV (8-bit BGR)
    if len(image1.shape) == 2:  # Grayscale
        image1_display = cv2.cvtColor((image1 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color
        image1_display = (image1 * 255).astype(np.uint8)
    
    if len(image2.shape) == 2:  # Grayscale
        image2_display = cv2.cvtColor((image2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color
        image2_display = (image2 * 255).astype(np.uint8)
    
    # Select points on first image
    selector1 = PointSelector(image1_display, "Select Points on Source Image")
    points1 = selector1.select_points()
    print(f"Selected {len(points1)} points on source image")
    
    # Select points on second image
    selector2 = PointSelector(image2_display, "Select Points on Target Image")
    points2 = selector2.select_points()
    print(f"Selected {len(points2)} points on target image")
    
    # Ensure same number of points
    min_points = min(len(points1), len(points2))
    if min_points < len(points1):
        print(f"Warning: Truncating points from first image to {min_points}")
        points1 = points1[:min_points]
    if min_points < len(points2):
        print(f"Warning: Truncating points from second image to {min_points}")
        points2 = points2[:min_points]
    
    return points1, points2


def rigid_registration(data1, data2):
    """
    Rigid transformation estimation between n pairs of points.
    This function returns a transformation matrix.
    
    Args:
        data1 (ndarray): Array of source points with shape (n, 2)
        data2 (ndarray): Array of target points with shape (n, 2)
        
    Returns:
        ndarray: Transformation matrix T of size 2x3
    """
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # Computes barycenters, and recenters the points
    m1 = np.mean(data1, 0)
    m2 = np.mean(data2, 0)
    data1_inv_shifted = data1 - m1
    data2_inv_shifted = data2 - m2
    
    # Evaluates SVD
    K = np.matmul(np.transpose(data2_inv_shifted), data1_inv_shifted)
    U, S, V = np.linalg.svd(K)
    
    # Computes Rotation
    S = np.eye(2)
    S[1, 1] = np.linalg.det(U) * np.linalg.det(V)
    R = np.matmul(U, S)
    R = np.matmul(R, np.transpose(V))
    
    # Computes Translation
    t = m2 - np.matmul(R, m1)
    
    T = np.zeros((2, 3))
    T[0:2, 0:2] = R
    T[0:2, 2] = t
    
    return T


def register_images_with_manual_points(source_image, target_image):
    """
    Register images using manually selected control points.
    
    Args:
        source_image (ndarray): Source image
        target_image (ndarray): Target image
        
    Returns:
        tuple: (T, registered_image) where T is the transformation matrix
               and registered_image is the registered source image
    """
    # Select corresponding points
    source_points, target_points = select_corresponding_points(source_image, target_image)
    
    if len(source_points) < 2 or len(target_points) < 2:
        print("Error: At least 2 corresponding points are required for registration")
        return None, source_image
    
    # Estimate rigid transformation
    T = rigid_registration(source_points, target_points)
    
    # Apply transformation to source image
    rows, cols = target_image.shape[:2]
    registered_image = cv2.warpAffine(source_image, T, (cols, rows))
    
    # Visualize result
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    if len(source_image.shape) == 2:
        plt.imshow(source_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.title("Source Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    if len(target_image.shape) == 2:
        plt.imshow(target_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.title("Target Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    if len(registered_image.shape) == 2:
        plt.imshow(registered_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(registered_image, cv2.COLOR_BGR2RGB))
    plt.title("Registered Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('manual_registration_result.png')
    plt.show()
    
    return T, registered_image


if __name__ == "__main__":
    import sys
    from skimage import io, color
    
    # Default image paths
    source_image_path = "data/Brain1.bmp"
    target_image_path = "data/Brain2.bmp"
    
    # Load images
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
    
    # Register images
    T, registered_image = register_images_with_manual_points(source_gray, target_gray)
    
    # Display result
    try:
        from src.image_processing.image_registration import superimpose
        
        # Superimpose registered image with target image
        superimposed = superimpose(registered_image, target_gray, "rigid_manual.png")
        
        print("Registration completed successfully!")
    except ImportError:
        print("Warning: Could not import superimpose function")
