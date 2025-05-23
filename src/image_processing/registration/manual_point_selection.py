#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registration

Image registration techniques including manual point selection and rigid transformation estimation.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color


# Global variables for point selection
pts = []
I = None


def on_mouse(event, x, y, flags, param):
    """
    Callback method for detecting click on image.
    It draws a circle on the global variable image I.
    
    Args:
        event: Mouse event type
        x: x-coordinate of mouse position
        y: y-coordinate of mouse position
        flags: Additional flags
        param: Additional parameters
    """
    global pts, I
    if event == cv2.EVENT_LBUTTONUP:
        pts.append((x, y))
        cv2.circle(I, (x, y), 2, (0, 0, 255), -1)


def cpselect(image, title="Select Points"):
    """
    Method for manually selecting the control points.
    It waits until 'q' key is pressed.
    
    Args:
        image (ndarray): Input image
        title (str): Window title
        
    Returns:
        list: List of selected points as (x, y) tuples
    """
    global pts, I
    
    # Reset points list
    pts = []
    
    # Make a copy of the image for display
    if len(image.shape) == 2:  # Grayscale image
        I = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color image
        I = image.copy()
    
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    
    print(f"Select points on '{title}' and press 'q' when finished")
    
    # Keep looping until the 'q' key is pressed
    while True:
        # Display the image and wait for a keypress
        cv2.imshow(title, I)
        key = cv2.waitKey(1) & 0xFF
        
        # If the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break
    
    # Close all open windows
    cv2.destroyAllWindows()
    
    return pts


def select_corresponding_points(image1, image2, title1="Select Points on Image 1", title2="Select Points on Image 2"):
    """
    Select corresponding points on two images.
    
    Args:
        image1 (ndarray): First image
        image2 (ndarray): Second image
        title1 (str): Title for the first image window
        title2 (str): Title for the second image window
        
    Returns:
        tuple: (points1, points2) where points1 and points2 are lists of corresponding points
    """
    # Ensure images are in the right format for OpenCV
    if len(image1.shape) == 2:  # Grayscale
        image1_display = (image1 * 255).astype(np.uint8)
    else:  # Color
        image1_display = (image1 * 255).astype(np.uint8)
    
    if len(image2.shape) == 2:  # Grayscale
        image2_display = (image2 * 255).astype(np.uint8)
    else:  # Color
        image2_display = (image2 * 255).astype(np.uint8)
    
    # Select points on first image
    points1 = cpselect(image1_display, title1)
    print(f"Selected {len(points1)} points on first image")
    
    # Select points on second image
    points2 = cpselect(image2_display, title2)
    print(f"Selected {len(points2)} points on second image")
    
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
    This function returns a rotation R and a translation t.
    
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


def applyTransform(points, T):
    """
    Apply transform to a list of points.
    
    Args:
        points (list): List of points
        T (ndarray): Rigid transformation matrix (shape 2x3)
        
    Returns:
        ndarray: Transformed points
    """
    dataA = np.array(points)
    src = np.array([dataA])
    data_dest = cv2.transform(src, T)
    a, b, c = data_dest.shape
    data_dest = np.reshape(data_dest, (b, c))
    
    return data_dest


def totuple(point):
    """
    Convert a point to a tuple of integers.
    
    Args:
        point (ndarray): Point coordinates
        
    Returns:
        tuple: Point coordinates as integers
    """
    return (int(point[0]), int(point[1]))


def visualize_registration_result_opencv(source_image, target_image, source_points, target_points, transformed_points):
    """
    Visualize registration result using OpenCV.
    
    Args:
        source_image (ndarray): Source image
        target_image (ndarray): Target image
        source_points (list): Source points
        target_points (list): Target points
        transformed_points (ndarray): Transformed source points
    """
    # Ensure images are in the right format for OpenCV
    if len(source_image.shape) == 2:  # Grayscale
        source_display = cv2.cvtColor((source_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color
        source_display = (source_image * 255).astype(np.uint8)
    
    if len(target_image.shape) == 2:  # Grayscale
        target_display = cv2.cvtColor((target_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color
        target_display = (target_image * 255).astype(np.uint8)
    
    # Create a copy of the target image for display
    result_image = target_display.copy()
    
    # Draw original target points in blue
    for point in target_points:
        cv2.circle(result_image, totuple(point), 3, (255, 0, 0), -1)
    
    # Draw transformed source points in red
    for point in transformed_points:
        cv2.circle(result_image, totuple(point), 3, (0, 0, 255), -1)
    
    # Display the result
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(source_display, cv2.COLOR_BGR2RGB))
    plt.title("Source Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(target_display, cv2.COLOR_BGR2RGB))
    plt.title("Target Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Registration Result")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


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
    source_points, target_points = select_corresponding_points(
        source_image, target_image, 
        "Select Points on Source Image", 
        "Select Points on Target Image"
    )
    
    # Estimate rigid transformation
    T = rigid_registration(source_points, target_points)
    
    # Apply transformation to source points
    transformed_points = applyTransform(source_points, T)
    
    # Apply transformation to source image
    rows, cols = target_image.shape[:2]
    registered_image = cv2.warpAffine(source_image, T, (cols, rows))
    
    # Visualize result
    visualize_registration_result_opencv(
        source_image, target_image, 
        source_points, target_points, 
        transformed_points
    )
    
    return T, registered_image


if __name__ == "__main__":
    # Example usage
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
    from src.image_processing.image_registration import superimpose
    
    # Superimpose registered image with target image
    superimposed = superimpose(registered_image, target_gray, "rigid_manual.png")
    
    print("Registration completed successfully!")
