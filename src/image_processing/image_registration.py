#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for image registration using Iterative Closest Point (ICP) method.

This module implements functions for rigid transformation estimation (rotation + translation)
between two images based on corresponding control points.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial import cKDTree
import cv2
from skimage import io, color, feature


def estimate_rigid_transform(source_points, target_points):
    """
    Estimate rigid transformation (rotation + translation) between two sets of points.

    Args:
        source_points (ndarray): Array of source points with shape (N, 2)
        target_points (ndarray): Array of target points with shape (N, 2)

    Returns:
        tuple: (R, t) where R is the rotation matrix and t is the translation vector
    """
    # 1. Compute the center of each set of points
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)

    # 2. Subtract centers to get centered coordinates
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # 3. Compute the matrix K = target_centered.T @ source_centered
    K = target_centered.T @ source_centered

    # 4. Use SVD decomposition
    U, _, Vt = linalg.svd(K)

    # 5. Evaluate the rotation matrix
    # Ensure proper rotation matrix (right-handed coordinate system)
    S = np.eye(2)
    S[1, 1] = np.linalg.det(U) * np.linalg.det(Vt.T)

    R = U @ S @ Vt

    # 6. Evaluate the translation
    t = target_center - R @ source_center

    return R, t


def apply_rigid_transform(image, R, t):
    """
    Apply rigid transformation to an image.

    Args:
        image (ndarray): Input image
        R (ndarray): 2x2 rotation matrix
        t (ndarray): Translation vector

    Returns:
        ndarray: Transformed image
    """
    # Create affine transformation matrix
    M = np.zeros((2, 3))
    M[:2, :2] = R
    M[:, 2] = t

    # Apply transformation
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, M, (cols, rows))

    return transformed_image


def transform_points(points, R, t):
    """
    Apply rigid transformation to a set of points.

    Args:
        points (ndarray): Array of points with shape (N, 2)
        R (ndarray): 2x2 rotation matrix
        t (ndarray): Translation vector

    Returns:
        ndarray: Transformed points
    """
    # Convert to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Create transformation matrix
    M = np.zeros((3, 3))
    M[:2, :2] = R
    M[:2, 2] = t
    M[2, 2] = 1

    # Apply transformation
    transformed_points = points_homogeneous @ M.T

    return transformed_points[:, :2]


def find_nearest_neighbors(source_points, target_points):
    """
    Find nearest neighbors in target_points for each point in source_points.

    Args:
        source_points (ndarray): Array of source points with shape (N, 2)
        target_points (ndarray): Array of target points with shape (M, 2)

    Returns:
        tuple: (matched_source, matched_target) arrays of matched points
    """
    # Build KD-Tree for target points
    tree = cKDTree(target_points)

    # Find nearest neighbors
    distances, indices = tree.query(source_points)

    # Get matched points
    matched_target = target_points[indices]

    return source_points, matched_target


def icp_registration(source_points, target_points, max_iterations=20, tolerance=1e-6):
    """
    Perform Iterative Closest Point (ICP) registration.

    Args:
        source_points (ndarray): Array of source points with shape (N, 2)
        target_points (ndarray): Array of target points with shape (M, 2)
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance

    Returns:
        tuple: (R, t, transformed_points, error) where R is the rotation matrix,
               t is the translation vector, transformed_points are the final
               transformed source points, and error is the final mean squared error
    """
    # Initialize
    current_points = source_points.copy()
    prev_error = float('inf')

    for i in range(max_iterations):
        # 1. Find nearest neighbors
        matched_source, matched_target = find_nearest_neighbors(current_points, target_points)

        # 2. Estimate transformation
        R, t = estimate_rigid_transform(matched_source, matched_target)

        # 3. Apply transformation
        current_points = transform_points(current_points, R, t)

        # 4. Compute error
        error = np.mean(np.sum((matched_target - current_points) ** 2, axis=1))

        # 5. Check convergence
        if abs(prev_error - error) < tolerance:
            break

        prev_error = error

    # Compute final transformation (from original source to final position)
    final_R, final_t = estimate_rigid_transform(source_points, current_points)

    return final_R, final_t, current_points, error


def detect_corners(image, max_corners=50, quality_level=0.01, min_distance=10):
    """
    Detect corners in an image using Shi-Tomasi corner detector.

    Args:
        image (ndarray): Input image
        max_corners (int): Maximum number of corners to detect
        quality_level (float): Quality level parameter
        min_distance (float): Minimum distance between corners

    Returns:
        ndarray: Array of corner coordinates with shape (N, 2)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    # Ensure image is in the right format
    gray = (gray * 255).astype(np.uint8)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    # Reshape to (N, 2)
    if corners is not None:
        corners = corners.reshape(-1, 2)
    else:
        corners = np.array([])

    return corners


def visualize_points(image, points, title="Image with Points", color='red', marker='o'):
    """
    Visualize points on an image.

    Args:
        image (ndarray): Input image
        points (ndarray): Array of points with shape (N, 2)
        title (str): Title for the plot
        color (str): Color for the points
        marker (str): Marker style for the points
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.scatter(points[:, 0], points[:, 1], c=color, marker=marker, s=50)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_point_pairs(image1, points1, image2, points2, title="Point Pairs"):
    """
    Visualize corresponding points between two images.

    Args:
        image1 (ndarray): First image
        points1 (ndarray): Points in first image with shape (N, 2)
        image2 (ndarray): Second image
        points2 (ndarray): Points in second image with shape (N, 2)
        title (str): Title for the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Plot first image with points
    axes[0].imshow(image1, cmap='gray')
    axes[0].scatter(points1[:, 0], points1[:, 1], c='red', marker='o', s=50)
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    # Plot second image with points
    axes[1].imshow(image2, cmap='gray')
    axes[1].scatter(points2[:, 0], points2[:, 1], c='blue', marker='o', s=50)
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    # Add lines connecting corresponding points
    for i in range(len(points1)):
        x = [points1[i, 0], points2[i, 0]]
        y = [points1[i, 1], points2[i, 1]]
        plt.plot(x, y, 'g-', alpha=0.5)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def superimpose(G1, G2, filename=None):
    """
    Superimpose 2 images, supposing they are grayscale images and of same shape.
    For display purposes.

    Args:
        G1 (ndarray): First grayscale image
        G2 (ndarray): Second grayscale image
        filename (str): Path to save the superimposed image (default: None)

    Returns:
        ndarray: Superimposed image as RGB
    """
    r, c = G1.shape
    S = np.zeros((r, c, 3))

    S[:,:,0] = np.maximum(G1-G2, 0) + G1
    S[:,:,1] = np.maximum(G2-G1, 0) + G2
    S[:,:,2] = (G1+G2) / 2

    S = 255 * S / np.max(S)
    S = S.astype('uint8')

    plt.figure(figsize=(10, 8))
    plt.imshow(S)
    plt.title("Superimposed Images")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if filename is not None:
        cv2.imwrite(filename, cv2.cvtColor(S, cv2.COLOR_RGB2BGR))

    return S


def visualize_registration_result(source_image, target_image, registered_image, title="Registration Result", save_superimposed=False, superimposed_filename=None):
    """
    Visualize registration result.

    Args:
        source_image (ndarray): Source image
        target_image (ndarray): Target image
        registered_image (ndarray): Registered image
        title (str): Title for the plot
        save_superimposed (bool): Whether to save the superimposed image
        superimposed_filename (str): Path to save the superimposed image
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(source_image, cmap='gray')
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(target_image, cmap='gray')
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    axes[2].imshow(registered_image, cmap='gray')
    axes[2].set_title("Registered Image")
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Create and display superimposed image
    if save_superimposed:
        superimpose(registered_image, target_image, superimposed_filename)


def manual_point_selection(image, n_points=4, title="Select Points"):
    """
    Manually select points on an image.

    Args:
        image (ndarray): Input image
        n_points (int): Number of points to select
        title (str): Title for the plot

    Returns:
        ndarray: Array of selected points with shape (N, 2)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title(f"{title} - Click to select {n_points} points")
    plt.axis('off')

    points = plt.ginput(n_points, timeout=0)
    plt.close()

    return np.array(points)


def test_image_registration(source_image_path, target_image_path, use_manual_points=True,
                           use_icp=False, max_iterations=20, visualize=True):
    """
    Test image registration on a pair of images.

    Args:
        source_image_path (str): Path to source image
        target_image_path (str): Path to target image
        use_manual_points (bool): Whether to use manual point selection
        use_icp (bool): Whether to use ICP for registration
        max_iterations (int): Maximum number of iterations for ICP
        visualize (bool): Whether to visualize results

    Returns:
        tuple: (R, t, registered_image) where R is the rotation matrix,
               t is the translation vector, and registered_image is the
               registered source image
    """
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

    # Get control points
    if use_manual_points:
        print("Select points on the source image:")
        source_points = manual_point_selection(source_gray, n_points=4, title="Source Image")

        print("Select corresponding points on the target image:")
        target_points = manual_point_selection(target_gray, n_points=4, title="Target Image")
    else:
        # Detect corners automatically
        source_points = detect_corners(source_gray, max_corners=20)
        target_points = detect_corners(target_gray, max_corners=20)

        # Ensure same number of points
        min_points = min(len(source_points), len(target_points))
        source_points = source_points[:min_points]
        target_points = target_points[:min_points]

    # Visualize selected points
    if visualize:
        visualize_point_pairs(source_gray, source_points, target_gray, target_points,
                             title="Selected Control Points")

    # Perform registration
    if use_icp:
        # Shuffle source points to simulate incorrect pairing
        if use_manual_points:
            np.random.shuffle(source_points)

        # Apply ICP
        R, t, transformed_points, error = icp_registration(
            source_points, target_points, max_iterations=max_iterations
        )
        print(f"ICP completed with error: {error:.6f}")
    else:
        # Direct rigid transformation estimation
        R, t = estimate_rigid_transform(source_points, target_points)

    # Apply transformation to the source image
    registered_image = apply_rigid_transform(source_gray, R, t)

    # Visualize result
    if visualize:
        visualize_registration_result(source_gray, target_gray, registered_image,
                                    title="Image Registration Result")

    return R, t, registered_image


if __name__ == "__main__":
    # Example usage
    source_image_path = "data/Brain1.bmp"
    target_image_path = "data/Brain2.bmp"

    # Define control points (as given in the tutorial)
    A_points = np.array([[136, 100], [127, 153], [96, 156], [87, 99]])
    B_points = np.array([[144, 99], [109, 140], [79, 128], [100, 74]])

    # Test with predefined points
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

    # Estimate transformation
    R, t = estimate_rigid_transform(A_points, B_points)
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(t)

    # Apply transformation
    registered_image = apply_rigid_transform(source_gray, R, t)

    # Visualize result
    visualize_registration_result(source_gray, target_gray, registered_image,
                                 title="Image Registration with Predefined Points")

    # Test with ICP
    print("\nTesting ICP registration...")
    R_icp, t_icp, transformed_points, error = icp_registration(A_points, B_points)
    print(f"ICP completed with error: {error:.6f}")
    print("ICP Rotation matrix:")
    print(R_icp)
    print("ICP Translation vector:")
    print(t_icp)

    # Apply ICP transformation
    registered_image_icp = apply_rigid_transform(source_gray, R_icp, t_icp)

    # Visualize ICP result
    visualize_registration_result(source_gray, target_gray, registered_image_icp,
                                title="Image Registration with ICP")

    # Test with automatic corner detection
    print("\nTesting registration with automatic corner detection...")
    source_corners = detect_corners(source_gray, max_corners=10)
    target_corners = detect_corners(target_gray, max_corners=10)

    # Ensure same number of points
    min_corners = min(len(source_corners), len(target_corners))
    source_corners = source_corners[:min_corners]
    target_corners = target_corners[:min_corners]

    # Visualize detected corners
    visualize_point_pairs(source_gray, source_corners, target_gray, target_corners,
                         title="Automatically Detected Corners")

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
