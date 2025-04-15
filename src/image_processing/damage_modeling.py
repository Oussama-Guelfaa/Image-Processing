#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for damage modeling and image restoration through deconvolution.

This module implements functions to:
1. Generate a checkerboard image
2. Generate a Point Spread Function (PSF)
3. Apply damage to an image using convolution and noise
4. Restore images using deconvolution algorithms

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io, img_as_float, img_as_ubyte
from skimage.color import rgb2gray
import os

def generate_checkerboard(size=8, square_size=32):
    """
    Generate a checkerboard image.
    
    Args:
        size (int): Number of squares in each dimension (default: 8)
        square_size (int): Size of each square in pixels (default: 32)
        
    Returns:
        ndarray: Checkerboard image with values 0 and 1
    """
    # Create a grid of coordinates
    x = np.arange(size * square_size)
    y = np.arange(size * square_size)
    X, Y = np.meshgrid(x, y)
    
    # Create the checkerboard pattern
    checkerboard = np.zeros((size * square_size, size * square_size))
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 1
    
    return checkerboard

def generate_gaussian_psf(size=64, sigma=3):
    """
    Generate a Gaussian Point Spread Function (PSF).
    
    Args:
        size (int): Size of the PSF in pixels (default: 64)
        sigma (float): Standard deviation of the Gaussian (default: 3)
        
    Returns:
        ndarray: Normalized Gaussian PSF
    """
    # Create a grid of coordinates
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    
    # Create the Gaussian PSF
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # Normalize the PSF so it sums to 1
    psf = psf / np.sum(psf)
    
    return psf

def generate_motion_blur_psf(size=64, length=15, angle=45):
    """
    Generate a motion blur Point Spread Function (PSF).
    
    Args:
        size (int): Size of the PSF in pixels (default: 64)
        length (int): Length of the motion blur in pixels (default: 15)
        angle (float): Angle of the motion blur in degrees (default: 45)
        
    Returns:
        ndarray: Normalized motion blur PSF
    """
    # Create a grid of coordinates
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, y)
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Create a line with the specified angle
    Z = np.zeros((size, size))
    for i in range(-length // 2, length // 2 + 1):
        x_pos = int(i * np.cos(angle_rad)) + size // 2
        y_pos = int(i * np.sin(angle_rad)) + size // 2
        if 0 <= x_pos < size and 0 <= y_pos < size:
            Z[y_pos, x_pos] = 1
    
    # Apply a small Gaussian blur to make it more realistic
    Z = signal.convolve2d(Z, generate_gaussian_psf(size=5, sigma=1), mode='same')
    
    # Normalize the PSF so it sums to 1
    Z = Z / np.sum(Z)
    
    return Z

def apply_damage(image, psf, noise_level=0.01):
    """
    Apply damage to an image using convolution with a PSF and additive noise.
    
    The damage model is: g = h * f + n
    where:
    - g is the damaged image
    - h is the PSF
    - f is the original image
    - n is the noise
    - * denotes convolution
    
    Args:
        image (ndarray): Original image (values between 0 and 1)
        psf (ndarray): Point Spread Function
        noise_level (float): Standard deviation of the Gaussian noise (default: 0.01)
        
    Returns:
        ndarray: Damaged image
    """
    # Ensure the image is in float format with values between 0 and 1
    if image.min() < 0 or image.max() > 1:
        print("Warning: Image should have values between 0 and 1. Normalizing...")
        image = (image - image.min()) / (image.max() - image.min())
    
    # Apply convolution in the spatial domain
    blurred = signal.convolve2d(image, psf, mode='same', boundary='wrap')
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, blurred.shape)
    damaged = blurred + noise
    
    # Clip values to [0, 1] range
    damaged = np.clip(damaged, 0, 1)
    
    return damaged

def apply_damage_frequency(image, psf, noise_level=0.01):
    """
    Apply damage to an image using convolution in the frequency domain and additive noise.
    
    The damage model in the frequency domain is: G = H·F + N
    where:
    - G is the Fourier transform of the damaged image
    - H is the Fourier transform of the PSF
    - F is the Fourier transform of the original image
    - N is the Fourier transform of the noise
    - · denotes element-wise multiplication
    
    Args:
        image (ndarray): Original image (values between 0 and 1)
        psf (ndarray): Point Spread Function
        noise_level (float): Standard deviation of the Gaussian noise (default: 0.01)
        
    Returns:
        ndarray: Damaged image
    """
    # Ensure the image is in float format with values between 0 and 1
    if image.min() < 0 or image.max() > 1:
        print("Warning: Image should have values between 0 and 1. Normalizing...")
        image = (image - image.min()) / (image.max() - image.min())
    
    # Pad the PSF to match the image size
    psf_padded = np.zeros_like(image)
    psf_center = psf.shape[0] // 2
    psf_start_x = image.shape[0] // 2 - psf_center
    psf_start_y = image.shape[1] // 2 - psf_center
    psf_padded[psf_start_x:psf_start_x+psf.shape[0], psf_start_y:psf_start_y+psf.shape[1]] = psf
    
    # Apply convolution in the frequency domain
    F = np.fft.fft2(image)
    H = np.fft.fft2(psf_padded)
    G = F * H
    
    # Convert back to spatial domain
    blurred = np.real(np.fft.ifft2(G))
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, blurred.shape)
    damaged = blurred + noise
    
    # Clip values to [0, 1] range
    damaged = np.clip(damaged, 0, 1)
    
    return damaged

def inverse_filter(damaged_image, psf, epsilon=1e-3):
    """
    Restore an image using the inverse filter method.
    
    The inverse filter in the frequency domain is: F = G / H
    where:
    - F is the Fourier transform of the restored image
    - G is the Fourier transform of the damaged image
    - H is the Fourier transform of the PSF
    
    A small epsilon is added to avoid division by zero.
    
    Args:
        damaged_image (ndarray): Damaged image (values between 0 and 1)
        psf (ndarray): Point Spread Function
        epsilon (float): Small value to avoid division by zero (default: 1e-3)
        
    Returns:
        ndarray: Restored image
    """
    # Pad the PSF to match the image size
    psf_padded = np.zeros_like(damaged_image)
    psf_center = psf.shape[0] // 2
    psf_start_x = damaged_image.shape[0] // 2 - psf_center
    psf_start_y = damaged_image.shape[1] // 2 - psf_center
    psf_padded[psf_start_x:psf_start_x+psf.shape[0], psf_start_y:psf_start_y+psf.shape[1]] = psf
    
    # Apply inverse filter in the frequency domain
    G = np.fft.fft2(damaged_image)
    H = np.fft.fft2(psf_padded)
    F = G / (H + epsilon)
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(F))
    
    # Clip values to [0, 1] range
    restored = np.clip(restored, 0, 1)
    
    return restored

def wiener_filter(damaged_image, psf, K=0.01):
    """
    Restore an image using the Wiener filter method.
    
    The Wiener filter in the frequency domain is: F = G · H* / (|H|² + K)
    where:
    - F is the Fourier transform of the restored image
    - G is the Fourier transform of the damaged image
    - H is the Fourier transform of the PSF
    - H* is the complex conjugate of H
    - |H|² is the squared magnitude of H
    - K is a parameter related to the noise-to-signal ratio
    
    Args:
        damaged_image (ndarray): Damaged image (values between 0 and 1)
        psf (ndarray): Point Spread Function
        K (float): Wiener filter parameter (default: 0.01)
        
    Returns:
        ndarray: Restored image
    """
    # Pad the PSF to match the image size
    psf_padded = np.zeros_like(damaged_image)
    psf_center = psf.shape[0] // 2
    psf_start_x = damaged_image.shape[0] // 2 - psf_center
    psf_start_y = damaged_image.shape[1] // 2 - psf_center
    psf_padded[psf_start_x:psf_start_x+psf.shape[0], psf_start_y:psf_start_y+psf.shape[1]] = psf
    
    # Apply Wiener filter in the frequency domain
    G = np.fft.fft2(damaged_image)
    H = np.fft.fft2(psf_padded)
    H_conj = np.conj(H)
    H_abs_squared = np.abs(H) ** 2
    F = G * H_conj / (H_abs_squared + K)
    
    # Convert back to spatial domain
    restored = np.real(np.fft.ifft2(F))
    
    # Clip values to [0, 1] range
    restored = np.clip(restored, 0, 1)
    
    return restored

def load_image(image_path=None):
    """
    Load an image from the specified path or use a default image.
    
    Args:
        image_path (str): Path to the image file (default: None)
        
    Returns:
        ndarray: Image as a float array with values between 0 and 1
    """
    if image_path is None:
        # Use a default image from the data folder
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(script_dir, 'data')
        image_path = os.path.join(data_dir, 'jupiter.jpg')
        print(f"Image chargée: jupiter.jpg")
    else:
        print(f"Image chargée: {os.path.basename(image_path)}")
    
    # Load the image
    image = io.imread(image_path)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    
    # Convert to float with values between 0 and 1
    image = img_as_float(image)
    
    # Print image information
    print(f"Dimensions: {image.shape}")
    print(f"Valeur min: {image.min():.4f}, Valeur max: {image.max():.4f}")
    
    return image

def visualize_psf(psf, title="Point Spread Function"):
    """
    Visualize a Point Spread Function.
    
    Args:
        psf (ndarray): Point Spread Function
        title (str): Title for the plot (default: "Point Spread Function")
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(psf, cmap='viridis')
    plt.title(title)
    plt.colorbar(label='Intensity')
    plt.tight_layout()
    plt.show()
    
    # Also visualize the PSF in the frequency domain
    psf_fft = np.fft.fftshift(np.abs(np.fft.fft2(psf)))
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log1p(psf_fft), cmap='viridis')
    plt.title(f"PSF in Frequency Domain")
    plt.colorbar(label='Log Magnitude')
    plt.tight_layout()
    plt.show()

def visualize_restoration_results(original, damaged, restored, titles=None):
    """
    Visualize the original, damaged, and restored images side by side.
    
    Args:
        original (ndarray): Original image
        damaged (ndarray): Damaged image
        restored (ndarray): Restored image
        titles (list): List of titles for the subplots (default: None)
    """
    if titles is None:
        titles = ["Original Image", "Damaged Image", "Restored Image"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(damaged, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    axes[2].imshow(restored, cmap='gray')
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_damage_modeling(image=None, psf_type='gaussian', noise_level=0.01):
    """
    Test the damage modeling functions on an image.
    
    Args:
        image (ndarray): Input image (default: None, will load a default image)
        psf_type (str): Type of PSF to use ('gaussian' or 'motion') (default: 'gaussian')
        noise_level (float): Noise level for the damage (default: 0.01)
    """
    # Load the image if not provided
    if image is None:
        image = load_image()
    
    # Generate the PSF
    if psf_type == 'gaussian':
        psf = generate_gaussian_psf(size=32, sigma=3)
        psf_title = "Gaussian PSF"
    else:  # motion
        psf = generate_motion_blur_psf(size=32, length=15, angle=45)
        psf_title = "Motion Blur PSF"
    
    # Visualize the PSF
    visualize_psf(psf, title=psf_title)
    
    # Apply damage to the image
    damaged = apply_damage(image, psf, noise_level=noise_level)
    
    # Visualize the original and damaged images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(damaged, cmap='gray')
    axes[1].set_title('Damaged Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    return image, psf, damaged

def test_restoration(image=None, psf_type='gaussian', noise_level=0.01, method='wiener', K=0.01):
    """
    Test the restoration functions on an image.
    
    Args:
        image (ndarray): Input image (default: None, will load a default image)
        psf_type (str): Type of PSF to use ('gaussian' or 'motion') (default: 'gaussian')
        noise_level (float): Noise level for the damage (default: 0.01)
        method (str): Restoration method ('inverse' or 'wiener') (default: 'wiener')
        K (float): Wiener filter parameter (default: 0.01)
    """
    # Test damage modeling to get the original, PSF, and damaged image
    original, psf, damaged = test_damage_modeling(image, psf_type, noise_level)
    
    # Apply restoration
    if method == 'inverse':
        restored = inverse_filter(damaged, psf, epsilon=1e-3)
        method_title = "Inverse Filter"
    else:  # wiener
        restored = wiener_filter(damaged, psf, K=K)
        method_title = f"Wiener Filter (K={K})"
    
    # Visualize the results
    visualize_restoration_results(original, damaged, restored, 
                                 titles=["Original Image", "Damaged Image", f"Restored Image ({method_title})"])
    
    return original, damaged, restored

def test_checkerboard():
    """
    Test the checkerboard generation function.
    """
    # Generate a checkerboard image
    checkerboard = generate_checkerboard(size=8, square_size=32)
    
    # Visualize the checkerboard
    plt.figure(figsize=(5, 5))
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Checkerboard Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return checkerboard

def compare_restoration_methods(image=None, psf_type='gaussian', noise_level=0.01):
    """
    Compare different restoration methods on an image.
    
    Args:
        image (ndarray): Input image (default: None, will load a default image)
        psf_type (str): Type of PSF to use ('gaussian' or 'motion') (default: 'gaussian')
        noise_level (float): Noise level for the damage (default: 0.01)
    """
    # Test damage modeling to get the original, PSF, and damaged image
    original, psf, damaged = test_damage_modeling(image, psf_type, noise_level)
    
    # Apply different restoration methods
    restored_inverse = inverse_filter(damaged, psf, epsilon=1e-3)
    restored_wiener_low = wiener_filter(damaged, psf, K=0.001)
    restored_wiener_med = wiener_filter(damaged, psf, K=0.01)
    restored_wiener_high = wiener_filter(damaged, psf, K=0.1)
    
    # Visualize the results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(damaged, cmap='gray')
    axes[0, 1].set_title('Damaged Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(restored_inverse, cmap='gray')
    axes[0, 2].set_title('Inverse Filter')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(restored_wiener_low, cmap='gray')
    axes[1, 0].set_title('Wiener Filter (K=0.001)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(restored_wiener_med, cmap='gray')
    axes[1, 1].set_title('Wiener Filter (K=0.01)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(restored_wiener_high, cmap='gray')
    axes[1, 2].set_title('Wiener Filter (K=0.1)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original, damaged, restored_inverse, restored_wiener_med

if __name__ == "__main__":
    # Test the checkerboard generation
    checkerboard = test_checkerboard()
    
    # Test damage modeling and restoration on the checkerboard
    test_restoration(checkerboard, psf_type='gaussian', noise_level=0.01, method='wiener', K=0.01)
    
    # Test damage modeling and restoration on a real image
    test_restoration(None, psf_type='motion', noise_level=0.01, method='wiener', K=0.01)
    
    # Compare different restoration methods
    compare_restoration_methods(None, psf_type='gaussian', noise_level=0.01)
