#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising

Techniques for removing noise from images, including various filtering methods.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
import os

def generate_uniform_noise(shape, a=0, b=1):
    """
    Generate uniform noise with values in range [a, b].

    The uniform noise is defined as: R = a + (b - a) * U(0, 1)
    where U(0, 1) is a uniform random variable in [0, 1].

    Args:
        shape (tuple): Shape of the noise array to generate
        a (float): Lower bound of the uniform distribution (default: 0)
        b (float): Upper bound of the uniform distribution (default: 1)

    Returns:
        ndarray: Uniform noise with values in range [a, b]
    """
    # Generate uniform random values in [0, 1]
    U = np.random.uniform(0, 1, shape)

    # Scale and shift to get values in [a, b]
    R = a + (b - a) * U

    return R

def generate_gaussian_noise(shape, mean=0, std=1):
    """
    Generate Gaussian noise with specified mean and standard deviation.

    The Gaussian noise is defined as: R = a + b * N(0, 1)
    where N(0, 1) is a standard normal random variable.

    Args:
        shape (tuple): Shape of the noise array to generate
        mean (float): Mean of the Gaussian distribution (default: 0)
        std (float): Standard deviation of the Gaussian distribution (default: 1)

    Returns:
        ndarray: Gaussian noise with specified mean and standard deviation
    """
    # Generate standard normal random values
    N = np.random.normal(0, 1, shape)

    # Scale and shift to get the desired mean and standard deviation
    R = mean + std * N

    return R

def generate_salt_pepper_noise(shape, a=0.3, b=0.7, p=0.1):
    """
    Generate salt and pepper noise.

    The salt and pepper noise is defined as:
    R = 0     if 0 ≤ U(0, 1) ≤ a
    R = 0.5   if a < U(0, 1) ≤ b
    R = 1     if b < U(0, 1) ≤ 1

    Args:
        shape (tuple): Shape of the noise array to generate
        a (float): First threshold (default: 0.3)
        b (float): Second threshold (default: 0.7)
        p (float): Probability of noise (default: 0.1)

    Returns:
        ndarray: Salt and pepper noise
    """
    # Generate uniform random values in [0, 1]
    U = np.random.uniform(0, 1, shape)

    # Initialize the result array with zeros
    R = np.zeros(shape)

    # Create a mask for pixels that will be affected by noise
    noise_mask = np.random.uniform(0, 1, shape) < p

    # Apply the salt and pepper noise definition only to affected pixels
    U_noise = U[noise_mask]

    # Initialize with mid-gray
    R[noise_mask] = 0.5

    # Apply pepper (black) to pixels where U <= a
    pepper_mask = noise_mask & (U <= a)
    R[pepper_mask] = 0

    # Apply salt (white) to pixels where U > b
    salt_mask = noise_mask & (U > b)
    R[salt_mask] = 1

    return R

def generate_exponential_noise(shape, a=1):
    """
    Generate exponential noise.

    The exponential noise is defined as: R = -1/a * ln(1 - U(0, 1))
    where U(0, 1) is a uniform random variable in [0, 1].

    Args:
        shape (tuple): Shape of the noise array to generate
        a (float): Scale parameter of the exponential distribution (default: 1)

    Returns:
        ndarray: Exponential noise
    """
    # Generate uniform random values in [0, 1)
    # Note: We use 0.9999 instead of 1 to avoid log(0)
    U = np.random.uniform(0, 0.9999, shape)

    # Apply the exponential noise definition
    R = -1/a * np.log(1 - U)

    return R

def add_noise_to_image(image, noise_type, **kwargs):
    """
    Add noise to an image.

    Args:
        image (ndarray): Input image (values between 0 and 1)
        noise_type (str): Type of noise to add ('uniform', 'gaussian', 'salt_pepper', 'exponential')
        **kwargs: Additional parameters for the noise generation function

    Returns:
        ndarray: Noisy image (clipped to [0, 1] range)
    """
    # Ensure the image is in float format with values between 0 and 1
    if image.min() < 0 or image.max() > 1:
        print("Warning: Image should have values between 0 and 1. Normalizing...")
        image = (image - image.min()) / (image.max() - image.min())

    # Generate noise based on the specified type
    if noise_type == 'uniform':
        noise = generate_uniform_noise(image.shape, **kwargs)
    elif noise_type == 'gaussian':
        noise = generate_gaussian_noise(image.shape, **kwargs)
    elif noise_type == 'salt_pepper':
        noise = generate_salt_pepper_noise(image.shape, **kwargs)
    elif noise_type == 'exponential':
        noise = generate_exponential_noise(image.shape, **kwargs)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Add the noise to the image
    noisy_image = image + noise

    # Clip values to [0, 1] range
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image

def visualize_noise_samples(shape=(256, 256), save_path=None):
    """
    Generate and visualize samples of different types of noise.

    Args:
        shape (tuple): Shape of the noise samples to generate (default: (256, 256))
        save_path (str): Path to save the visualization (default: None)
    """
    # Generate noise samples
    uniform_noise = generate_uniform_noise(shape, a=-0.5, b=0.5)
    gaussian_noise = generate_gaussian_noise(shape, mean=0, std=0.1)
    salt_pepper_noise = generate_salt_pepper_noise(shape, a=0.3, b=0.7)
    exponential_noise = generate_exponential_noise(shape, a=5)

    # Create histograms
    bins = 256
    uniform_hist, _ = np.histogram(uniform_noise, bins=bins, range=(-0.5, 0.5), density=True)
    gaussian_hist, _ = np.histogram(gaussian_noise, bins=bins, range=(-0.3, 0.3), density=True)
    salt_pepper_hist, _ = np.histogram(salt_pepper_noise, bins=bins, range=(0, 1), density=True)
    exponential_hist, _ = np.histogram(exponential_noise, bins=bins, range=(0, 1), density=True)

    # Visualize the noise samples and their histograms
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    # Uniform noise
    axes[0, 0].imshow(uniform_noise, cmap='gray')
    axes[0, 0].set_title('Uniform Noise')
    axes[0, 0].axis('off')

    axes[0, 1].bar(np.linspace(-0.5, 0.5, bins), uniform_hist, width=1/bins)
    axes[0, 1].set_title('Uniform Noise Histogram')
    axes[0, 1].set_xlim(-0.5, 0.5)

    # Gaussian noise
    axes[1, 0].imshow(gaussian_noise, cmap='gray')
    axes[1, 0].set_title('Gaussian Noise')
    axes[1, 0].axis('off')

    axes[1, 1].bar(np.linspace(-0.3, 0.3, bins), gaussian_hist, width=0.6/bins)
    axes[1, 1].set_title('Gaussian Noise Histogram')
    axes[1, 1].set_xlim(-0.3, 0.3)

    # Salt and pepper noise
    axes[2, 0].imshow(salt_pepper_noise, cmap='gray')
    axes[2, 0].set_title('Salt and Pepper Noise')
    axes[2, 0].axis('off')

    axes[2, 1].bar(np.linspace(0, 1, bins), salt_pepper_hist, width=1/bins)
    axes[2, 1].set_title('Salt and Pepper Noise Histogram')
    axes[2, 1].set_xlim(0, 1)

    # Exponential noise
    axes[3, 0].imshow(exponential_noise, cmap='gray')
    axes[3, 0].set_title('Exponential Noise')
    axes[3, 0].axis('off')

    axes[3, 1].bar(np.linspace(0, 1, bins), exponential_hist, width=1/bins)
    axes[3, 1].set_title('Exponential Noise Histogram')
    axes[3, 1].set_xlim(0, 1)

    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Noise samples visualization saved to: {save_path}")

    plt.show()

def load_image(image_path=None):
    """
    Load an image from the specified path or use a default image.

    Args:
        image_path (str): Path to the image file (default: None)

    Returns:
        ndarray: Loaded image as float array with values in [0, 1]
    """
    if image_path is None:
        # Use the default image (jambe.tif)
        image_path = os.path.join("data", "jambe.tif")

    # Load the image
    image = io.imread(image_path)

    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Convert to float with values in [0, 1]
    image = img_as_float(image)

    return image

def test_noise_generation(image_path=None):
    """
    Test the noise generation functions on an image.

    Args:
        image_path (str): Path to the image file (default: None)
    """
    # Load the image
    image = load_image(image_path)

    # Generate noisy images
    noisy_uniform = add_noise_to_image(image, 'uniform', a=-0.2, b=0.2)
    noisy_gaussian = add_noise_to_image(image, 'gaussian', mean=0, std=0.1)
    noisy_salt_pepper = add_noise_to_image(image, 'salt_pepper', a=0.01, b=0.99)
    noisy_exponential = add_noise_to_image(image, 'exponential', a=10)

    # Visualize the results
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Uniform noise
    axes[0, 1].imshow(noisy_uniform, cmap='gray')
    axes[0, 1].set_title('Uniform Noise')
    axes[0, 1].axis('off')

    # Gaussian noise
    axes[1, 0].imshow(noisy_gaussian, cmap='gray')
    axes[1, 0].set_title('Gaussian Noise')
    axes[1, 0].axis('off')

    # Salt and pepper noise
    axes[1, 1].imshow(noisy_salt_pepper, cmap='gray')
    axes[1, 1].set_title('Salt and Pepper Noise')
    axes[1, 1].axis('off')

    # Exponential noise
    axes[2, 0].imshow(noisy_exponential, cmap='gray')
    axes[2, 0].set_title('Exponential Noise')
    axes[2, 0].axis('off')

    plt.tight_layout()
    plt.show()

    return image, noisy_uniform, noisy_gaussian, noisy_salt_pepper, noisy_exponential

if __name__ == "__main__":
    # Visualize noise samples
    visualize_noise_samples(save_path="output/noise_samples.png")

    # Test noise generation on an image
    test_noise_generation(image_path="data/jambe.tif")
