# Damage Modeling Module

This module provides functionality for modeling and simulating damage to images, as well as restoring damaged images.

## Overview

The damage modeling module allows you to:

1. Generate synthetic damage to images using different Point Spread Functions (PSFs)
2. Add noise to images to simulate real-world degradation
3. Restore damaged images using inverse filtering and Wiener filtering
4. Generate test patterns like checkerboards for evaluating restoration algorithms

## Functions

### Damage Generation

- `generate_gaussian_psf(size, sigma)`: Generate a Gaussian PSF
- `generate_motion_blur_psf(size, length, angle)`: Generate a motion blur PSF
- `apply_damage(image, psf, noise_level)`: Apply damage to an image using a PSF and noise

### Image Restoration

- `inverse_filter(damaged_image, psf, epsilon)`: Restore an image using inverse filtering
- `wiener_filter(damaged_image, psf, K)`: Restore an image using Wiener filtering

### Utilities

- `generate_checkerboard(size, square_size)`: Generate a checkerboard pattern
- `visualize_psf(psf, title)`: Visualize a PSF
- `load_image()`: Load a sample image for testing

## Usage Examples

### Command Line

```bash
# Apply damage to an image using a Gaussian PSF
imgproc damage --psf gaussian --sigma 3.0 --noise 0.01 --image path/to/image.jpg

# Apply damage using a motion blur PSF
imgproc damage --psf motion --length 15 --angle 45 --noise 0.01 --image path/to/image.jpg

# Restore an image using the inverse filter
imgproc restore --method inverse --psf gaussian --sigma 3.0 --image damaged.png

# Restore an image using the Wiener filter
imgproc restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image damaged.png

# Compare different restoration methods
imgproc restore --method compare --psf gaussian --sigma 3.0 --image damaged.png

# Generate a checkerboard image
imgproc checkerboard --size 8 --square_size 32 --output checkerboard.png
```

### Python API

```python
from src.image_processing.damage_modeling import (
    generate_gaussian_psf,
    apply_damage,
    wiener_filter
)
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('path/to/image.jpg', as_gray=True))

# Generate a Gaussian PSF
psf = generate_gaussian_psf(size=64, sigma=3.0)

# Apply damage to the image
damaged = apply_damage(image, psf, noise_level=0.01)

# Restore the image using Wiener filtering
restored = wiener_filter(damaged, psf, K=0.01)
```

## Theory

### Point Spread Function (PSF)

The Point Spread Function (PSF) describes how a point source of light is spread out when captured by an imaging system. In the context of image degradation, the PSF represents the blurring kernel that is convolved with the original image.

### Image Degradation Model

The degradation model can be expressed as:

g = h * f + n

Where:
- g is the degraded image
- f is the original image
- h is the PSF
- * denotes convolution
- n is additive noise

### Inverse Filtering

Inverse filtering attempts to recover the original image by dividing the Fourier transform of the degraded image by the Fourier transform of the PSF:

F = G / H

Where F, G, and H are the Fourier transforms of f, g, and h, respectively.

### Wiener Filtering

Wiener filtering improves upon inverse filtering by accounting for noise. The Wiener filter in the frequency domain is:

W = H* / (|H|² + K)

Where H* is the complex conjugate of H, |H|² is the squared magnitude of H, and K is a parameter that controls the trade-off between noise suppression and image restoration.

## Author

Oussama GUELFAA
Date: 01-04-2025
