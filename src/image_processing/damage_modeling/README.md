# Damage Modeling and Restoration

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This module provides tools for modeling damage to images and restoring them using various techniques.

## Features

- Damage modeling with various point spread functions (PSF)
- Gaussian and motion blur PSF generation
- Image restoration using inverse and Wiener filters

## Usage


```python
from src.image_processing.damage_modeling import generate_gaussian_psf, apply_damage, wiener_filter

# Generate a Gaussian PSF
psf = generate_gaussian_psf(shape=(32, 32), sigma=3.0)

# Apply damage to an image
damaged = apply_damage(image, psf, noise_level=0.01)

# Restore the image using the Wiener filter
restored = wiener_filter(damaged, psf, k=0.01)
```


## Examples


Example 1: Damage modeling and restoration
```python
from src.image_processing.damage_modeling import generate_gaussian_psf, apply_damage, wiener_filter
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('data/image.jpg', as_gray=True))

# Generate a Gaussian PSF
psf = generate_gaussian_psf(shape=(32, 32), sigma=3.0)

# Apply damage to the image
damaged = apply_damage(image, psf, noise_level=0.01)

# Restore the image using the Wiener filter
restored = wiener_filter(damaged, psf, k=0.01)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(damaged, cmap='gray'), plt.title('Damaged')
plt.subplot(133), plt.imshow(restored, cmap='gray'), plt.title('Restored')
plt.tight_layout()
plt.show()
```


## Functions

Error getting module functions: No module named 'src'

## References


- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
- Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press.

