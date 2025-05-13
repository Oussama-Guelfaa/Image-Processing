# Intensity Transformations Module

This module provides functionality for applying intensity transformations to images.

## Overview

The intensity transformations module allows you to:

1. Apply gamma correction to images
2. Apply contrast stretching
3. Apply logarithmic transformations
4. Apply power-law transformations
5. Apply negative transformations

## Functions

### Gamma Correction

- `apply_gamma_correction(image, gamma)`: Apply gamma correction to an image
- `visualize_gamma_correction(image, gamma_values)`: Visualize the effect of different gamma values

### Contrast Stretching

- `apply_contrast_stretching(image, E, m)`: Apply contrast stretching to an image
- `visualize_contrast_stretching(image, E_values, m_values)`: Visualize the effect of different E and m values

### Other Transformations

- `apply_log_transform(image, c)`: Apply logarithmic transformation
- `apply_power_law_transform(image, c, gamma)`: Apply power-law transformation
- `apply_negative_transform(image)`: Apply negative transformation

## Usage Examples

### Command Line

```bash
# Apply gamma correction to an image
imgproc intensity --method gamma --gamma 0.5 --image path/to/image.jpg

# Apply contrast stretching to an image
imgproc intensity --method contrast --E 4.0 --m 0.5 --image path/to/image.jpg

# Apply both transformations to an image
imgproc intensity --method both --gamma 0.5 --E 4.0 --m 0.5 --image path/to/image.jpg
```

### Python API

```python
from src.image_processing.transformations import apply_gamma_correction, apply_contrast_stretching
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('path/to/image.jpg', as_gray=True))

# Apply gamma correction
gamma_corrected = apply_gamma_correction(image, gamma=0.5)

# Apply contrast stretching
contrast_stretched = apply_contrast_stretching(image, E=4.0, m=0.5)
```

## Theory

### Gamma Correction

Gamma correction is a nonlinear operation used to encode and decode luminance values in images. The general form of gamma correction is:

s = c * r^γ

Where:
- r is the input pixel value (normalized to [0, 1])
- s is the output pixel value
- c is a constant (usually 1)
- γ (gamma) is the gamma value

Gamma values less than 1 make the image lighter, while gamma values greater than 1 make the image darker.

### Contrast Stretching

Contrast stretching (also called normalization) is a simple image enhancement technique that attempts to improve the contrast in an image by stretching the range of intensity values. The general form of contrast stretching is:

s = 1 / (1 + (m / r)^E)

Where:
- r is the input pixel value (normalized to [0, 1])
- s is the output pixel value
- m is the midpoint value (where s = 0.5)
- E controls the slope of the transformation

## Author

Oussama GUELFAA
Date: 01-04-2025
