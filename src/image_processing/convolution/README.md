# Convolution Operations

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This module provides convolution operations for image processing, including various kernel types.

## Features

- Convolution with various kernels (Gaussian, Sobel, etc.)
- Custom kernel creation
- Efficient convolution implementation

## Usage


```python
from src.image_processing.convolution import convolve, create_kernel

# Create a Gaussian kernel
kernel = create_kernel('gaussian', size=5, sigma=1.0)

# Apply convolution to an image
result = convolve(image, kernel)
```


## Examples


Example 1: Applying a Gaussian blur
```python
from src.image_processing.convolution import convolve, create_kernel
import matplotlib.pyplot as plt
from skimage import io

# Load an image
image = io.imread('data/image.jpg', as_gray=True)

# Create a Gaussian kernel
kernel = create_kernel('gaussian', size=5, sigma=1.0)

# Apply convolution
blurred = convolve(image, kernel)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(blurred, cmap='gray'), plt.title('Blurred')
plt.tight_layout()
plt.show()
```


## Functions

Error getting module functions: No module named 'src'

## References


- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
- Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.

