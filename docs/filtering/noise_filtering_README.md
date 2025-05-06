# Image Noise Filtering Documentation

## Overview

This document provides a comprehensive explanation of the image noise filtering techniques implemented in this project, with a particular focus on salt and pepper noise removal. The implementation demonstrates various filtering methods and their effectiveness in removing impulse noise while preserving image structures.

## Table of Contents

1. [Introduction to Image Noise](#introduction-to-image-noise)
2. [Types of Noise](#types-of-noise)
3. [Filtering Techniques](#filtering-techniques)
   - [Order Statistic Filters](#order-statistic-filters)
   - [Mean Filter](#mean-filter)
   - [Median Filter](#median-filter)
   - [Maximum Filter](#maximum-filter)
   - [Minimum Filter](#minimum-filter)
   - [Adaptive Median Filter](#adaptive-median-filter)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Results and Analysis](#results-and-analysis)
7. [References](#references)

## Introduction to Image Noise

Image noise is random variation of brightness or color information in images. It is an undesirable by-product of image capture that obscures the desired information. Noise can be introduced during image acquisition, transmission, or processing and can significantly degrade image quality.

In digital image processing, noise reduction is a critical preprocessing step that aims to remove noise while preserving important image features such as edges and textures.

## Types of Noise

Several types of noise can affect digital images:

1. **Gaussian Noise**: Follows a normal distribution and affects each pixel independently.
2. **Salt and Pepper Noise (Impulse Noise)**: Appears as randomly occurring white and black pixels.
3. **Speckle Noise**: Multiplicative noise that occurs in coherent imaging systems.
4. **Poisson Noise**: Occurs due to the statistical nature of electromagnetic waves.
5. **Quantization Noise**: Occurs during the quantization process in analog-to-digital conversion.

This documentation focuses primarily on salt and pepper noise and the filtering techniques effective for its removal.

### Salt and Pepper Noise

Salt and pepper noise (also known as impulse noise) is characterized by sparse light (salt) and dark (pepper) pixels that appear randomly in the image. This type of noise can be caused by:

- Errors in the image acquisition process
- Faulty memory locations in hardware
- Transmission errors
- Malfunctioning pixel elements in camera sensors

Salt and pepper noise is particularly challenging because it can significantly alter the affected pixels' values, making them very different from their neighbors.

## Filtering Techniques

### Order Statistic Filters

Order statistic filters are non-linear spatial filters that operate on the local neighborhood of a pixel. They work by ordering (ranking) the pixel values in the neighborhood and selecting a value based on its rank.

#### Mean Filter

The mean filter replaces each pixel with the average value of its neighborhood:

```python
def apply_mean_filter(image, kernel_size=3):
    """
    Apply a mean filter to an image.
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)
        
    Returns:
        ndarray: Denoised image
    """
    # Convert to uint8 for rank filters
    image_uint8 = img_as_ubyte(image)
    
    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)
    
    # Apply mean filter
    filtered_uint8 = mean(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)
    
    return filtered
```

**Characteristics**:
- Reduces random noise effectively
- Blurs edges and fine details
- Not very effective for salt and pepper noise
- Simple to implement and computationally efficient

#### Median Filter

The median filter replaces each pixel with the median value of its neighborhood:

```python
def apply_median_filter(image, kernel_size=3):
    """
    Apply a median filter to an image.
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)
        
    Returns:
        ndarray: Denoised image
    """
    # Convert to uint8 for rank filters
    image_uint8 = img_as_ubyte(image)
    
    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)
    
    # Apply median filter
    filtered_uint8 = median(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)
    
    return filtered
```

**Characteristics**:
- Very effective for salt and pepper noise
- Preserves edges better than the mean filter
- Less blurring of image details
- More computationally intensive than the mean filter

#### Maximum Filter

The maximum filter replaces each pixel with the maximum value in its neighborhood:

```python
def apply_max_filter(image, kernel_size=3):
    """
    Apply a maximum filter to an image.
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)
        
    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    image_uint8 = img_as_ubyte(image)
    
    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)
    
    # Apply maximum filter
    filtered_uint8 = maximum(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)
    
    return filtered
```

**Characteristics**:
- Effective for removing 'pepper' noise (dark pixels)
- Brightens the image and expands light regions
- Equivalent to morphological dilation
- Can distort image features significantly

#### Minimum Filter

The minimum filter replaces each pixel with the minimum value in its neighborhood:

```python
def apply_min_filter(image, kernel_size=3):
    """
    Apply a minimum filter to an image.
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the filter kernel (default: 3)
        
    Returns:
        ndarray: Filtered image
    """
    # Convert to uint8 for rank filters
    image_uint8 = img_as_ubyte(image)
    
    # Create a disk-shaped structuring element
    selem = disk(kernel_size // 2)
    
    # Apply minimum filter
    filtered_uint8 = minimum(image_uint8, selem)
    filtered = img_as_float(filtered_uint8)
    
    return filtered
```

**Characteristics**:
- Effective for removing 'salt' noise (bright pixels)
- Darkens the image and expands dark regions
- Equivalent to morphological erosion
- Can distort image features significantly

### Adaptive Median Filter

The adaptive median filter is an advanced version of the median filter that adapts its window size based on local image characteristics:

```python
def adaptive_median_filter(image, max_window_size=7):
    """
    Apply adaptive median filter to an image.
    
    The adaptive median filter algorithm works as follows:
    1. For each pixel, start with a small window size (3x3)
    2. Calculate the median, min, and max values in the window
    3. If the median is not between min and max, increase the window size
    4. If the pixel value is not an impulse noise, keep it unchanged
    5. Otherwise, replace it with the median value
    
    Args:
        image (ndarray): Input image (grayscale)
        max_window_size (int): Maximum window size (default: 7)
        
    Returns:
        ndarray: Filtered image
    """
    # Implementation details...
```

**Characteristics**:
- Highly effective for salt and pepper noise
- Preserves edges and details better than standard median filter
- Adapts to local image content
- More computationally intensive
- Particularly useful for high noise densities

## Implementation Details

The implementation of these filters in our project uses the scikit-image library for most of the core functionality, with custom implementations for specialized filters like the adaptive median filter.

The key components include:

1. **Noise Generation**: Functions to add various types of noise to test images
2. **Filter Application**: Functions to apply different filters to noisy images
3. **Visualization**: Tools to display and compare the results of different filtering techniques

## Usage Examples

### Basic Usage

```python
from skimage import io, img_as_float
from skimage.util import random_noise
from src.image_processing.denoising import apply_median_filter

# Load an image
image = img_as_float(io.imread('data/jambe.png', as_gray=True))

# Add salt and pepper noise
noisy_image = random_noise(image, mode='s&p', amount=0.1, salt_vs_pepper=0.5)

# Apply median filter
filtered_image = apply_median_filter(noisy_image, kernel_size=5)

# Display results
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[2].imshow(filtered_image, cmap='gray')
axes[2].set_title('Filtered Image')
plt.show()
```

### Comparing Multiple Filters

The `filter_comparison.py` script demonstrates how to compare different filtering techniques on a noisy image:

```python
# Load the image
image_path = os.path.join("data", "jambe.png")
image = img_as_float(io.imread(image_path, as_gray=True))

# Add salt and pepper noise
noise_level = 0.1  # 10% of pixels affected
noisy_image = random_noise(image, mode='s&p', amount=noise_level, salt_vs_pepper=0.5)

# Apply different filters
kernel_size = 5
max_filtered = apply_max_filter(noisy_image, kernel_size)
min_filtered = apply_min_filter(noisy_image, kernel_size)
mean_filtered = apply_mean_filter(noisy_image, kernel_size)
median_filtered = apply_median_filter(noisy_image, kernel_size)
```

## Results and Analysis

The effectiveness of different filters for salt and pepper noise removal can be summarized as follows:

1. **Mean Filter**: Reduces noise but blurs edges significantly. Not ideal for salt and pepper noise.
2. **Median Filter**: Excellent for salt and pepper noise removal while preserving edges.
3. **Maximum Filter**: Good for removing pepper noise but amplifies salt noise.
4. **Minimum Filter**: Good for removing salt noise but amplifies pepper noise.
5. **Adaptive Median Filter**: Best overall performance, especially for high noise densities.

The figure below illustrates the performance of these filters on an image with salt and pepper noise:

![Filter Comparison](../../output/filter_comparison.png)

*Figure: Different filters applied to a noisy image. The median filter is particularly adapted in the case of salt and pepper noise (impulse noise), but still may destroy some structures observed in the images.*

## References

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
2. Hwang, H., & Haddad, R. A. (1995). Adaptive median filters: new algorithms and results. IEEE Transactions on Image Processing, 4(4), 499-502.
3. Scikit-image: Image processing in Python. (n.d.). Retrieved from https://scikit-image.org/
