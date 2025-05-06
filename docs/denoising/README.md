# Image Denoising Documentation

This document provides a comprehensive overview of the image denoising module implemented in this project. The module includes functions for generating different types of noise, adding them to images, and applying various denoising techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Noise Generation](#noise-generation)
   - [Uniform Noise](#uniform-noise)
   - [Gaussian Noise](#gaussian-noise)
   - [Salt and Pepper Noise](#salt-and-pepper-noise)
   - [Exponential Noise](#exponential-noise)
3. [Noise Estimation](#noise-estimation)
   - [Region of Interest (ROI) Selection](#region-of-interest-roi-selection)
   - [Histogram Analysis](#histogram-analysis)
   - [Parameter Estimation](#parameter-estimation)
4. [Denoising Techniques](#denoising-techniques)
   - [Mean Filter](#mean-filter)
   - [Median Filter](#median-filter)
   - [Gaussian Filter](#gaussian-filter)
   - [Bilateral Filter](#bilateral-filter)
   - [Non-Local Means Filter](#non-local-means-filter)
5. [Usage Examples](#usage-examples)
   - [Command-Line Interface](#command-line-interface)
   - [Python API](#python-api)
6. [Results and Comparison](#results-and-comparison)

## Introduction

Image denoising is a fundamental problem in image processing that aims to remove noise from an image while preserving important features such as edges and textures. This module provides a comprehensive set of tools for generating different types of noise, estimating noise parameters, and applying various denoising techniques.

## Noise Generation

The module includes functions for generating four types of random noise:

### Uniform Noise

Uniform noise is characterized by a uniform distribution of pixel values within a specified range [a, b]. The mathematical definition is:

```
R = a + (b - a) * U(0, 1)
```

where U(0, 1) is a uniform random variable in [0, 1].

### Gaussian Noise

Gaussian noise follows a normal distribution with a specified mean and standard deviation. The mathematical definition is:

```
R = mean + std * N(0, 1)
```

where N(0, 1) is a standard normal random variable.

### Salt and Pepper Noise

Salt and pepper noise is characterized by random occurrences of black and white pixels. The mathematical definition is:

```
R = 0     if 0 ≤ U(0, 1) ≤ a
R = 0.5   if a < U(0, 1) ≤ b
R = 1     if b < U(0, 1) ≤ 1
```

where U(0, 1) is a uniform random variable in [0, 1], and a and b are thresholds.

### Exponential Noise

Exponential noise follows an exponential distribution with a specified scale parameter. The mathematical definition is:

```
R = -1/a * ln(1 - U(0, 1))
```

where U(0, 1) is a uniform random variable in [0, 1], and a is the scale parameter.

## Noise Estimation

The module includes functions for estimating noise parameters from a Region of Interest (ROI) in an image:

### Region of Interest (ROI) Selection

The module provides functions for selecting a Region of Interest (ROI) from an image, either interactively or programmatically. The ROI should be a region with uniform intensity, which makes it easier to estimate noise parameters.

### Histogram Analysis

The module includes functions for visualizing the histogram of a Region of Interest (ROI) and analyzing its statistical properties.

### Parameter Estimation

The module provides functions for estimating noise parameters from a Region of Interest (ROI) based on the type of noise:

- For Gaussian noise, the mean and standard deviation are estimated.
- For uniform noise, the lower and upper bounds are estimated.
- For salt and pepper noise, the thresholds are estimated.
- For exponential noise, the scale parameter is estimated.

## Denoising Techniques

The module includes implementations of five denoising techniques:

### Mean Filter

The mean filter replaces each pixel with the average of its neighborhood. It is effective for removing Gaussian noise but tends to blur edges.

### Median Filter

The median filter replaces each pixel with the median of its neighborhood. It is effective for removing salt and pepper noise while preserving edges better than the mean filter.

### Gaussian Filter

The Gaussian filter applies a weighted average of the neighborhood, with weights following a Gaussian distribution. It is effective for removing Gaussian noise and provides a better trade-off between noise removal and edge preservation compared to the mean filter.

### Bilateral Filter

The bilateral filter is an edge-preserving filter that applies a weighted average of the neighborhood, with weights based on both spatial distance and intensity difference. It is effective for removing Gaussian noise while preserving edges and textures.

### Non-Local Means Filter

The Non-Local Means (NLM) filter is an advanced denoising technique that exploits the redundancy in natural images. It replaces each pixel with a weighted average of pixels with similar neighborhoods, even if they are far apart in the image. It is effective for removing Gaussian noise while preserving fine details and textures.

## Usage Examples

### Command-Line Interface

The module can be used from the command line using the `imgproc` tool:

```bash
# Add Gaussian noise and apply mean filter
imgproc denoising --noise gaussian --method mean --noise_param 0.1 --kernel_size 3 --image data/jambe.tif

# Add salt and pepper noise and apply median filter
imgproc denoising --noise salt_pepper --method median --noise_param 0.05 --kernel_size 3 --image data/jambe.tif

# Add exponential noise and apply bilateral filter
imgproc denoising --noise exponential --method bilateral --noise_param 0.1 --kernel_size 3 --sigma 0.1 --image data/jambe.tif

# Compare all denoising methods
imgproc denoising --noise gaussian --method all --noise_param 0.1 --image data/jambe.tif
```

### Python API

The module can also be used programmatically:

```python
from src.image_processing.denoising import (
    generate_gaussian_noise,
    add_noise_to_image,
    apply_median_filter,
    compare_denoising_methods
)
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('data/jambe.tif', as_gray=True))

# Add Gaussian noise
noisy_image = add_noise_to_image(image, 'gaussian', mean=0, std=0.1)

# Apply median filter
denoised_image = apply_median_filter(noisy_image, kernel_size=3)

# Compare all denoising methods
denoised_images = compare_denoising_methods(image, noisy_image)
```

## Results and Comparison

The module includes functions for comparing the performance of different denoising techniques in terms of:

- Visual quality
- Peak Signal-to-Noise Ratio (PSNR)
- Computational time

The results show that:

- The mean filter is the fastest but tends to blur edges.
- The median filter is effective for removing salt and pepper noise while preserving edges.
- The Gaussian filter provides a good trade-off between noise removal and edge preservation.
- The bilateral filter is effective for preserving edges and textures but is computationally more expensive.
- The Non-Local Means filter provides the best visual quality but is the most computationally expensive.

The choice of denoising technique depends on the type of noise, the desired trade-off between noise removal and detail preservation, and the computational constraints.

## References

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
2. Buades, A., Coll, B., & Morel, J. M. (2005). A non-local algorithm for image denoising. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05) (Vol. 2, pp. 60-65). IEEE.
3. Tomasi, C., & Manduchi, R. (1998). Bilateral filtering for gray and color images. In Sixth International Conference on Computer Vision (IEEE Cat. No. 98CH36271) (pp. 839-846). IEEE.

## Author

Oussama GUELFAA
Date: 01-05-2025
