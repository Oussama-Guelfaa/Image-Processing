# Image Filtering Documentation

This directory contains documentation for the image filtering module.

## Contents

- [Damage Modeling and Restoration](damage_modeling_README.md) - Documentation for damage modeling and image restoration
- [Wiener Filter Optimal K](wiener_filter_optimal_k.md) - Documentation for finding the optimal K parameter for Wiener filtering
- [Noise Filtering](noise_filtering_README.md) - Documentation for noise filtering techniques, particularly for salt and pepper noise

## Overview

The filtering module provides various image filtering techniques including:

- Low-pass filtering
- High-pass filtering
- Derivative filtering
- Wiener filtering for image restoration
- Noise filtering (mean, median, maximum, minimum, adaptive median)

## Usage

### Frequency Domain Filtering

```python
from src.image_processing.filtering import apply_lowpass_filter, apply_highpass_filter, apply_wiener_filter

# Apply low-pass filter
filtered_image = apply_lowpass_filter(image, cutoff=30)

# Apply high-pass filter
filtered_image = apply_highpass_filter(image, cutoff=30)

# Apply Wiener filter
restored_image = apply_wiener_filter(damaged_image, psf, k=0.01)
```

### Noise Filtering

```python
from src.image_processing.denoising import (
    apply_mean_filter,
    apply_median_filter,
    adaptive_median_filter
)
from skimage.util import random_noise

# Add salt and pepper noise
noisy_image = random_noise(image, mode='s&p', amount=0.1)

# Apply median filter
median_filtered = apply_median_filter(noisy_image, kernel_size=5)

# Apply adaptive median filter
adaptive_filtered = adaptive_median_filter(noisy_image, max_window_size=7)
```

For more detailed examples and explanations, see the [Noise Filtering documentation](noise_filtering_README.md).
