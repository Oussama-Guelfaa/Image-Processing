# Image Filtering Documentation

This directory contains documentation for the image filtering module.

## Contents

- [Damage Modeling and Restoration](damage_modeling_README.md) - Documentation for damage modeling and image restoration
- [Wiener Filter Optimal K](wiener_filter_optimal_k.md) - Documentation for finding the optimal K parameter for Wiener filtering

## Overview

The filtering module provides various image filtering techniques including:

- Low-pass filtering
- High-pass filtering
- Derivative filtering
- Wiener filtering for image restoration

## Usage

```python
from src.image_processing.filtering import apply_lowpass_filter, apply_highpass_filter, apply_wiener_filter

# Apply low-pass filter
filtered_image = apply_lowpass_filter(image, cutoff=30)

# Apply high-pass filter
filtered_image = apply_highpass_filter(image, cutoff=30)

# Apply Wiener filter
restored_image = apply_wiener_filter(damaged_image, psf, k=0.01)
```
