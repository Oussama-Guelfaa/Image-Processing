# Intensity Transformations Documentation

This directory contains documentation for the intensity transformations module.

## Contents

- [Intensity Transformations](README_INTENSITY_TRANSFORMATIONS.md) - Documentation for intensity transformations

## Overview

The transformations module provides techniques for manipulating pixel intensities:

- Gamma correction
- Logarithmic transformation
- Contrast stretching
- Negative transformation

## Usage

```python
from src.image_processing.transformations import apply_gamma_correction, apply_contrast_stretching

# Apply gamma correction
corrected_image = apply_gamma_correction(image, gamma=0.5)

# Apply contrast stretching
stretched_image = apply_contrast_stretching(image, E=4.0)
```
