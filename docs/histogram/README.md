# Histogram Processing Documentation

This directory contains documentation for the histogram processing module.

## Contents

- [Histogram Equalization](README_HISTOGRAM_EQUALIZATION.md) - Documentation for histogram equalization
- [Histogram Matching](README_HISTOGRAM_MATCHING.md) - Documentation for histogram matching

## Overview

The histogram module provides techniques for manipulating image histograms to improve contrast and match specific distributions:

- Histogram equalization for contrast enhancement
- Histogram matching to match the histogram of one image to another

## Usage

```python
from src.image_processing.histogram import equalize_histogram, match_histogram

# Apply histogram equalization
equalized_image = equalize_histogram(image)

# Apply histogram matching
matched_image = match_histogram(image, reference_image)
```
