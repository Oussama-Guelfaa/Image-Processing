# Image Segmentation Documentation

This directory contains documentation for the image segmentation module.

## Overview

The segmentation module provides techniques for partitioning an image into multiple segments:

- K-means clustering for color segmentation
- Thresholding-based segmentation
- Region-based segmentation

## Usage

```python
from src.image_processing.segmentation import kmeans_segmentation

# Apply K-means segmentation
segmented_image = kmeans_segmentation(image, k=5)
```
