# Image Registration Documentation

This directory contains documentation for the image registration module.

## Contents

- [Image Registration](image_registration_README.md) - Comprehensive documentation for image registration techniques

## Overview

The registration module provides techniques for aligning two or more images of the same scene:

- Rigid transformation estimation (rotation + translation)
- ICP (Iterative Closest Point) algorithm
- Manual point selection for registration
- Automatic corner detection

## Usage

```python
from src.image_processing.registration import estimate_rigid_transform, apply_rigid_transform, icp_registration

# Estimate rigid transformation between two sets of points
R, t = estimate_rigid_transform(source_points, target_points)

# Apply rigid transformation to an image
registered_image = apply_rigid_transform(source_image, R, t)

# Apply ICP algorithm
R, t, transformed_points, error = icp_registration(source_points, target_points)
```
