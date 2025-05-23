# Multiscale Analysis

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This module implements multiscale analysis techniques for image processing, focusing on:

1. **Pyramidal Decomposition and Reconstruction**: Analyzing images at different scales using Gaussian and Laplacian pyramids, and reconstructing them with varying levels of detail.

2. **Scale-Space Decomposition and Multiscale Filtering**: Analyzing images using morphological operations and Kramer-Bruckner filtering to extract features at different scales.

## Theoretical Background

### Pyramidal Representation

Pyramidal representation is a multi-resolution technique that represents an image at multiple scales. This is achieved by creating a series of images, each at a lower resolution than the previous one, forming a pyramid structure.

### Gaussian Pyramid

The Gaussian pyramid is constructed by:
1. Starting with the original image
2. Applying Gaussian blur to the image
3. Downsampling the blurred image (typically by a factor of 2)
4. Repeating steps 2-3 for each level of the pyramid

Each level of the Gaussian pyramid represents the image at a lower resolution, with high-frequency details progressively removed.

### Laplacian Pyramid

The Laplacian pyramid captures the details lost between successive levels of the Gaussian pyramid:
1. For each level (except the last), compute the difference between the current Gaussian level and an upsampled version of the next level
2. The last level of the Laplacian pyramid is the same as the last level of the Gaussian pyramid

The Laplacian pyramid effectively stores the high-frequency details that are lost when moving from one level to the next in the Gaussian pyramid.

### Image Reconstruction

The original image can be reconstructed from the Laplacian pyramid by:
1. Starting with the smallest level
2. Upsampling and adding the corresponding Laplacian level
3. Repeating until reaching the original image size

This process allows for perfect reconstruction of the original image.

## Implementation Results

### Original Image
![Original Image](../../output/multiscale/original_image.png)

### Gaussian Pyramid
![Gaussian Pyramid](../../output/multiscale/gaussian_pyramid.png)

The Gaussian pyramid shows the image at progressively lower resolutions. Each level is a blurred and downsampled version of the previous level.

### Laplacian Pyramid
![Laplacian Pyramid](../../output/multiscale/laplacian_pyramid.png)

The Laplacian pyramid shows the details lost between successive levels of the Gaussian pyramid. Note that the last level is the same as the last level of the Gaussian pyramid.

### Reconstruction Comparison
![Reconstruction Comparison](../../output/multiscale/reconstruction_comparison.png)

This comparison shows:
- The original image (left)
- Reconstruction without details using only the coarsest level of the Gaussian pyramid (middle)
- Perfect reconstruction using the Laplacian pyramid (right)

### Reconstruction Error

- **With Laplacian details**: 0.00000000 (perfect reconstruction)
- **Without details**: 0.02392446 (loss of high-frequency information)

## Key Observations

1. **Perfect Reconstruction**: The Laplacian pyramid allows for perfect reconstruction of the original image, as demonstrated by the zero reconstruction error.

2. **Information Distribution**: The Laplacian pyramid effectively separates image information across different scales, with each level capturing details at a specific frequency band.

3. **Coarse-to-Fine Representation**: The Gaussian pyramid provides a coarse-to-fine representation of the image, with the coarsest level capturing the overall structure and finer levels adding details.

4. **Detail Loss**: Reconstruction from only the coarsest level of the Gaussian pyramid results in a loss of high-frequency details, as shown by the non-zero reconstruction error.

## Applications

This multiscale analysis technique has several practical applications:

1. **Image Compression**: By quantizing or discarding some levels of the Laplacian pyramid, images can be compressed with controlled loss of detail.

2. **Image Blending**: Laplacian pyramids enable seamless blending of images by combining them at different scales.

3. **Feature Detection**: Different scales capture different features, making multiscale analysis useful for robust feature detection.

4. **Noise Reduction**: Noise can be effectively reduced by manipulating coefficients at different scales while preserving important image features.

## Scale-Space Decomposition

In addition to pyramidal decomposition, this module implements scale-space decomposition techniques:

### Morphological Multiscale Decomposition

Morphological multiscale decomposition uses dilation and erosion operations with structuring elements of increasing size to analyze images at different scales:

- **Dilation**: Expands bright regions in an image
- **Erosion**: Shrinks bright regions in an image

By applying these operations with disks of increasing radius, we create two pyramids:
- A pyramid of dilations
- A pyramid of erosions

This approach preserves the original image size while analyzing features at different scales.

### Kramer and Bruckner Multiscale Decomposition

The Kramer and Bruckner filter (also known as a toggle filter) combines dilation and erosion in a way that enhances edges while smoothing homogeneous regions. The filter is applied iteratively to create a multiscale decomposition.

For each pixel, the filter chooses between the dilated and eroded value based on which is closer to the original image value. This creates a sharpening effect that enhances edges.

For detailed information about scale-space decomposition, see the [Scale-Space Documentation](scale_space.md).

## Conclusion

The multiscale analysis implementation demonstrates the power of both pyramidal decomposition and scale-space decomposition for image processing:

- **Pyramidal Decomposition**: The Laplacian pyramid provides a complete representation of the image across different scales, allowing for perfect reconstruction, while the Gaussian pyramid offers a simplified, multi-resolution view of the image structure.

- **Scale-Space Decomposition**: Morphological operations and Kramer-Bruckner filtering provide alternative ways to analyze images at different scales, with particular strengths in edge enhancement and feature detection.

These techniques are powerful tools for various image processing applications, from compression and blending to feature detection and segmentation.
