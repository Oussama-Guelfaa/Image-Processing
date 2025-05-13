# Convolution Module

This module provides functionality for applying convolution operations to images.

## Overview

The convolution module allows you to:

1. Apply convolution operations to images
2. Create custom convolution kernels
3. Implement various image processing operations using convolution

## Functions

### Core Functions

- `convolve(image, kernel)`: Apply a convolution kernel to an image
- `apply_kernel(image, kernel, padding='reflect')`: Apply a kernel with different padding options
- `create_kernel(kernel_type, size)`: Create a predefined kernel of a specific type and size

### Kernel Types

- `'box'`: Box blur kernel (all values are equal)
- `'gaussian'`: Gaussian blur kernel
- `'sobel_x'`: Sobel operator for horizontal edges
- `'sobel_y'`: Sobel operator for vertical edges
- `'laplacian'`: Laplacian operator for edge detection
- `'sharpen'`: Sharpening kernel
- `'emboss'`: Emboss effect kernel

## Usage Examples

### Python API

```python
from src.image_processing.convolution import convolve, create_kernel
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('path/to/image.jpg', as_gray=True))

# Create a Gaussian blur kernel
kernel = create_kernel('gaussian', size=5)

# Apply convolution
blurred_image = convolve(image, kernel)

# Create and apply a Sobel operator for edge detection
sobel_x = create_kernel('sobel_x', size=3)
edges_x = convolve(image, sobel_x)
```

## Theory

### Convolution

Convolution is a mathematical operation that combines two functions to produce a third function. In image processing, convolution is used to apply various filters to images.

The 2D discrete convolution of an image f with a kernel h is defined as:

(f * h)[m, n] = Σ Σ f[i, j] · h[m-i, n-j]

Where:
- f is the input image
- h is the convolution kernel
- * denotes the convolution operation
- [m, n] are the coordinates in the output image
- [i, j] are the coordinates in the input image

### Padding

When applying convolution to an image, the output size can be smaller than the input size. To maintain the same size, padding is applied to the input image. Common padding methods include:

- `'constant'`: Pad with a constant value (usually 0)
- `'reflect'`: Pad with the reflection of the image
- `'symmetric'`: Pad with the symmetric reflection of the image
- `'wrap'`: Pad with the wrapped version of the image

### Common Kernels

#### Box Blur

```
1/9 * [1 1 1]
      [1 1 1]
      [1 1 1]
```

#### Gaussian Blur

```
1/16 * [1 2 1]
        [2 4 2]
        [1 2 1]
```

#### Sobel (Horizontal)

```
[1  0  -1]
[2  0  -2]
[1  0  -1]
```

#### Sobel (Vertical)

```
[ 1  2  1]
[ 0  0  0]
[-1 -2 -1]
```

#### Laplacian

```
[0  1  0]
[1 -4  1]
[0  1  0]
```

#### Sharpen

```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

## Author

Oussama GUELFAA
Date: 01-04-2025
