# Fourier Transform Module

This module provides functionality for applying Fourier transforms to images and performing frequency domain operations.

## Overview

The Fourier transform module allows you to:

1. Apply the Fourier transform to images
2. Apply the inverse Fourier transform
3. Visualize the frequency spectrum of images
4. Perform operations in the frequency domain
5. Understand aliasing effects in image processing

## Functions

### Fourier Transform

- `fourier_transform(image)`: Apply the Fourier transform to an image
- `inverse_fourier_transform(spectrum)`: Apply the inverse Fourier transform
- `visualize_spectrum(spectrum, title)`: Visualize the frequency spectrum
- `shift_spectrum(spectrum)`: Shift the zero-frequency component to the center
- `unshift_spectrum(spectrum)`: Shift the zero-frequency component back to the origin

### Aliasing Effects

- `demonstrate_aliasing(image, sampling_factor)`: Demonstrate aliasing effects by downsampling
- `visualize_aliasing(original, downsampled, spectrum_original, spectrum_downsampled)`: Visualize aliasing effects

## Usage Examples

### Command Line

```bash
# Run Fourier transform analysis on an image
imgproc fourier --image path/to/image.jpg

# Apply a low-pass filter in the frequency domain
imgproc filter --type lowpass --cutoff 30 --image path/to/image.jpg

# Apply a high-pass filter in the frequency domain
imgproc filter --type highpass --cutoff 30 --image path/to/image.jpg
```

### Python API

```python
from src.image_processing.fourier import fourier_transform, inverse_fourier_transform, visualize_spectrum
from skimage import io, img_as_float
import numpy as np

# Load an image
image = img_as_float(io.imread('path/to/image.jpg', as_gray=True))

# Apply Fourier transform
spectrum = fourier_transform(image)

# Visualize the spectrum
visualize_spectrum(spectrum, title='Frequency Spectrum')

# Apply a low-pass filter in the frequency domain
rows, cols = spectrum.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), dtype=np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
filtered_spectrum = spectrum * mask

# Apply inverse Fourier transform
filtered_image = inverse_fourier_transform(filtered_spectrum)
```

## Theory

### Fourier Transform

The Fourier transform decomposes an image into its frequency components. For a 2D image f(x, y), the Fourier transform F(u, v) is defined as:

F(u, v) = ∑∑ f(x, y) * e^(-j2π(ux/M + vy/N))

Where:
- f(x, y) is the image in the spatial domain
- F(u, v) is the image in the frequency domain
- M and N are the dimensions of the image
- j is the imaginary unit
- u and v are the frequency variables

### Inverse Fourier Transform

The inverse Fourier transform reconstructs an image from its frequency components. For a frequency spectrum F(u, v), the inverse Fourier transform f(x, y) is defined as:

f(x, y) = (1/MN) * ∑∑ F(u, v) * e^(j2π(ux/M + vy/N))

### Frequency Spectrum

The frequency spectrum of an image represents the magnitude and phase of the frequency components. The magnitude spectrum |F(u, v)| is often visualized using a logarithmic scale:

log(1 + |F(u, v)|)

### Aliasing

Aliasing occurs when an image is sampled at a rate lower than the Nyquist rate, causing high-frequency components to appear as lower frequencies. This can lead to artifacts such as jagged edges and moiré patterns.

## Applications

### Filtering

The Fourier transform is useful for filtering operations in the frequency domain:

- Low-pass filtering: Remove high-frequency components (blur the image)
- High-pass filtering: Remove low-frequency components (enhance edges)
- Band-pass filtering: Keep a specific range of frequencies
- Notch filtering: Remove specific frequencies (e.g., to remove periodic noise)

### Image Compression

The Fourier transform is used in image compression algorithms such as JPEG, where high-frequency components are quantized more aggressively than low-frequency components.

### Image Analysis

The Fourier transform is useful for analyzing the frequency content of images, which can be used for tasks such as texture analysis and pattern recognition.

## Author

Oussama GUELFAA
Date: 01-04-2025
