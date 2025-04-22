# Image Processing Tools

This project provides tools for image processing and analysis, including intensity transformations, histogram equalization, histogram matching, filtering, Fourier transforms, image registration, segmentation, and more. It is designed to be easy to use and educational.

## Project Structure

```
Image-Processing/
├── data/                      # Input data (images, etc.)
├── docs/                      # Documentation
│   ├── filtering/             # Documentation on filtering
│   ├── histogram/             # Documentation on histograms
│   ├── registration/          # Documentation on image registration
│   └── segmentation/          # Documentation on segmentation
├── examples/                  # Usage examples
├── latex_documentation/       # LaTeX documentation
├── src/                       # Main source code
│   ├── image_processing/      # Image processing modules
│   │   ├── filtering/         # Filters (low-pass, high-pass, Wiener, etc.)
│   │   ├── fourier/           # Fourier transforms
│   │   ├── histogram/         # Histogram operations
│   │   ├── registration/      # Image registration
│   │   ├── segmentation/      # Image segmentation
│   │   └── transformations/   # Intensity transformations
│   └── utils/                 # Utilities
├── tests/                     # Unit and integration tests
├── main.py                    # Main entry point
├── setup.py                   # Package configuration
└── README.md                  # This file
```

## Modules

### Filtering

Image filtering operations including low-pass, high-pass, derivative, and Wiener filters.

```python
from src.image_processing.filtering import apply_lowpass_filter, apply_highpass_filter
```

### Fourier Transforms

Fourier transform and inverse Fourier transform operations for image processing.

```python
from src.image_processing.fourier import fourier_transform, inverse_fourier_transform
```

### Histogram Operations

Histogram equalization, histogram matching, and other histogram-based techniques.

```python
from src.image_processing.histogram import equalize_histogram, match_histogram
```

### Image Registration

Image registration techniques including manual point selection, rigid transformation estimation, and ICP algorithm.

```python
from src.image_processing.registration import estimate_rigid_transform, apply_rigid_transform
```

### Segmentation

Image segmentation techniques including K-means clustering and other segmentation algorithms.

```python
from src.image_processing.segmentation import kmeans_segmentation
```

### Intensity Transformations

Intensity transformation techniques including gamma correction, logarithmic transformation, and others.

```python
from src.image_processing.transformations import apply_gamma_correction
```

## Installation

You can install this package directly from GitHub:

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/Image-Processing.git
```

Or you can clone the repository and install it locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-Processing.git
cd Image-Processing

# Install the package
pip install -e .
```

## Usage

After installation, you can use the `imgproc` command-line tool:

### Intensity Transformations

```bash
# Apply gamma correction to an image
imgproc intensity --method gamma --gamma 0.5 --image path/to/image.jpg

# Apply contrast stretching
imgproc intensity --method contrast --E 4.0 --image path/to/image.jpg

# Apply both transformations
imgproc intensity --method both --gamma 0.5 --E 4.0 --image path/to/image.jpg
```

### Histogram Processing

```bash
# Apply histogram equalization
imgproc histogram --method custom --bins 256 --image path/to/image.jpg

# Apply histogram matching
imgproc matching --method custom --peak1 0.3 --peak2 0.7 --image path/to/image.jpg
```

### Damage Modeling and Image Restoration

```bash
# Generate a checkerboard image
imgproc checkerboard --size 8 --square_size 32 --output checkerboard.png

# Apply damage to an image using a Gaussian PSF
imgproc damage --psf gaussian --sigma 3.0 --noise 0.01 --image path/to/image.jpg --output damaged.png

# Apply damage using a motion blur PSF
imgproc damage --psf motion --length 15 --angle 45 --noise 0.01 --image path/to/image.jpg

# Restore an image using the inverse filter
imgproc restore --method inverse --psf gaussian --sigma 3.0 --image damaged.png --output restored.png

# Restore an image using the Wiener filter
imgproc restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image damaged.png

# Compare different restoration methods
imgproc restore --method compare --psf gaussian --sigma 3.0 --image damaged.png
```

You can also use positional arguments for image paths:

```bash
# Using positional arguments for image paths
imgproc intensity --method gamma --gamma 0.5 path/to/image.jpg
imgproc damage --psf gaussian --sigma 3.0 path/to/image.jpg
imgproc restore --method wiener --k 0.01 damaged.png
```

You can also run the project using the main.py script without installation:

```bash
# Apply intensity transformations
python main.py intensity --method gamma --gamma 0.5 --image path/to/image.jpg

# Apply damage to an image
python main.py damage --psf gaussian --sigma 3.0 --noise 0.01 --image path/to/image.jpg

# Restore an image
python main.py restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image damaged.png
```

## Examples

Check the examples directory for sample applications.

## Dependencies

- NumPy
- Matplotlib
- scikit-image
