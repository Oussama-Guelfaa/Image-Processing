# Scale-Space Decomposition and Multiscale Filtering

## Introduction

Scale-space decomposition is a technique used in image processing to analyze images at different scales. It allows for the extraction of features at various levels of detail, which is particularly useful for tasks such as feature detection, segmentation, and pattern recognition.

This document explains the implementation of two scale-space decomposition methods:
1. Morphological Multiscale Decomposition
2. Kramer and Bruckner Multiscale Decomposition

## Morphological Multiscale Decomposition

Morphological multiscale decomposition involves applying morphological operations (dilation and erosion) to an image using structuring elements of increasing size. This creates a pyramid of images where each level represents the image at a different scale.

### Theory

In morphological multiscale decomposition, we use two basic operations:

1. **Dilation**: Expands the bright regions in an image. Mathematically, dilation of an image $f$ by a structuring element $B$ is defined as:
   
   $$(f \oplus B)(x) = \max_{b \in B} f(x-b)$$

2. **Erosion**: Shrinks the bright regions in an image. Mathematically, erosion of an image $f$ by a structuring element $B$ is defined as:
   
   $$(f \ominus B)(x) = \min_{b \in B} f(x+b)$$

By applying these operations with structuring elements of increasing size (typically disks with increasing radius), we create a scale-space representation of the image.

### Implementation

The implementation creates two pyramids:
- A pyramid of dilations, where each level is the result of dilating the original image with a disk of increasing radius.
- A pyramid of erosions, where each level is the result of eroding the original image with a disk of increasing radius.

```python
def morphological_multiscale(image, levels=4):
    pyramid_dilations = []
    pyramid_erosions = []
    
    for r in range(1, levels + 1):
        # Create disk structuring element with increasing radius
        se = disk(r)
        
        # Apply dilation
        dilated = dilation(image, selem=se)
        pyramid_dilations.append(dilated)
        
        # Apply erosion
        eroded = erosion(image, selem=se)
        pyramid_erosions.append(eroded)
    
    return pyramid_dilations, pyramid_erosions
```

## Kramer and Bruckner Multiscale Decomposition

Kramer and Bruckner multiscale decomposition is based on an iterative filter that combines dilation and erosion in a way that preserves edges while smoothing homogeneous regions.

### Theory

The Kramer and Bruckner filter (also known as a toggle filter) is defined as:

$$K_B(f)(x) = \begin{cases}
D_B(f)(x) & \text{if } D_B(f)(x) - f \leq f - E_B(f)(x) \\
E_B(f)(x) & \text{otherwise}
\end{cases}$$

where:
- $D_B(f)$ is the dilation of image $f$ with structuring element $B$
- $E_B(f)$ is the erosion of image $f$ with structuring element $B$

The iterative filter is then defined as:

$$MK^n_B(f) = K_B(MK^{n-1}_B(f))$$

where $MK^0_B(f) = f$ (the original image).

This filter has the property of enhancing edges while smoothing homogeneous regions, making it useful for edge detection and segmentation tasks.

### Implementation

The implementation consists of two main functions:

1. The elementary Kramer-Bruckner filter:

```python
def kramer_bruckner_filter(image, radius=5):
    # Create disk structuring element
    se = disk(radius)
    
    # Apply dilation and erosion
    dilated = dilation(image, selem=se)
    eroded = erosion(image, selem=se)
    
    # Calculate difference between image and operations
    diff_dilation = np.abs(dilated - image)
    diff_erosion = np.abs(image - eroded)
    
    # Apply the filter rule: choose dilation or erosion based on which is closer to original
    filtered = np.where(diff_dilation <= diff_erosion, dilated, eroded)
    
    return filtered
```

2. The iterative application of the filter to create a multiscale decomposition:

```python
def kramer_bruckner_multiscale(image, levels=3, radius=5):
    kb_filters = [image]  # Start with original image
    
    current_image = image.copy()
    for i in range(levels):
        # Apply KB filter to current image
        filtered = kramer_bruckner_filter(current_image, radius)
        kb_filters.append(filtered)
        current_image = filtered
    
    return kb_filters
```

## Usage

The scale-space decomposition functionality can be used both programmatically and via the command-line interface.

### Programmatic Usage

```python
from image_processing.multiscale import morphological_multiscale, kramer_bruckner_multiscale

# Load an image
image = load_image('path/to/image.jpg')

# Morphological multiscale decomposition
pyramid_dilations, pyramid_erosions = morphological_multiscale(image, levels=4)

# Kramer and Bruckner multiscale decomposition
kb_filters = kramer_bruckner_multiscale(image, levels=3, radius=5)
```

### Command-Line Usage

```bash
python -m image_processing.multiscale.scale_space_cli --image data/cerveau.jpg --levels 4 --kb-iterations 3 --radius 5
```

Options:
- `--image`: Path to the input image (default: data/cerveau.jpg)
- `--output-dir`: Directory to save output images (default: output/multiscale/scale_space)
- `--levels`: Number of levels for morphological decomposition (default: 4)
- `--kb-iterations`: Number of iterations for Kramer-Bruckner filter (default: 3)
- `--radius`: Radius of the structuring element (disk) (default: 5)
- `--method`: Decomposition method to use (choices: morphological, kb, both; default: both)

## Applications

Scale-space decomposition has several applications in image processing and computer vision:

1. **Feature Detection**: Identifying features at different scales.
2. **Edge Detection**: Enhancing edges while suppressing noise.
3. **Image Segmentation**: Separating objects at different scales.
4. **Texture Analysis**: Analyzing texture patterns at multiple scales.
5. **Medical Image Analysis**: Identifying structures of different sizes in medical images.

## References

1. Kramer, H. P., & Bruckner, J. B. (1975). Iterations of a non-linear transformation for enhancement of digital images. Pattern Recognition, 7(1-2), 53-58.
2. Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.
3. Maragos, P. (1989). Pattern spectrum and multiscale shape representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 11(7), 701-716.

## Author

Oussama GUELFAA  
Date: 01-04-2025
