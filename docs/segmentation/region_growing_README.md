# Region Growing Segmentation

**Author:** Oussama GUELFAA  
**Date:** 23-05-2025

## Table of Contents
1. [Introduction](#introduction)
2. [Theoretical Background](#theoretical-background)
3. [Implementation Approach](#implementation-approach)
4. [Predicate Functions](#predicate-functions)
5. [Algorithm](#algorithm)
6. [Usage](#usage)
7. [Examples](#examples)
8. [References](#references)

## Introduction

Region growing is a simple yet powerful image segmentation technique that groups pixels or subregions into larger regions based on predefined criteria. The algorithm starts with a set of "seed" points and grows regions from these seeds by appending neighboring pixels that satisfy a homogeneity criterion.

This document explains the theory behind region growing segmentation, details our implementation approach, and provides usage examples.

## Theoretical Background

Region growing is a pixel-based image segmentation method that involves the following steps:

1. Select initial seed points
2. Define a homogeneity criterion (predicate function)
3. Grow regions by adding neighboring pixels that satisfy the criterion
4. Stop when no more pixels can be added to any region

The algorithm is particularly useful for segmenting images with clear boundaries or homogeneous regions. It's widely used in medical image analysis, object detection, and image understanding tasks.

The key advantage of region growing is its ability to correctly separate regions that have the same properties and are spatially separated. It also provides good edge definition in the final segmentation result.

## Implementation Approach

Our implementation of region growing segmentation follows these design principles:

1. **Flexibility**: Support for multiple predicate functions to accommodate different segmentation needs
2. **Interactivity**: Allow users to select seed points manually
3. **Visualization**: Provide comprehensive visualization of segmentation results
4. **Efficiency**: Optimize the algorithm for performance using queue-based processing

The implementation uses a breadth-first search approach with a queue data structure to efficiently process pixels in the order they are added to the region.

## Predicate Functions

The predicate function is the heart of the region growing algorithm. It determines whether a pixel should be included in a region. We've implemented three different predicate functions:

### 1. Intensity Difference from Seed

This predicate compares the intensity of a candidate pixel with the intensity of the seed pixel:

```python
|I(p) - I(seed)| ≤ T
```

Where:
- `I(p)` is the intensity of the candidate pixel
- `I(seed)` is the intensity of the seed pixel
- `T` is a threshold value

### 2. Intensity Difference from Region Mean

This predicate compares the intensity of a candidate pixel with the mean intensity of the current region:

```python
|I(p) - mᵣ| ≤ T
```

Where:
- `I(p)` is the intensity of the candidate pixel
- `mᵣ` is the mean intensity of the current region
- `T` is a threshold value

### 3. Adaptive Threshold

This predicate uses an adaptive threshold that varies depending on the region statistics:

```python
|I(p) - mᵣ| ≤ T × (1 - σᵣ/mᵣ)
```

Where:
- `I(p)` is the intensity of the candidate pixel
- `mᵣ` is the mean intensity of the current region
- `σᵣ` is the standard deviation of the current region
- `T` is a base threshold value

This adaptive approach allows the threshold to adjust based on the homogeneity of the region. In homogeneous regions (low standard deviation), the threshold is close to T. In heterogeneous regions (high standard deviation), the threshold is reduced.

## Algorithm

The region growing algorithm is implemented as follows:

```
Data: I: image
Data: seed: starting point
Data: predicate: homogeneity criterion
Result: visited: boolean matrix, same size as I

begin
    queue.enqueue(seed);
    visited[seed] = true;
    
    while queue is not empty do
        p = queue.dequeue();
        
        foreach neighbor of p do
            if not visited[neighbor] and neighbor verifies predicate then
                queue.enqueue(neighbor);
                visited[neighbor] = true;
            end
        end
    end
    
    return visited
end
```

The algorithm uses a queue to keep track of pixels that need to be processed. It starts with the seed pixel and iteratively adds neighboring pixels that satisfy the predicate function.

## Usage

### Command Line Interface

The region growing segmentation can be run from the command line using:

```bash
python -m src.cli segmentation --method region_growing --image path/to/image.jpg --output path/to/output/dir
```

Parameters:
- `--method`: Set to `region_growing` to use the region growing algorithm
- `--image`: Path to the input image (optional, uses default image if not provided)
- `--output`: Directory to save the output images (optional)
- `--threshold`: Threshold value for the predicate function (default: 20)

### Interactive Mode

When running the algorithm, you'll be prompted to select a seed point by clicking on the image. After selecting the seed, the algorithm will run with all three predicate functions and display the results.

## Examples

Here are some examples of region growing segmentation with different predicate functions:

### Example 1: Medical Image Segmentation

Input image: Brain MRI scan
Seed point: Selected in the tumor region
Results:
- Intensity difference from seed: Captures the core tumor region
- Intensity difference from region mean: Captures the tumor with surrounding edema
- Adaptive threshold: Provides the most accurate tumor segmentation

### Example 2: Natural Image Segmentation

Input image: Natural scene with distinct objects
Seed point: Selected in the foreground object
Results:
- Intensity difference from seed: Captures parts of the object with similar intensity
- Intensity difference from region mean: Captures more of the object as the region grows
- Adaptive threshold: Adapts to intensity variations within the object

## References

1. Adams, R., & Bischof, L. (1994). Seeded region growing. IEEE Transactions on Pattern Analysis and Machine Intelligence, 16(6), 641-647.
2. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
3. Shapiro, L. G., & Stockman, G. C. (2001). Computer Vision. Prentice Hall.
