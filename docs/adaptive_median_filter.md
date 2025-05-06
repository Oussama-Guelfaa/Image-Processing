# Adaptive Median Filter

## Overview

The adaptive median filter is an advanced noise reduction technique specifically designed to handle salt-and-pepper noise while preserving image details. Unlike the standard median filter that uses a fixed window size, the adaptive median filter adjusts its window size based on local image characteristics.

Author: Oussama GUELFAA  
Date: 01-04-2025

## Theory

### Salt-and-Pepper Noise

Salt-and-pepper noise (also known as impulse noise) is characterized by random occurrences of white and black pixels in an image. This type of noise typically arises from errors in image transmission, faulty camera sensors, or analog-to-digital converter errors.

### Limitations of Standard Median Filter

The standard median filter replaces each pixel with the median value of its neighborhood. While effective for removing salt-and-pepper noise, it has limitations:

1. A small window size may not effectively remove noise if the noise density is high
2. A large window size can cause significant blurring and loss of image details
3. The fixed window size doesn't adapt to local image characteristics

### Adaptive Median Filter Algorithm

The adaptive median filter addresses these limitations by:

1. Starting with a small window size (typically 3×3)
2. Examining the local neighborhood to determine if the current pixel is likely noise
3. If the pixel is identified as noise, replacing it with the median value
4. If the pixel is not noise, keeping its original value
5. If a decision cannot be made with the current window size, increasing the window size and repeating the process

The algorithm can be described as follows:

#### Level A: Determine if the median value is an impulse

1. Calculate the minimum (Zmin), maximum (Zmax), and median (Zmed) values in the window
2. If Zmin < Zmed < Zmax, go to Level B
3. Else, increase the window size and repeat Level A
4. If the maximum window size is reached, set the output to Zmed

#### Level B: Determine if the center pixel is an impulse

1. Let Zxy be the value of the center pixel
2. If Zmin < Zxy < Zmax, the pixel is not an impulse, set the output to Zxy
3. Else, the pixel is an impulse, set the output to Zmed

## Implementation

Our implementation provides two versions of the adaptive median filter:

1. `adaptive_median_filter`: A straightforward implementation that follows the algorithm described above
2. `fast_adaptive_median_filter`: An optimized version that uses more efficient data structures and processing techniques

### Usage

```python
from src.image_processing.denoising import adaptive_median_filter, fast_adaptive_median_filter

# Apply the adaptive median filter
filtered_image = adaptive_median_filter(noisy_image, max_window_size=7)

# Apply the fast adaptive median filter
filtered_image = fast_adaptive_median_filter(noisy_image, max_window_size=7)
```

### Parameters

- `image`: Input image (grayscale)
- `max_window_size`: Maximum window size (default: 7)

### Command-Line Usage

```bash
python main.py denoising --method adaptive_median --noise salt_pepper --noise_param 0.1
python main.py denoising --method fast_adaptive_median --noise salt_pepper --noise_param 0.1
```

## Performance

The adaptive median filter performs significantly better than the standard median filter for images with salt-and-pepper noise, especially at higher noise densities. It preserves edges and fine details while effectively removing noise.

### Advantages

1. Better preservation of image details compared to standard median filter
2. More effective at removing high-density impulse noise
3. Adaptive behavior based on local image characteristics

### Limitations

1. Higher computational complexity compared to standard median filter
2. May not perform as well on other types of noise (e.g., Gaussian noise)

## Experimental Results

Our experiments show that the adaptive median filter achieves higher PSNR (Peak Signal-to-Noise Ratio) values compared to the standard median filter, especially at higher noise densities.

The fast implementation provides similar quality results with improved processing time, making it suitable for real-time applications.

## References

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
2. Chan, R. H., Ho, C. W., & Nikolova, M. (2005). Salt-and-pepper noise removal by median-type noise detectors and detail-preserving regularization. IEEE Transactions on Image Processing, 14(10), 1479-1485.
3. Hwang, H., & Haddad, R. A. (1995). Adaptive median filters: new algorithms and results. IEEE Transactions on Image Processing, 4(4), 499-502.
