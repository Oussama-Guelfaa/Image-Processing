"""
Demonstration of Fourier transform and filtering techniques

This script demonstrates how to use the image processing functionality
of the project to perform Fourier transforms and apply filters to images.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.util import img_as_float

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the project
from src.image_processing.filtering_hp_lp import LowPassFilter, HighPassFilter

def main():
    # Load an image
    image_path = '../data/cornee.png'
    image = img_as_float(imread(image_path, as_gray=True))
    
    # Calculate the Fourier transform
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    
    # Apply low-pass and high-pass filters
    low_pass = LowPassFilter(fft_shift, 30)
    high_pass = HighPassFilter(fft_shift, 30)
    
    # Inverse Fourier transform to get filtered images
    img_low = np.fft.ifft2(np.fft.ifftshift(low_pass)).real
    img_high = np.fft.ifft2(np.fft.ifftshift(high_pass)).real
    
    # Display the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_low, cmap='gray')
    plt.title('Low-Pass Filtered')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_high, cmap='gray')
    plt.title('High-Pass Filtered')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
