"""
Feature extraction module for image processing.

This module contains functions to extract features from images for machine learning tasks.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
from skimage import measure, io
import glob
import os

def extract_region_props(image_path):
    """
    Extract region properties from a binary image.
    
    Parameters
    ----------
    image_path : str
        Path to the binary image.
        
    Returns
    -------
    features : ndarray
        Array of features extracted from the image.
    """
    # Read the image
    img = io.imread(image_path, as_gray=True)
    
    # Ensure binary image
    if img.max() > 1:
        img = img > 128
    
    # Extract region properties
    props = measure.regionprops(measure.label(img))[0]
    
    # Extract features
    features = np.array([
        props.area,                      # Area of the region
        props.perimeter,                 # Perimeter of the region
        props.eccentricity,              # Eccentricity of the region
        props.equivalent_diameter_area,  # Diameter of circle with same area
        props.euler_number,              # Euler number
        props.extent,                    # Ratio of pixels in region to pixels in bounding box
        props.major_axis_length,         # Length of major axis
        props.minor_axis_length,         # Length of minor axis
        props.solidity                   # Ratio of pixels in the region to pixels in the convex hull
    ])
    
    return features

def load_dataset(dataset_path, classes):
    """
    Load all images from the dataset and extract features.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory.
    classes : list
        List of class names.
        
    Returns
    -------
    features : ndarray
        Array of features for all images.
    targets : ndarray
        Array of target classes for all images.
    """
    all_features = []
    all_targets = []
    
    for idx, class_name in enumerate(classes):
        # Get all images for this class
        class_path = os.path.join(dataset_path, class_name + '*')
        image_files = glob.glob(class_path)
        
        for image_file in image_files:
            try:
                # Extract features
                features = extract_region_props(image_file)
                all_features.append(features)
                all_targets.append(idx)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    return np.array(all_features), np.array(all_targets)
