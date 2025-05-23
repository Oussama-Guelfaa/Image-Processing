#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Region Growing Segmentation

Unit tests for the region growing segmentation algorithm.

Author: Oussama GUELFAA
Date: 23-05-2025
"""

import unittest
import numpy as np
from skimage import data
from src.image_processing.segmentation.region_growing import (
    predicate_intensity_diff,
    predicate_region_mean,
    predicate_adaptive_threshold,
    region_growing
)

class TestRegionGrowing(unittest.TestCase):
    """Test cases for region growing segmentation."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple test image
        self.test_image = np.zeros((10, 10), dtype=np.float32)
        self.test_image[2:8, 2:8] = 0.5  # Create a square region
        
        # Create a more complex test image with gradients
        self.gradient_image = np.zeros((20, 20), dtype=np.float32)
        for i in range(20):
            for j in range(20):
                self.gradient_image[i, j] = (i + j) / 40.0
        
        # Use a real image for more realistic tests
        self.real_image = data.camera() / 255.0  # Normalize to [0, 1]
    
    def test_predicate_intensity_diff(self):
        """Test the intensity difference predicate function."""
        # Test with seed in the square region
        seed = (5, 5)  # Inside the square
        
        # Pixel inside the square should be included
        self.assertTrue(predicate_intensity_diff(self.test_image, 4, 4, seed, threshold=20))
        
        # Pixel outside the square should not be included
        self.assertFalse(predicate_intensity_diff(self.test_image, 1, 1, seed, threshold=20))
        
        # Test with a smaller threshold
        self.assertTrue(predicate_intensity_diff(self.test_image, 4, 4, seed, threshold=1))
        self.assertFalse(predicate_intensity_diff(self.test_image, 1, 1, seed, threshold=1))
    
    def test_predicate_region_mean(self):
        """Test the region mean predicate function."""
        # Create a region mask with some pixels already in the region
        region_mask = np.zeros_like(self.test_image, dtype=bool)
        region_mask[4:7, 4:7] = True  # Small region inside the square
        
        # Pixel inside the square should be included
        self.assertTrue(predicate_region_mean(self.test_image, 3, 3, region_mask, threshold=20))
        
        # Pixel outside the square should not be included
        self.assertFalse(predicate_region_mean(self.test_image, 1, 1, region_mask, threshold=20))
        
        # Test with a smaller threshold
        self.assertTrue(predicate_region_mean(self.test_image, 3, 3, region_mask, threshold=1))
        self.assertFalse(predicate_region_mean(self.test_image, 1, 1, region_mask, threshold=1))
    
    def test_predicate_adaptive_threshold(self):
        """Test the adaptive threshold predicate function."""
        # Create a region mask with some pixels already in the region
        region_mask = np.zeros_like(self.gradient_image, dtype=bool)
        region_mask[5:10, 5:10] = True  # Region in the gradient image
        
        # Pixel close to the region should be included
        self.assertTrue(predicate_adaptive_threshold(self.gradient_image, 10, 10, region_mask, T0=20))
        
        # Pixel far from the region should not be included
        self.assertFalse(predicate_adaptive_threshold(self.gradient_image, 19, 19, region_mask, T0=20))
    
    def test_region_growing_simple(self):
        """Test region growing on a simple image."""
        # Seed in the square region
        seed = (5, 5)
        
        # Run region growing
        result = region_growing(self.test_image, seed, predicate_intensity_diff, threshold=20)
        
        # The result should include the entire square region
        expected = np.zeros_like(self.test_image, dtype=bool)
        expected[2:8, 2:8] = True
        
        # Check if the result matches the expected segmentation
        np.testing.assert_array_equal(result, expected)
    
    def test_region_growing_gradient(self):
        """Test region growing on a gradient image."""
        # Seed in the middle of the gradient
        seed = (10, 10)
        
        # Run region growing with a small threshold
        result = region_growing(self.gradient_image, seed, predicate_intensity_diff, threshold=5)
        
        # The result should include a region around the seed
        # We don't check the exact shape, just that some pixels are included
        self.assertTrue(np.sum(result) > 0)
        self.assertTrue(np.sum(result) < self.gradient_image.size)
        
        # The seed should be included
        self.assertTrue(result[seed[0], seed[1]])
    
    def test_region_growing_real_image(self):
        """Test region growing on a real image."""
        # Seed in the middle of the image
        seed = (self.real_image.shape[0] // 2, self.real_image.shape[1] // 2)
        
        # Run region growing with different predicates
        result1 = region_growing(self.real_image, seed, predicate_intensity_diff, threshold=20)
        result2 = region_growing(self.real_image, seed, predicate_region_mean, threshold=20)
        result3 = region_growing(self.real_image, seed, predicate_adaptive_threshold, T0=20)
        
        # All results should include some pixels
        self.assertTrue(np.sum(result1) > 0)
        self.assertTrue(np.sum(result2) > 0)
        self.assertTrue(np.sum(result3) > 0)
        
        # All results should include the seed
        self.assertTrue(result1[seed[0], seed[1]])
        self.assertTrue(result2[seed[0], seed[1]])
        self.assertTrue(result3[seed[0], seed[1]])

if __name__ == '__main__':
    unittest.main()
