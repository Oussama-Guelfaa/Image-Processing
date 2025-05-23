"""
Machine Learning module for image processing.

This module contains functions and classes for machine learning tasks on images,
including feature extraction, classification, and neural networks.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

from .feature_extraction import extract_region_props
from .classification import train_classifier, evaluate_classifier

__all__ = [
    'extract_region_props',
    'train_classifier',
    'evaluate_classifier'
]
