#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine_learning

Machine learning techniques for image processing and analysis.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

from .feature_extraction import (
    extract_features,
    extract_hu_moments,
    extract_zernike_moments,
    extract_geometric_features,
    extract_hog_features,
    extract_dataset_features,
    load_kimia_dataset
)

from .classification import (
    train_test_split_dataset,
    train_classifier,
    evaluate_classifier,
    classify_image,
    cross_validate
)

from .visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    visualize_classification_results,
    visualize_dataset
)

__all__ = [
    'extract_features',
    'extract_hu_moments',
    'extract_zernike_moments',
    'extract_geometric_features',
    'extract_hog_features',
    'extract_dataset_features',
    'load_kimia_dataset',
    'train_test_split_dataset',
    'train_classifier',
    'evaluate_classifier',
    'classify_image',
    'cross_validate',
    'plot_confusion_matrix',
    'plot_feature_importance',
    'visualize_classification_results',
    'visualize_dataset'
]
