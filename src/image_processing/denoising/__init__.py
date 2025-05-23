#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising

Techniques for removing noise from images, including various filtering methods.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

# Import denoising modules
from .noise_generation import (
    generate_uniform_noise,
    generate_gaussian_noise,
    generate_salt_pepper_noise,
    generate_exponential_noise,
    add_noise_to_image
)

from .noise_estimation import (
    extract_roi,
    estimate_noise_parameters,
    visualize_roi_histogram
)

from .denoising import (
    apply_mean_filter,
    apply_median_filter,
    apply_gaussian_filter,
    apply_bilateral_filter,
    apply_nlm_filter,
    compare_denoising_methods
)

from .adaptive_median import (
    adaptive_median_filter,
    fast_adaptive_median_filter,
    is_impulse_noise
)
