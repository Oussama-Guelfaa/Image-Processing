#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Damage modeling module for image processing.

This module provides functions for modeling and simulating damage to images,
including blurring, noise addition, and restoration.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

from .damage_modeling import (
    generate_checkerboard,
    generate_gaussian_psf,
    generate_motion_blur_psf,
    visualize_psf,
    apply_damage,
    inverse_filter,
    wiener_filter,
    load_image
)

__all__ = [
    'generate_checkerboard',
    'generate_gaussian_psf',
    'generate_motion_blur_psf',
    'visualize_psf',
    'apply_damage',
    'inverse_filter',
    'wiener_filter',
    'load_image'
]
