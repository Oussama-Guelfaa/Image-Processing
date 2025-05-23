#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiscale

Multiscale analysis techniques including pyramidal decomposition and scale-space decomposition.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

from .pyramidal_decomposition import (
    gaussian_pyramid,
    laplacian_pyramid,
    reconstruct_from_laplacian_pyramid,
    reconstruct_from_gaussian_pyramid,
    calculate_reconstruction_error
)

from .scale_space import (
    morphological_multiscale,
    kramer_bruckner_filter,
    kramer_bruckner_multiscale
)

__all__ = [
    # Pyramidal decomposition
    'gaussian_pyramid',
    'laplacian_pyramid',
    'reconstruct_from_laplacian_pyramid',
    'reconstruct_from_gaussian_pyramid',
    'calculate_reconstruction_error',

    # Scale-space decomposition
    'morphological_multiscale',
    'kramer_bruckner_filter',
    'kramer_bruckner_multiscale'
]
