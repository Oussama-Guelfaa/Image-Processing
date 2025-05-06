#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package for image processing functionality.
This includes intensity transformations, histogram equalization, histogram matching,
image registration, filtering, Fourier transforms, segmentation, and other image
manipulation techniques.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

# Import submodules
from . import filtering
from . import fourier
from . import histogram
from . import registration
from . import segmentation
from . import transformations
from . import denoising

# Import legacy modules for backward compatibility
# Note: These imports are done in a try-except block to avoid circular imports
try:
    from . import intensity_transformations
    from . import histogram_equalization
    from . import histogram_matching
    from . import damage_modeling
    from . import image_registration
except ImportError:
    pass  # Ignore circular imports
