#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__init__.py

Module for image processing operations.

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
from . import damage_modeling
from . import convolution

# Import legacy modules for backward compatibility
# Note: These imports are done in a try-except block to avoid circular imports
try:
    from . import intensity_transformations
    from . import histogram_equalization
    from . import histogram_matching
    from . import image_registration
except ImportError:
    pass  # Ignore circular imports
