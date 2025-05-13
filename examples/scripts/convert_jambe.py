#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert jambe.tif to jambe.png

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import numpy as np
from PIL import Image

def main():
    """Convert jambe.tif to jambe.png."""
    try:
        # Open the TIFF file with PIL
        image_path = os.path.join("data", "jambe.tif")
        img = Image.open(image_path)
        
        # Save as PNG
        output_path = os.path.join("data", "jambe.png")
        img.save(output_path)
        
        print(f"Successfully converted {image_path} to {output_path}")
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    main()
