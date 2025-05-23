#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to standardize file headers across all source files.

This script adds or updates the file headers in all Python source files
to ensure consistency and improve searchability.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import re
import glob

# Define the standard header template
HEADER_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{module_name}

{description}

Author: Oussama GUELFAA
Date: 01-04-2025
"""
'''

# Define module descriptions
MODULE_DESCRIPTIONS = {
    'convolution': 'Convolution operations for image processing, including various kernel types.',
    'damage_modeling': 'Tools for modeling damage to images and restoring them using various techniques.',
    'denoising': 'Techniques for removing noise from images, including various filtering methods.',
    'filtering': 'Image filtering operations including low-pass, high-pass, derivative, and Wiener filters.',
    'fourier': 'Fourier transform and inverse Fourier transform operations for image processing.',
    'histogram': 'Histogram equalization, histogram matching, and other histogram-based techniques.',
    'machine_learning': 'Machine learning techniques for image processing and analysis.',
    'multiscale': 'Multiscale analysis techniques including pyramidal decomposition and scale-space decomposition.',
    'registration': 'Image registration techniques including manual point selection and rigid transformation estimation.',
    'segmentation': 'Image segmentation techniques including K-means clustering and other segmentation algorithms.',
    'transformations': 'Intensity transformation techniques including gamma correction and contrast stretching.',
    'utils': 'Utility functions for image processing operations.'
}

def get_module_name(file_path):
    """
    Extract the module name from the file path.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file.
        
    Returns
    -------
    str
        Module name.
    """
    # Extract the module name from the file path
    parts = file_path.split(os.sep)
    if 'image_processing' in parts:
        idx = parts.index('image_processing')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    
    # If the module name cannot be determined, use the file name
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    return module_name

def get_module_description(module_name):
    """
    Get the description for a module.
    
    Parameters
    ----------
    module_name : str
        Name of the module.
        
    Returns
    -------
    str
        Module description.
    """
    # Get the description for the module
    if module_name in MODULE_DESCRIPTIONS:
        return MODULE_DESCRIPTIONS[module_name]
    else:
        return 'Module for image processing operations.'

def standardize_header(file_path):
    """
    Standardize the header of a Python file.
    
    Parameters
    ----------
    file_path : str
        Path to the Python file.
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the module name and description
    module_name = get_module_name(file_path)
    description = get_module_description(module_name)
    
    # Create the standard header
    header = HEADER_TEMPLATE.format(
        module_name=module_name.capitalize(),
        description=description
    )
    
    # Check if the file already has a header
    header_pattern = r'^#!/usr/bin/env python3\n# -\*- coding: utf-8 -\*-\n""".*?"""'
    if re.match(header_pattern, content, re.DOTALL):
        # Replace the existing header
        new_content = re.sub(header_pattern, header.strip(), content, flags=re.DOTALL)
    else:
        # Add the header to the beginning of the file
        new_content = header + '\n' + content
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(new_content)
    
    print(f"Standardized header for {file_path}")

def main():
    """Main function to standardize file headers."""
    print("Standardizing file headers...")
    
    # Get all Python files in the src directory
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    # Standardize headers for all Python files
    for file_path in python_files:
        standardize_header(file_path)
    
    print(f"Standardized headers for {len(python_files)} files.")

if __name__ == "__main__":
    main()
