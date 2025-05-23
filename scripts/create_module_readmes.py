#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create README.md files for each module.

This script ensures that each module has a README.md file that explains
its purpose, usage, and functionality.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import glob
import inspect
import importlib
import sys

# Define the README template
README_TEMPLATE = '''# {module_title}

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

{module_description}

## Features

{module_features}

## Usage

{module_usage}

## Examples

{module_examples}

## Functions

{module_functions}

## References

{module_references}
'''

# Define module descriptions and features
MODULE_INFO = {
    'convolution': {
        'title': 'Convolution Operations',
        'description': 'This module provides convolution operations for image processing, including various kernel types.',
        'features': [
            'Convolution with various kernels (Gaussian, Sobel, etc.)',
            'Custom kernel creation',
            'Efficient convolution implementation'
        ],
        'usage': '''
```python
from src.image_processing.convolution import convolve, create_kernel

# Create a Gaussian kernel
kernel = create_kernel('gaussian', size=5, sigma=1.0)

# Apply convolution to an image
result = convolve(image, kernel)
```
''',
        'examples': '''
Example 1: Applying a Gaussian blur
```python
from src.image_processing.convolution import convolve, create_kernel
import matplotlib.pyplot as plt
from skimage import io

# Load an image
image = io.imread('data/image.jpg', as_gray=True)

# Create a Gaussian kernel
kernel = create_kernel('gaussian', size=5, sigma=1.0)

# Apply convolution
blurred = convolve(image, kernel)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(blurred, cmap='gray'), plt.title('Blurred')
plt.tight_layout()
plt.show()
```
''',
        'references': '''
- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
- Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
'''
    },
    'damage_modeling': {
        'title': 'Damage Modeling and Restoration',
        'description': 'This module provides tools for modeling damage to images and restoring them using various techniques.',
        'features': [
            'Damage modeling with various point spread functions (PSF)',
            'Gaussian and motion blur PSF generation',
            'Image restoration using inverse and Wiener filters'
        ],
        'usage': '''
```python
from src.image_processing.damage_modeling import generate_gaussian_psf, apply_damage, wiener_filter

# Generate a Gaussian PSF
psf = generate_gaussian_psf(shape=(32, 32), sigma=3.0)

# Apply damage to an image
damaged = apply_damage(image, psf, noise_level=0.01)

# Restore the image using the Wiener filter
restored = wiener_filter(damaged, psf, k=0.01)
```
''',
        'examples': '''
Example 1: Damage modeling and restoration
```python
from src.image_processing.damage_modeling import generate_gaussian_psf, apply_damage, wiener_filter
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Load an image
image = img_as_float(io.imread('data/image.jpg', as_gray=True))

# Generate a Gaussian PSF
psf = generate_gaussian_psf(shape=(32, 32), sigma=3.0)

# Apply damage to the image
damaged = apply_damage(image, psf, noise_level=0.01)

# Restore the image using the Wiener filter
restored = wiener_filter(damaged, psf, k=0.01)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(damaged, cmap='gray'), plt.title('Damaged')
plt.subplot(133), plt.imshow(restored, cmap='gray'), plt.title('Restored')
plt.tight_layout()
plt.show()
```
''',
        'references': '''
- Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
- Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press.
'''
    },
    # Add more modules as needed
}

def get_module_functions(module_path):
    """
    Get the functions defined in a module.
    
    Parameters
    ----------
    module_path : str
        Path to the module.
        
    Returns
    -------
    str
        Markdown-formatted list of functions with their docstrings.
    """
    # Get the module name from the path
    module_name = os.path.basename(module_path)
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    
    # Import the module
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(module_path)))
        module = importlib.import_module(f"src.image_processing.{os.path.basename(os.path.dirname(module_path))}.{module_name}")
        
        # Get all functions in the module
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                functions.append((name, obj))
        
        # Format the functions as markdown
        if functions:
            result = ''
            for name, func in functions:
                result += f"### `{name}`\n\n"
                if func.__doc__:
                    result += f"{inspect.getdoc(func)}\n\n"
                else:
                    result += "No documentation available.\n\n"
            return result
        else:
            return "No functions found in this module."
    except Exception as e:
        return f"Error getting module functions: {e}"

def create_module_readme(module_dir):
    """
    Create a README.md file for a module.
    
    Parameters
    ----------
    module_dir : str
        Path to the module directory.
    """
    # Get the module name
    module_name = os.path.basename(module_dir)
    
    # Check if the module has a README.md file
    readme_path = os.path.join(module_dir, 'README.md')
    if os.path.exists(readme_path):
        print(f"README.md already exists for {module_name}")
        return
    
    # Get module information
    if module_name in MODULE_INFO:
        module_info = MODULE_INFO[module_name]
    else:
        # Default module information
        module_info = {
            'title': f"{module_name.capitalize()} Module",
            'description': f"This module provides {module_name} functionality for image processing.",
            'features': [f"{module_name.capitalize()} operations for image processing"],
            'usage': f"```python\nfrom src.image_processing.{module_name} import *\n```",
            'examples': "Examples coming soon.",
            'references': "References coming soon."
        }
    
    # Format the features as a bulleted list
    features = '\n'.join([f"- {feature}" for feature in module_info.get('features', [])])
    
    # Get the module functions
    module_files = glob.glob(os.path.join(module_dir, '*.py'))
    functions = ''
    for file_path in module_files:
        if not os.path.basename(file_path).startswith('__'):
            functions += get_module_functions(file_path)
    
    # Create the README content
    readme_content = README_TEMPLATE.format(
        module_title=module_info.get('title', f"{module_name.capitalize()} Module"),
        module_description=module_info.get('description', ''),
        module_features=features,
        module_usage=module_info.get('usage', ''),
        module_examples=module_info.get('examples', ''),
        module_functions=functions,
        module_references=module_info.get('references', '')
    )
    
    # Write the README file
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README.md for {module_name}")

def create_docs_readme(module_dir):
    """
    Create a README.md file for a module in the docs directory.
    
    Parameters
    ----------
    module_dir : str
        Path to the module directory in the docs folder.
    """
    # Get the module name
    module_name = os.path.basename(module_dir)
    
    # Check if the module has a README.md file
    readme_path = os.path.join(module_dir, 'README.md')
    if os.path.exists(readme_path):
        print(f"README.md already exists for docs/{module_name}")
        return
    
    # Get module information
    if module_name in MODULE_INFO:
        module_info = MODULE_INFO[module_name]
    else:
        # Default module information
        module_info = {
            'title': f"{module_name.capitalize()} Module",
            'description': f"This module provides {module_name} functionality for image processing.",
            'features': [f"{module_name.capitalize()} operations for image processing"],
            'usage': f"```python\nfrom src.image_processing.{module_name} import *\n```",
            'examples': "Examples coming soon.",
            'references': "References coming soon."
        }
    
    # Format the features as a bulleted list
    features = '\n'.join([f"- {feature}" for feature in module_info.get('features', [])])
    
    # Create the README content
    readme_content = README_TEMPLATE.format(
        module_title=module_info.get('title', f"{module_name.capitalize()} Module"),
        module_description=module_info.get('description', ''),
        module_features=features,
        module_usage=module_info.get('usage', ''),
        module_examples=module_info.get('examples', ''),
        module_functions="See the source code documentation for details on the functions provided by this module.",
        module_references=module_info.get('references', '')
    )
    
    # Write the README file
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Created README.md for docs/{module_name}")

def main():
    """Main function to create README.md files for each module."""
    print("Creating README.md files for each module...")
    
    # Get all module directories in the src/image_processing directory
    module_dirs = []
    for item in os.listdir('src/image_processing'):
        item_path = os.path.join('src/image_processing', item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            module_dirs.append(item_path)
    
    # Create README.md files for each module
    for module_dir in module_dirs:
        create_module_readme(module_dir)
    
    # Get all module directories in the docs directory
    docs_dirs = []
    for item in os.listdir('docs'):
        item_path = os.path.join('docs', item)
        if os.path.isdir(item_path) and not item.startswith('__'):
            docs_dirs.append(item_path)
    
    # Create README.md files for each module in the docs directory
    for docs_dir in docs_dirs:
        create_docs_readme(docs_dir)
    
    print("README.md files created successfully.")

if __name__ == "__main__":
    main()
