#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to update the headers of test files.

This script adds or updates the file headers in all test files
to ensure consistency and improve searchability.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import re
import glob

# Define the standard header template for test files
TEST_HEADER_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test module for {module_name}.

{description}

Author: Oussama GUELFAA
Date: 01-04-2025
"""
'''

# Define test module descriptions
TEST_MODULE_DESCRIPTIONS = {
    'machine_learning': 'This module tests the machine learning functionality for image processing.',
    'convolution': 'This module tests the convolution operations for image processing.',
    'damage_modeling': 'This module tests the damage modeling and restoration functionality.',
    'denoising': 'This module tests the denoising techniques for image processing.',
    'filtering': 'This module tests the filtering operations for image processing.',
    'fourier': 'This module tests the Fourier transform operations for image processing.',
    'histogram': 'This module tests the histogram operations for image processing.',
    'multiscale': 'This module tests the multiscale analysis techniques for image processing.',
    'registration': 'This module tests the image registration techniques.',
    'segmentation': 'This module tests the segmentation techniques for image processing.',
    'transformations': 'This module tests the intensity transformation techniques for image processing.'
}

def get_test_module_name(file_path):
    """
    Extract the test module name from the file path.
    
    Parameters
    ----------
    file_path : str
        Path to the test file.
        
    Returns
    -------
    str
        Test module name.
    """
    # Extract the module name from the file path
    parts = file_path.split(os.sep)
    if 'tests' in parts:
        idx = parts.index('tests')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    
    # If the module name cannot be determined, use the file name
    file_name = os.path.basename(file_path)
    if file_name.startswith('test_'):
        module_name = file_name[5:]  # Remove 'test_' prefix
    else:
        module_name = file_name
    
    module_name = os.path.splitext(module_name)[0]
    return module_name

def get_test_description(module_name):
    """
    Get the description for a test module.
    
    Parameters
    ----------
    module_name : str
        Name of the test module.
        
    Returns
    -------
    str
        Test module description.
    """
    # Get the description for the test module
    if module_name in TEST_MODULE_DESCRIPTIONS:
        return TEST_MODULE_DESCRIPTIONS[module_name]
    else:
        return 'This module tests the functionality for image processing.'

def update_test_header(file_path):
    """
    Update the header of a test file.
    
    Parameters
    ----------
    file_path : str
        Path to the test file.
    """
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract the module name and description
    module_name = get_test_module_name(file_path)
    description = get_test_description(module_name)
    
    # Create the standard header
    header = TEST_HEADER_TEMPLATE.format(
        module_name=module_name,
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
    
    print(f"Updated header for {file_path}")

def main():
    """Main function to update test file headers."""
    print("Updating test file headers...")
    
    # Get all Python files in the tests directory
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    # Update headers for all test files
    for file_path in test_files:
        update_test_header(file_path)
    
    print(f"Updated headers for {len(test_files)} test files.")

if __name__ == "__main__":
    main()
