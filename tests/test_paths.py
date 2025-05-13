"""
Test script to verify that the path utility functions work correctly.
"""

import os
import sys
from src.utils.path_utils import get_data_path

def main():
    """
    Test the path utility functions.
    """
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Print the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root directory: {project_root}")
    
    # Print the data directory
    data_dir = os.path.join(project_root, 'data')
    print(f"Data directory: {data_dir}")
    
    # Print the path to an image file
    image_path = get_data_path('cornee.png')
    print(f"Path to cornee.png: {image_path}")
    
    # Check if the image file exists
    if os.path.exists(image_path):
        print(f"The file {image_path} exists.")
    else:
        print(f"The file {image_path} does not exist.")
    
    # List all files in the data directory
    print("\nFiles in the data directory:")
    for file in os.listdir(data_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
