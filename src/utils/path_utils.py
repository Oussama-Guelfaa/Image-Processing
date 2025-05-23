#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path_utils

Module for image processing operations.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Utility functions for handling file paths.
"""

import os

def get_project_root():
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: The absolute path to the project root directory.
    """
    # The project root is two directories up from this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def get_data_path(filename):
    """
    Get the absolute path to a file in the data directory.
    
    Args:
        filename (str): The name of the file in the data directory.
        
    Returns:
        str: The absolute path to the file.
    """
    return os.path.join(get_project_root(), 'data', filename)
