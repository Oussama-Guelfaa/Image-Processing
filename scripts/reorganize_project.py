#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reorganize the project structure.

This script runs all the reorganization tasks to improve the project structure,
including organizing output files, standardizing file headers, and creating
README.md files for each module.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import subprocess
import sys

def run_script(script_path):
    """
    Run a Python script.
    
    Parameters
    ----------
    script_path : str
        Path to the script to run.
        
    Returns
    -------
    bool
        True if the script ran successfully, False otherwise.
    """
    print(f"\n=== Running {script_path} ===\n")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist."""
    print("\n=== Creating necessary directories ===\n")
    
    directories = [
        'scripts',
        'output',
        'docs',
        'tests',
        'data'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def main():
    """Main function to reorganize the project structure."""
    print("Starting project reorganization...")
    
    # Create necessary directories
    create_directories()
    
    # Run the reorganization scripts
    scripts = [
        'scripts/organize_output.py',
        'scripts/standardize_headers.py',
        'scripts/create_module_readmes.py'
    ]
    
    success = True
    for script in scripts:
        if not run_script(script):
            success = False
            print(f"Failed to run {script}")
    
    if success:
        print("\nProject reorganization completed successfully.")
    else:
        print("\nProject reorganization completed with errors.")

if __name__ == "__main__":
    main()
