# Scripts for Image Processing

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This directory contains utility scripts for the Image Processing project. These scripts are used for various tasks such as project organization, environment setup, and running examples.

## Scripts

### Project Organization

- **reorganize_project.py**: Main script to reorganize the project structure
- **organize_output.py**: Script to organize output files into appropriate directories
- **standardize_headers.py**: Script to standardize file headers across all source files
- **create_module_readmes.py**: Script to create README.md files for each module
- **update_test_headers.py**: Script to update headers for test files

### Environment Setup

- **create_conda_env.sh**: Script to create a Conda environment for the project

### Running Examples

- **run_ml.sh**: Script to run machine learning examples
- **wiener_filter_example.sh**: Script to run Wiener filter examples

### Documentation

- **compile_latex.sh**: Script to compile LaTeX documentation

## Usage

### Project Organization

```bash
# Run the main reorganization script
python scripts/reorganize_project.py

# Run individual organization scripts
python scripts/organize_output.py
python scripts/standardize_headers.py
python scripts/create_module_readmes.py
python scripts/update_test_headers.py
```

### Environment Setup

```bash
# Create a Conda environment for the project
./scripts/create_conda_env.sh
```

### Running Examples

```bash
# Run machine learning examples
./scripts/run_ml.sh

# Run Wiener filter examples
./scripts/wiener_filter_example.sh
```

### Documentation

```bash
# Compile LaTeX documentation
./scripts/compile_latex.sh
```

## Adding New Scripts

When adding new scripts, follow these guidelines:

1. **Script Naming**: Use descriptive names that indicate the purpose of the script.
2. **File Headers**: Include a standard file header with author information and date.
3. **Documentation**: Include docstrings and comments explaining what the script does.
4. **Permissions**: Make the script executable with `chmod +x script_name.py`.
5. **README**: Update this README.md file to include information about the new script.
