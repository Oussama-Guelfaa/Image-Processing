# Configuration Files for Image Processing

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This directory contains configuration files for the Image Processing project. These files are used to configure the project, development environment, and tools.

## Configuration Files

### Project Configuration

- **pyproject.toml**: Configuration file for Python project metadata and build system

### Development Environment

- **temp_settings.json**: Temporary settings for development environment

## Usage

### Project Configuration

The `pyproject.toml` file is used by Python packaging tools like pip and build to determine how to build and install the package. It contains metadata about the project, dependencies, and build system requirements.

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "image-processing"
version = "0.1.0"
description = "Tools for image processing and analysis"
authors = [{name = "Oussama GUELFAA", email = "example@example.com"}]
readme = "README.md"
requires-python = ">=3.8"
```

### Development Environment

The `temp_settings.json` file contains temporary settings for the development environment. These settings are used by development tools and IDEs.

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

## Adding New Configuration Files

When adding new configuration files, follow these guidelines:

1. **File Naming**: Use descriptive names that indicate the purpose of the configuration file.
2. **Documentation**: Include comments explaining what the configuration file does.
3. **README**: Update this README.md file to include information about the new configuration file.
