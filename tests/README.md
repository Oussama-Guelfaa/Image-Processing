# Tests for Image Processing

## Author: Oussama GUELFAA
## Date: 01-04-2025

## Introduction

This directory contains tests for the Image Processing project. The tests are organized by module, with each module having its own subdirectory.

## Test Structure

```
tests/
├── __init__.py                  # Package initialization
├── README.md                    # This file
├── test_adaptive_median.py      # Tests for adaptive median filter
├── test_denoising.py            # Tests for denoising techniques
├── test_paths.py                # Tests for path utilities
├── convolution/                 # Tests for convolution operations
├── damage_modeling/             # Tests for damage modeling and restoration
├── denoising/                   # Tests for denoising techniques
├── filtering/                   # Tests for filtering operations
├── fourier/                     # Tests for Fourier transform operations
├── histogram/                   # Tests for histogram operations
├── machine_learning/            # Tests for machine learning functionality
│   ├── test_kimia216.py         # Tests for Kimia216 dataset
│   ├── test_ml.py               # Tests for machine learning functionality
│   └── test_ml_simple.py        # Simplified tests for machine learning
├── multiscale/                  # Tests for multiscale analysis
├── registration/                # Tests for image registration
│   ├── test_image_registration.py  # Tests for image registration
│   └── test_manual_registration.py # Tests for manual registration
├── segmentation/                # Tests for segmentation techniques
└── transformations/             # Tests for intensity transformations
```

## Running Tests

You can run all tests using the following command:

```bash
python -m unittest discover tests
```

Or you can run tests for a specific module:

```bash
python -m unittest discover tests/machine_learning
```

Or you can run a specific test file:

```bash
python -m unittest tests/machine_learning/test_ml.py
```

## Writing Tests

When writing tests, follow these guidelines:

1. **Test File Naming**: Name test files with the prefix `test_` followed by the name of the module or functionality being tested.
2. **Test Class Naming**: Name test classes with the prefix `Test` followed by the name of the class or functionality being tested.
3. **Test Method Naming**: Name test methods with the prefix `test_` followed by a description of what is being tested.
4. **Documentation**: Include docstrings for test classes and methods explaining what is being tested.
5. **Assertions**: Use appropriate assertions for the type of test being performed.
6. **Setup and Teardown**: Use `setUp` and `tearDown` methods for common setup and cleanup operations.
7. **Test Independence**: Ensure that tests are independent of each other and can be run in any order.

## Example Test

```python
import unittest
from src.image_processing.transformations import apply_gamma_correction

class TestGammaCorrection(unittest.TestCase):
    """Tests for gamma correction functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.image = np.ones((10, 10))
    
    def test_gamma_correction(self):
        """Test that gamma correction works as expected."""
        gamma = 0.5
        corrected = apply_gamma_correction(self.image, gamma)
        self.assertEqual(corrected.shape, self.image.shape)
        self.assertTrue(np.allclose(corrected, self.image ** gamma))
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.image = None

if __name__ == '__main__':
    unittest.main()
```

## Test Coverage

To check test coverage, you can use the `coverage` package:

```bash
# Install coverage
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests

# Generate coverage report
coverage report -m
```

## Continuous Integration

Tests are automatically run on each commit using GitHub Actions. See the `.github/workflows` directory for the CI configuration.

## References

- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [coverage documentation](https://coverage.readthedocs.io/)
