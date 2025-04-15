# Image Processing Tools

This project provides tools for image processing and analysis, including intensity transformations, histogram equalization, histogram matching, and more. It is designed to be easy to use and educational.

## Project Structure

```
Image-Processing/
├── data/                  # Image files and other data
├── docs/                  # Documentation
├── examples/              # Example scripts
├── latex_documentation/   # LaTeX documentation
├── src/                   # Source code
│   ├── image_processing/  # Image processing modules
│   │   ├── histogram_equalization.py  # Histogram equalization
│   │   ├── histogram_matching.py      # Histogram matching
│   │   ├── intensity_transformations.py # Gamma correction and contrast stretching
│   │   └── ... (other modules)
│   └── cli.py             # Command-line interface
├── main.py                # Main entry point
├── setup.py               # Package installation configuration
└── README.md              # This file
```

## Installation

You can install this package directly from GitHub:

```bash
# Install from GitHub
pip install git+https://github.com/yourusername/Image-Processing.git
```

Or you can clone the repository and install it locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-Processing.git
cd Image-Processing

# Install the package
pip install -e .
```

## Usage

After installation, you can use the `imgproc` command-line tool:

```bash
# Apply gamma correction to an image
imgproc intensity --method gamma --gamma 0.5 --image path/to/image.jpg

# Apply histogram equalization
imgproc histogram --method custom --bins 256 --image path/to/image.jpg

# Apply histogram matching
imgproc matching --method custom --peak1 0.3 --peak2 0.7 --image path/to/image.jpg
```

You can also run the project using the main.py script without installation:

```bash
# Apply intensity transformations
python main.py intensity --method gamma --gamma 0.5 --image path/to/image.jpg

# Apply histogram equalization
python main.py histogram --method custom --bins 256 --image path/to/image.jpg

# Apply histogram matching
python main.py matching --method custom --peak1 0.3 --peak2 0.7 --image path/to/image.jpg
```

## Examples

Check the examples directory for sample applications.

## Dependencies

- NumPy
- Matplotlib
- scikit-image
