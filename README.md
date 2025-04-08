# Image Processing Project

This project provides tools for image processing and analysis, with a focus on Fourier transforms and frequency domain filtering. It also includes graph visualization capabilities.

## Project Structure

```
PythonProject/
├── data/                  # Image files and other data
├── docs/                  # Documentation
├── examples/              # Example scripts
│   └── application.py     # Example application for frequency domain filtering
├── src/                   # Source code
│   ├── image_processing/  # Image processing modules
│   │   ├── Convolution.py
│   │   ├── Transformer_fourier_inverse.py
│   │   ├── aliasing_effect.py
│   │   ├── filtering_hp_lp.py
│   │   ├── filtre_derivative.py
│   │   ├── filtre_passbas.py
│   │   ├── filtre_passhaut.py
│   │   └── transformer_fourier.py
│   └── visualization/     # Visualization modules
│       └── draw.py        # Graph visualization
├── main.py                # Main entry point
└── README.md              # This file
```

## Usage

You can run the project using the main.py script:

```bash
# Run Fourier transform analysis
python main.py fourier --image data/cornee.png

# Apply a filter to an image
python main.py filter --image data/cornee.png --type lowpass --cutoff 30

# Generate a graph visualization
python main.py graph
```

## Examples

Check the examples directory for sample applications:

```bash
# Run the example application
python examples/application.py
```

## Dependencies

- NumPy
- Matplotlib
- scikit-image
- graphviz (for graph visualization)
