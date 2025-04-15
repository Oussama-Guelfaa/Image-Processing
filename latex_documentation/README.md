# LaTeX Documentation for the Image Processing Project

This folder contains the LaTeX documentation for the image processing project. The document explains in detail the theory and implementation of the following techniques:

1. Intensity transformations (gamma correction and contrast stretching)
2. Histogram equalization
3. Histogram matching

## Prerequisites

To compile the LaTeX document, you need a LaTeX distribution like BasicTeX or TeX Live.

### Installing BasicTeX

#### macOS
```bash
brew install --cask basictex
```

#### Ubuntu/Debian
```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra
```

#### Windows
Download and install MiKTeX or TeX Live from their official websites.

## Compiling the Document

### Automatic Method (Recommended)

Use the provided shell script:

```bash
./compile_latex.sh
```

This script checks if pdflatex is installed, compiles the document, and opens the resulting PDF.

### Manual Method

If you prefer to compile manually, run the following commands:

```bash
pdflatex image_processing_documentation_en.tex
pdflatex image_processing_documentation_en.tex
```

The command is executed twice to ensure that all references are properly resolved.

## Document Content

The generated PDF document contains:

- An introduction to image processing techniques
- The theoretical foundations of each technique
- The mathematical formulas used
- Python code excerpts showing the implementation
- Explanations of the parameters and their influence on the results

## Author

Oussama GUELFAA
Date: 01-04-2025
