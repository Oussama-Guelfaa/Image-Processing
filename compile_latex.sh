#!/bin/bash

# Script to compile the LaTeX documentation

# Change to the latex_documentation directory
cd latex_documentation

# Run pdflatex twice to resolve references
pdflatex main.tex
pdflatex main.tex

# Clean up auxiliary files
rm -f *.aux *.log *.toc *.out

echo "Documentation compiled successfully. Output file: latex_documentation/main.pdf"
