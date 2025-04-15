#!/bin/bash

# Script pour compiler le document LaTeX

# Vérifier si pdflatex est installé
if ! command -v pdflatex &> /dev/null
then
    echo "pdflatex n'est pas installé. Veuillez installer BasicTeX ou une autre distribution LaTeX."
    echo "Pour macOS: brew install --cask basictex"
    echo "Pour Ubuntu: sudo apt-get install texlive-latex-base"
    exit 1
fi

# Se placer dans le répertoire du script
cd "$(dirname "$0")"

# Compiler le document LaTeX en anglais (deux fois pour les références)
pdflatex image_processing_documentation_en.tex
pdflatex image_processing_documentation_en.tex

# Vérifier si la compilation a réussi
if [ -f "image_processing_documentation_en.pdf" ]; then
    echo "Compilation successful! The PDF document has been created."

    # Ouvrir le PDF (fonctionne sur macOS, Linux et Windows avec WSL)
    if [ "$(uname)" == "Darwin" ]; then
        open image_processing_documentation_en.pdf
    elif [ "$(uname)" == "Linux" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open image_processing_documentation_en.pdf
        else
            echo "The PDF document is available at: $(pwd)/image_processing_documentation_en.pdf"
        fi
    else
        echo "The PDF document is available at: $(pwd)/image_processing_documentation_en.pdf"
    fi
else
    echo "Compilation failed. Please check the errors above."
fi

# Uncomment the following lines to compile the French version (requires additional packages)
# # Compiler le document LaTeX en français (deux fois pour les références)
# pdflatex image_processing_documentation.tex
# pdflatex image_processing_documentation.tex

# Nettoyer les fichiers temporaires
rm -f *.aux *.log *.toc *.out
