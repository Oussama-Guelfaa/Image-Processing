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

# Compiler le document LaTeX (deux fois pour les références)
pdflatex image_processing_documentation.tex
pdflatex image_processing_documentation.tex

# Vérifier si la compilation a réussi
if [ -f "image_processing_documentation.pdf" ]; then
    echo "Compilation réussie ! Le document PDF a été créé."
    
    # Ouvrir le PDF (fonctionne sur macOS, Linux et Windows avec WSL)
    if [ "$(uname)" == "Darwin" ]; then
        open image_processing_documentation.pdf
    elif [ "$(uname)" == "Linux" ]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open image_processing_documentation.pdf
        else
            echo "Le document PDF est disponible à: $(pwd)/image_processing_documentation.pdf"
        fi
    else
        echo "Le document PDF est disponible à: $(pwd)/image_processing_documentation.pdf"
    fi
else
    echo "La compilation a échoué. Veuillez vérifier les erreurs ci-dessus."
fi

# Nettoyer les fichiers temporaires
rm -f *.aux *.log *.toc *.out
