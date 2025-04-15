# Documentation LaTeX du Projet de Traitement d'Images

Ce dossier contient la documentation LaTeX du projet de traitement d'images. Le document explique en détail la théorie et l'implémentation des techniques suivantes :

1. Transformations d'intensité (correction gamma et étirement de contraste)
2. Égalisation d'histogramme
3. Appariement d'histogramme

## Prérequis

Pour compiler le document LaTeX, vous avez besoin d'une distribution LaTeX comme BasicTeX ou TeX Live.

### Installation de BasicTeX

#### macOS
```bash
brew install --cask basictex
```

#### Ubuntu/Debian
```bash
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra
```

#### Windows
Téléchargez et installez MiKTeX ou TeX Live depuis leurs sites officiels.

## Compilation du document

### Méthode automatique (recommandée)

Utilisez le script shell fourni :

```bash
./compile_latex.sh
```

Ce script vérifie si pdflatex est installé, compile le document et ouvre le PDF résultant.

### Méthode manuelle

Si vous préférez compiler manuellement, exécutez les commandes suivantes :

```bash
pdflatex image_processing_documentation.tex
pdflatex image_processing_documentation.tex
```

La commande est exécutée deux fois pour assurer que toutes les références sont correctement résolues.

## Contenu du document

Le document PDF généré contient :

- Une introduction aux techniques de traitement d'images
- Les fondements théoriques de chaque technique
- Les formules mathématiques utilisées
- Des extraits de code Python montrant l'implémentation
- Des explications sur les paramètres et leur influence sur les résultats

## Auteur

Oussama GUELFAA
Date: 01-04-2025
