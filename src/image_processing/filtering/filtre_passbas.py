#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtering

Image filtering operations including low-pass, high-pass, derivative, and Wiener filters.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Filtres passe-bas pour le traitement d'images

Ce script démontre l'application de différents filtres passe-bas sur une image d'ostéoblastes.
Les filtres passe-bas permettent de réduire le bruit et de lisser l'image en supprimant
les hautes fréquences (détails fins). Chaque filtre a des caractéristiques spécifiques
qui le rendent plus ou moins adapté à certaines applications.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian
from skimage.filters.rank import mean, median, minimum, maximum
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Chargement de l'image d'ostéoblastes et conversion en niveaux de gris
# Puis conversion en format 8 bits (0-255) pour les filtres de rang
img = io.imread(get_data_path('osteoblaste.jpg'), as_gray=True)
img = img_as_ubyte(img)

# Création d'un élément structurant circulaire de rayon 3
# Cet élément définit le voisinage utilisé pour les filtres de rang
selem = disk(3)

# Application des différents filtres passe-bas

# Filtre moyenneur - remplace chaque pixel par la moyenne de son voisinage
# Effet: lissage uniforme, mais peut flouter les contours
img_mean = mean(img, selem)

# Filtre médian - remplace chaque pixel par la valeur médiane de son voisinage
# Effet: réduction du bruit tout en préservant mieux les contours que le filtre moyenneur
img_median = median(img, selem)

# Filtre minimum - remplace chaque pixel par la valeur minimale de son voisinage
# Effet: assombrit l'image et élargit les régions sombres (opération d'érosion)
img_min = minimum(img, selem)

# Filtre maximum - remplace chaque pixel par la valeur maximale de son voisinage
# Effet: éclaircit l'image et élargit les régions claires (opération de dilatation)
img_max = maximum(img, selem)

# Filtre gaussien - convolution avec une fonction gaussienne
# Effet: lissage plus naturel que le filtre moyenneur, pondération selon la distance
img_gaussian = gaussian(img, sigma=1)

# Préparation de l'affichage des résultats
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), sharex=True, sharey=True)
ax = axes.ravel()  # Conversion du tableau 2D d'axes en tableau 1D pour faciliter l'accès

# Affichage de l'image originale
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Image originale')

# Affichage des résultats des différents filtres
ax[1].imshow(img_mean, cmap='gray')
ax[1].set_title('Filtre moyenneur')

ax[2].imshow(img_median, cmap='gray')
ax[2].set_title('Filtre médian')

ax[3].imshow(img_min, cmap='gray')
ax[3].set_title('Filtre minimum')

ax[4].imshow(img_max, cmap='gray')
ax[4].set_title('Filtre maximum')

ax[5].imshow(img_gaussian, cmap='gray')
ax[5].set_title('Filtre gaussien')

# Désactivation des axes pour toutes les images
for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
