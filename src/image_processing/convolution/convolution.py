#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convolution

Convolution operations for image processing, including various kernel types.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Convolution d'images et application de différents filtres

Ce script démontre l'application de différents filtres de convolution sur une image
de cellules sanguines pour illustrer les effets de lissage et de détection de contours.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve2d
import numpy as np
from skimage.util import img_as_float
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Chargement de l'image originale en niveaux de gris
# On utilise une image de cellules sanguines pour notre démonstration
image = img_as_float(io.imread(get_data_path('blood.jpg'), as_gray=True))

# Définition des différents noyaux (kernels) de convolution

# Filtre moyenneur 3x3 - lisse l'image en remplaçant chaque pixel par la moyenne de son voisinage
# Effet: réduction du bruit, mais aussi des détails fins
mean_kernel = (1/9) * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

# Filtre laplacien - accentue les changements rapides d'intensité (contours)
# Effet: détection des bords et amélioration de la netteté
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

# Filtre gaussien - lissage plus naturel que le filtre moyenneur
# Effet: réduction du bruit tout en préservant mieux les structures importantes
gaussian_kernel = (1/16) * np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]])

# Application des filtres par convolution 2D
# Le mode 'same' conserve les dimensions de l'image originale
# La limite 'symm' gère les bords en reflétant les pixels
mean_filtered = convolve2d(image, mean_kernel, mode='same', boundary='symm')
laplacian_filtered = convolve2d(image, laplacian_kernel, mode='same', boundary='symm')
gaussian_filtered = convolve2d(image, gaussian_kernel, mode='same', boundary='symm')

# Affichage des résultats pour comparaison visuelle
fig, axes = plt.subplots(1, 4, figsize=(15, 6))

# Image originale comme référence
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Image Originale")
axes[0].axis('off')

# Résultat du filtre moyenneur
axes[1].imshow(mean_filtered, cmap='gray')
axes[1].set_title("Filtre Moyenneur")
axes[1].axis('off')

# Résultat du filtre laplacien
axes[2].imshow(laplacian_filtered, cmap='gray')
axes[2].set_title("Filtre Laplacien")
axes[2].axis('off')

# Résultat du filtre gaussien
axes[3].imshow(gaussian_filtered, cmap='gray')
axes[3].set_title("Filtre Gaussien")
axes[3].axis('off')

plt.tight_layout()
plt.show()
