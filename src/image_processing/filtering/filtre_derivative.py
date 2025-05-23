#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtering

Image filtering operations including low-pass, high-pass, derivative, and Wiener filters.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Filtres dérivatifs pour la détection de contours

Ce script démontre l'utilisation des opérateurs de Prewitt et Sobel pour la détection
de contours dans une image. Ces filtres calculent les dérivées directionnelles (gradients)
de l'image pour mettre en évidence les zones de changement rapide d'intensité.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve2d
from skimage.util import img_as_float
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Chargement de l'image de cellules sanguines et conversion en niveaux de gris
image = img_as_float(io.imread(get_data_path('blood.jpg'), as_gray=True))

# ---------- Opérateur de Prewitt ---------- #
# Le filtre de Prewitt est un opérateur simple pour calculer le gradient d'une image

# Masque vertical (Prewitt) - détecte les contours horizontaux
prewitt_vertical = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

# Masque horizontal (Prewitt) - détecte les contours verticaux
prewitt_horizontal = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]])

# Application des filtres par convolution
prewitt_v_result = convolve2d(image, prewitt_vertical, mode='same', boundary='symm')
prewitt_h_result = convolve2d(image, prewitt_horizontal, mode='same', boundary='symm')

# Calcul de la magnitude du gradient (combinaison des résultats vertical et horizontal)
# Cette combinaison donne la force totale du contour à chaque point
prewitt_edges = np.sqrt(prewitt_v_result**2 + prewitt_h_result**2)

# ---------- Opérateur de Sobel ---------- #
# Le filtre de Sobel est similaire à Prewitt mais donne plus de poids aux pixels centraux
# Ce qui le rend moins sensible au bruit

# Masque vertical (Sobel) - détecte les contours horizontaux
sobel_vertical = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

# Masque horizontal (Sobel) - détecte les contours verticaux
sobel_horizontal = np.array([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]])

# Application des filtres par convolution
sobel_v_result = convolve2d(image, sobel_vertical, mode='same', boundary='symm')
sobel_h_result = convolve2d(image, sobel_horizontal, mode='same', boundary='symm')

# Calcul de la magnitude du gradient (combinaison des résultats vertical et horizontal)
sobel_edges = np.sqrt(sobel_v_result**2 + sobel_h_result**2)

# ---------- Affichage des résultats ---------- #
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Résultat complet de Prewitt (magnitude du gradient)
axes[0, 0].imshow(prewitt_edges, cmap='gray')
axes[0, 0].set_title("Prewitt Complet")
axes[0, 0].axis('off')

# Composante verticale de Prewitt (détection des contours horizontaux)
axes[0, 1].imshow(prewitt_v_result, cmap='gray')
axes[0, 1].set_title("Prewitt Vertical")
axes[0, 1].axis('off')

# Composante horizontale de Prewitt (détection des contours verticaux)
axes[0, 2].imshow(prewitt_h_result, cmap='gray')
axes[0, 2].set_title("Prewitt Horizontal")
axes[0, 2].axis('off')

# Composante verticale de Sobel (détection des contours horizontaux)
axes[1, 0].imshow(sobel_v_result, cmap='gray')
axes[1, 0].set_title("Sobel Vertical")
axes[1, 0].axis('off')

# Composante horizontale de Sobel (détection des contours verticaux)
axes[1, 1].imshow(sobel_h_result, cmap='gray')
axes[1, 1].set_title("Sobel Horizontal")
axes[1, 1].axis('off')

# Résultat complet de Sobel (magnitude du gradient)
axes[1, 2].imshow(sobel_edges, cmap='gray')
axes[1, 2].set_title("Sobel Complet (Horizontal + Vertical)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()