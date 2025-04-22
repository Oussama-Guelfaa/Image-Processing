"""
Filtres passe-haut pour le traitement d'images

Ce script démontre l'application de différents filtres passe-haut sur des images de cellules.
Les filtres passe-haut permettent de mettre en évidence les contours et les détails fins
en supprimant les basses fréquences. Deux approches sont comparées : la soustraction
d'un filtre passe-bas et l'application directe d'un filtre laplacien.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import gaussian, laplace
from skimage.util import img_as_float
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Charger les deux images
img1 = img_as_float(io.imread(get_data_path('blood.jpg'),as_gray=True))
img2 = img_as_float(io.imread(get_data_path('osteoblaste.jpg'),as_gray=True))

# On applique d'abord un filtre passe-bas (ici gaussian)
low_pass_img1 = gaussian(img1, sigma=1)
low_pass_img2 = gaussian(img2, sigma=1)

# Calculer le filtre passe-haut en soustrayant le low-pass à l'image originale
high_pass_img1 = img1 - low_pass_img1
high_pass_img2 = img2 - low_pass_img2

# --------- Filtrage Passe-haut Laplacien ---------

# Utilisation directe du filtre laplacien de skimage
laplacian_img1 = laplace(img1)
laplacian_img2 = laplace(img2)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Image originale 1
axes[0, 0].imshow(img1, cmap='gray')
axes[0, 0].set_title('Originale - blood cells')
axes[0, 0].axis('off')

# High-pass manuel Image 1
axes[0, 1].imshow(high_pass_img1, cmap='gray')
axes[0, 1].set_title('High-pass (f - LP(f))')
axes[0, 1].axis('off')

# Laplacian Image 1
axes[0, 2].imshow(laplacian_img1, cmap='gray')
axes[0, 2].set_title('Laplacian High-pass')
axes[0, 2].axis('off')

# Image originale 2
axes[1, 0].imshow(img2, cmap='gray')
axes[1, 0].set_title('Originale - osteoblast cells')
axes[1, 0].axis('off')

# High-pass manuel Image 2
axes[1, 1].imshow(high_pass_img2, cmap='gray')
axes[1, 1].set_title('High-pass (f - LP(f))')
axes[1, 1].axis('off')

# Laplacian Image 2
axes[1, 2].imshow(laplacian_img2, cmap='gray')
axes[1, 2].set_title('Laplacian High-pass')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()