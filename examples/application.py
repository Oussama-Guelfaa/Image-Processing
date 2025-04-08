"""
Application de filtrage dans le domaine fréquentiel pour l'analyse de structures cellulaires

Ce script démontre comment utiliser la transformée de Fourier et le filtrage gaussien
pour analyser les motifs périodiques dans une image de cornée. Cette technique permet
d'identifier les fréquences dominantes qui correspondent à la taille et à l'espacement
des cellules dans l'image.
"""

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
from skimage.filters import gaussian
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Chargement de l'image de la cornée et conversion en niveaux de gris
cornea = img_as_float(imread(get_data_path('cornee.png'), as_gray=True))

# Calcul de la transformée de Fourier pour passer dans le domaine fréquentiel
fft = np.fft.fft2(cornea)  # Transformée de Fourier 2D
fft_shift = np.fft.fftshift(fft)  # Déplacement des basses fréquences au centre
amplitude = np.abs(fft_shift)  # Extraction de l'amplitude du spectre

# Application d'un filtre gaussien sur le spectre d'amplitude
# Ce filtrage permet de lisser le spectre et de mieux visualiser les pics de fréquence
blurred = gaussian(amplitude, sigma=5)

# Visualisation des résultats
plt.figure(figsize=(10, 5))

# Affichage du spectre d'amplitude filtré (en échelle logarithmique pour mieux voir les détails)
plt.subplot(1, 2, 1)
plt.imshow(np.log(1 + blurred), cmap='gray')
plt.title('Spectre d\'amplitude filtré')
plt.axis('off')

# Affichage d'une coupe horizontale au milieu du spectre
# Cela permet de visualiser les pics de fréquence qui correspondent aux structures périodiques
plt.subplot(1, 2, 2)
midY = blurred.shape[0] // 2  # Ligne centrale de l'image
plt.plot(np.log(1 + blurred[midY, :]))
plt.title('Observation des pics et fréquences cellulaires')

plt.tight_layout()
plt.show()