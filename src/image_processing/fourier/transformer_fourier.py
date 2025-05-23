#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier

Fourier transform and inverse Fourier transform operations for image processing.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Transformée de Fourier et analyse spectrale d'images

Ce script démontre comment calculer et visualiser la transformée de Fourier d'une image.
Il permet d'observer la décomposition d'une image en ses composantes fréquentielles,
en séparant l'amplitude (intensité des fréquences) et la phase (position des structures).

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.util import img_as_float
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Chargement de l'image de la cornée et conversion en niveaux de gris
image = img_as_float(imread(get_data_path('cornee.png'), as_gray=True))

# Calcul de la transformée de Fourier 2D
fft = np.fft.fft2(image)

# Déplacement des basses fréquences au centre de l'image
# (par défaut, elles sont aux coins)
fft_shift = np.fft.fftshift(fft)

# Extraction de l'amplitude (en échelle logarithmique pour mieux visualiser)
# On ajoute 1 avant de prendre le logarithme pour éviter log(0)
amplitude = np.log(1 + np.abs(fft_shift))

# Extraction de la phase (angles des nombres complexes)
phase = np.angle(fft_shift)

# Affichage des résultats
plt.figure(figsize=(15, 5))
titles = ['Image Originale', 'Spectre d\'Amplitude', 'Spectre de Phase']
images = [image, amplitude, phase]

# Affichage côte à côte des trois composantes
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()