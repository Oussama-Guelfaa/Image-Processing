#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier

Fourier transform and inverse Fourier transform operations for image processing.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Transformée de Fourier inverse et importance de l'amplitude et de la phase

Ce script explore l'importance relative de l'amplitude et de la phase dans la
transformée de Fourier d'une image. Il démontre comment la reconstruction
d'une image est affectée lorsqu'on utilise uniquement l'information d'amplitude
ou uniquement l'information de phase.

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

# Déplacement des basses fréquences au centre
fft_shift = np.fft.fftshift(fft)

# Extraction de l'amplitude et de la phase du spectre
amplitude = np.abs(fft_shift)  # Module du nombre complexe
phase = np.angle(fft_shift)    # Argument (angle) du nombre complexe

# 1. Reconstruction complète avec amplitude et phase
# C'est l'inverse exact de la transformée de Fourier originale
recon_full = np.fft.ifft2(np.fft.ifftshift(fft_shift)).real

# 2. Reconstruction en utilisant uniquement l'amplitude
# On remplace la phase par zéro (exp(0j) = 1)
amp_only = amplitude * np.exp(1j * 0)
recon_amp = np.fft.ifft2(np.fft.ifftshift(amp_only)).real

# 3. Reconstruction en utilisant uniquement la phase
# On utilise une amplitude constante (m = 1) pour toutes les fréquences
complex_phase = np.exp(1j * phase)  # Amplitude unitaire avec la phase originale
recon_phase = np.fft.ifft2(np.fft.ifftshift(complex_phase)).real

# Préparation pour l'affichage des résultats
titles = ['Reconstruction complète', 'Reconstruction avec amplitude seule', 'Reconstruction avec phase seule']
images = [recon_full, recon_amp, recon_phase]

# Affichage des trois reconstructions côte à côte
plt.figure(figsize=(15, 5))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()