"""
Filtrage passe-haut et passe-bas dans le domaine fréquentiel

Ce script démontre comment appliquer des filtres passe-haut et passe-bas
à une image dans le domaine fréquentiel en utilisant la transformée de Fourier.
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

# Calcul de la transformée de Fourier 2D de l'image
# La FFT nous permet de passer du domaine spatial au domaine fréquentiel
fft = np.fft.fft2(image)

# Déplacement des basses fréquences au centre de l'image
# C'est plus intuitif pour appliquer nos filtres
fft_shift = np.fft.fftshift(fft)

def LowPassFilter(spectrum, cut):
    """
    Filtre passe-bas qui ne conserve que les basses fréquences

    Paramètres:
        spectrum: Le spectre de fréquences de l'image
        cut: La taille de la fenêtre de coupure (plus grande = plus de fréquences conservées)
    """
    X, Y = spectrum.shape
    # Création d'un masque qui ne garde que les fréquences centrales (basses fréquences)
    mask = np.zeros((X, Y), "int")
    mx, my = X//2, Y//2
    # On met à 1 uniquement la région centrale
    mask[mx-cut:mx+cut, my-cut:my+cut] = 1
    return spectrum * mask

def HighPassFilter(spectrum, cut):
    """
    Filtre passe-haut qui ne conserve que les hautes fréquences

    Paramètres:
        spectrum: Le spectre de fréquences de l'image
        cut: La taille de la fenêtre de coupure (plus grande = plus de fréquences supprimées)
    """
    X, Y = spectrum.shape
    # Création d'un masque qui supprime les fréquences centrales (basses fréquences)
    mask = np.ones((X, Y), "int")
    mx, my = X//2, Y//2
    # On met à 0 uniquement la région centrale
    mask[mx-cut:mx+cut, my-cut:my+cut] = 0
    return spectrum * mask

# Application des filtres au spectre de fréquences
# La valeur 30 détermine la taille de la fenêtre de coupure
low = LowPassFilter(fft_shift, 30)
high = HighPassFilter(fft_shift, 30)

# Transformée de Fourier inverse pour revenir au domaine spatial
# On utilise .real pour ne garder que la partie réelle du résultat
img_low = np.fft.ifft2(np.fft.ifftshift(low)).real
img_high = np.fft.ifft2(np.fft.ifftshift(high)).real

# Préparation pour l'affichage
images = [img_low, img_high]
titles = ['Reconstruction après filtrage passe-bas', 'Reconstruction après filtrage passe-haut']

# Affichage des résultats côte à côte
plt.figure(figsize=(10, 10))
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()