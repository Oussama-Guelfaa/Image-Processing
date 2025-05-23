#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation

Image segmentation techniques including K-means clustering and other segmentation algorithms.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Segmentation d'images par seuillage manuel et clustering k-means

Ce script implémente deux méthodes de segmentation d'images basées sur les histogrammes:
1. Seuillage manuel (thresholding) - segmentation basée sur un seuil d'intensité
2. Clustering k-means - segmentation par regroupement des pixels en k classes

Ces méthodes sont appliquées sur des images de cellules pour démontrer leur efficacité
dans la séparation des objets d'intérêt du fond.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters
from skimage.util import img_as_float
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path, get_project_root

# Nom de l'image à utiliser pour la segmentation
CELLS_IMAGE = "blood.jpg"  # Image de cellules sanguines

def load_image():
    """
    Charge l'image de cellules pour la segmentation.

    Returns:
        ndarray: Image en niveaux de gris
    """
    # Charger l'image de cellules
    cells = img_as_float(io.imread(get_data_path(CELLS_IMAGE), as_gray=True))
    print(f"Image de cellules chargée: {CELLS_IMAGE}")

    return cells

def plot_histogram(image, ax=None, bins=256):
    """
    Affiche l'histogramme d'une image en niveaux de gris.

    Args:
        image (ndarray): Image en niveaux de gris
        ax (matplotlib.axes.Axes, optional): Axes pour l'affichage
        bins (int, optional): Nombre de bins pour l'histogramme

    Returns:
        tuple: (hist, bin_centers) - Valeurs de l'histogramme et centres des bins
    """
    if ax is None:
        _, ax = plt.subplots()

    # Calcul de l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Affichage de l'histogramme
    ax.plot(bin_centers, hist, lw=2)
    ax.set_title('Histogramme')
    ax.set_xlabel('Intensité des pixels')
    ax.set_ylabel('Nombre de pixels')

    return hist, bin_centers

def manual_thresholding(image, threshold=0.5):
    """
    Applique un seuillage manuel sur une image en niveaux de gris.

    Args:
        image (ndarray): Image en niveaux de gris
        threshold (float, optional): Valeur de seuil.

    Returns:
        ndarray: Image binaire après seuillage
    """
    # Application du seuillage
    binary = image > threshold

    return binary

def autothresh(image):
    """
    Automatic threshold method

    Args:
        image (ndarray): Image to segment

    Returns:
        float: Threshold value
    """
    # Initialisation avec la moyenne des valeurs min et max
    s = 0.5 * (np.min(image) + np.max(image))
    done = False

    # Boucle itérative pour trouver le seuil optimal
    while not done:
        # Séparer l'image en deux classes (supérieure et inférieure au seuil)
        B = image >= s
        # Calculer le nouveau seuil comme la moyenne des moyennes des deux classes
        sNext = 0.5 * (np.mean(image[B]) + np.mean(image[~B]))
        # Vérifier la condition d'arrêt
        done = abs(s - sNext) < 0.5/255  # Seuil de convergence
        # Mettre à jour le seuil
        s = sNext

    print(f"Seuil déterminé par méthode itérative: {s:.3f}")
    return s

def kmeans_clustering(image, n_clusters=2):
    """
    Applique l'algorithme k-means pour segmenter une image en niveaux de gris.

    Args:
        image (ndarray): Image en niveaux de gris
        n_clusters (int, optional): Nombre de clusters (classes)

    Returns:
        ndarray: Image segmentée où chaque pixel a la valeur de son cluster
    """
    # Reshape de l'image pour l'algorithme k-means
    # Conversion en vecteur de caractéristiques (ici, juste l'intensité)
    X = image.reshape((-1, 1))

    # Application de l'algorithme k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Récupération des labels (clusters) pour chaque pixel
    labels = kmeans.labels_

    # Reshape des labels pour obtenir une image segmentée
    segmented = labels.reshape(image.shape)

    return segmented

def main():
    """
    Fonction principale qui charge l'image et applique les méthodes de segmentation.
    """
    # Chargement de l'image
    cells = load_image()

    # Calcul des seuils avec différentes méthodes
    # 1. Méthode itérative (algorithme 6)
    s_auto = autothresh(cells)

    # 2. Méthode d'Otsu (scikit-image)
    s_otsu = filters.threshold_otsu(cells)
    print(f"Seuil déterminé par méthode d'Otsu: {s_otsu:.3f}")

    # 3. Seuillage manuel avec une valeur fixe
    s_manual = 0.5
    print(f"Seuil manuel: {s_manual:.3f}")

    # Application des seuils
    binary_auto = cells > s_auto
    binary_otsu = cells > s_otsu
    binary_manual = manual_thresholding(cells, s_manual)

    # 4. K-means clustering
    kmeans_result = kmeans_clustering(cells, n_clusters=3)

    # Affichage de tous les résultats dans une seule figure avec plusieurs sous-graphiques
    fig = plt.figure(figsize=(15, 12))

    # Définition de la grille de sous-graphiques (3 lignes, 3 colonnes)
    gs = fig.add_gridspec(3, 3)

    # 1. Image originale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cells, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 2. Histogramme avec les seuils
    ax2 = fig.add_subplot(gs[0, 1:3])
    hist, bin_edges = np.histogram(cells.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax2.plot(bin_centers, hist, lw=2)
    ax2.axvline(x=s_auto, color='r', linestyle='--', label=f'Auto threshold: {s_auto:.3f}')
    ax2.axvline(x=s_otsu, color='g', linestyle='--', label=f'Otsu threshold: {s_otsu:.3f}')
    ax2.axvline(x=s_manual, color='b', linestyle='--', label=f'Manual threshold: {s_manual:.3f}')
    ax2.set_title('Histogram with Thresholds')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Pixel Count')
    ax2.legend()

    # 3. Seuillage manuel
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(binary_manual, cmap='gray')
    ax3.set_title(f'Manual Thresholding (t={s_manual:.3f})')
    ax3.axis('off')

    # 4. Seuillage automatique (méthode itérative)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(binary_auto, cmap='gray')
    ax4.set_title(f'Automatic Thresholding (t={s_auto:.3f})')
    ax4.axis('off')

    # 5. Seuillage d'Otsu
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(binary_otsu, cmap='gray')
    ax5.set_title(f'Otsu Thresholding (t={s_otsu:.3f})')
    ax5.axis('off')

    # 6. K-means clustering
    ax6 = fig.add_subplot(gs[2, 0:3])
    cmap = plt.cm.viridis
    ax6.imshow(kmeans_result, cmap=cmap)
    ax6.set_title(f'K-means Clustering (k=3)')
    ax6.axis('off')

    # Ajustement de la mise en page et affichage
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
