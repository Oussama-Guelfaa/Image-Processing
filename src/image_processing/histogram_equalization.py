"""
Égalisation d'histogramme pour l'amélioration d'images

Ce module implémente l'égalisation d'histogramme, une technique qui transforme
l'image de sorte que son histogramme soit constant (et sa fonction de distribution
cumulative soit linéaire).

L'égalisation d'histogramme est définie par la transformation:
T(x_k) = L * cdf_I(k)

où:
- x_k est la valeur d'intensité k
- L est la valeur maximale d'intensité (255 pour les images 8 bits)
- cdf_I(k) est la fonction de distribution cumulative de l'image

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, img_as_float, img_as_ubyte
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the path utility functions
from src.utils.path_utils import get_data_path

# Nom de l'image à utiliser pour les transformations
OSTEOBLAST_IMAGE = "osteoblaste.jpg"  # Image d'ostéoblastes

def load_image():
    """
    Charge l'image d'ostéoblastes pour les transformations.
    
    Returns:
        ndarray: Image en niveaux de gris
    """
    # Charger l'image
    image_path = get_data_path(OSTEOBLAST_IMAGE)
    image = img_as_float(io.imread(image_path, as_gray=True))
    print(f"Image chargée: {OSTEOBLAST_IMAGE}")
    print(f"Dimensions: {image.shape}")
    print(f"Valeur min: {image.min():.4f}, Valeur max: {image.max():.4f}")
    
    return image

def compute_histogram(image, bins=256):
    """
    Calcule l'histogramme de l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        tuple: (hist, bin_centers) où hist est l'histogramme et bin_centers sont les centres des bins
    """
    # Calculer l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return hist, bin_centers

def visualize_histogram(image, bins=256, title="Histogramme de l'image"):
    """
    Visualise l'histogramme de l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        title (str): Titre du graphique
    """
    # Calculer l'histogramme
    hist, bin_centers = compute_histogram(image, bins)
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer l'histogramme
    plt.bar(bin_centers, hist, width=1/bins, alpha=0.7)
    
    # Ajouter les labels et le titre
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()

def compute_cdf(image, bins=256):
    """
    Calcule la fonction de distribution cumulative (CDF) de l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        tuple: (cdf, bin_centers) où cdf est la fonction de distribution cumulative
               et bin_centers sont les centres des bins
    """
    # Calculer l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculer la CDF
    cdf = hist.cumsum()
    
    # Normaliser la CDF
    cdf = cdf / cdf[-1]
    
    return cdf, bin_centers

def visualize_cdf(image, bins=256, title="Fonction de distribution cumulative"):
    """
    Visualise la fonction de distribution cumulative (CDF) de l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        title (str): Titre du graphique
    """
    # Calculer la CDF
    cdf, bin_centers = compute_cdf(image, bins)
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer la CDF
    plt.plot(bin_centers, cdf, 'b-', linewidth=2)
    
    # Ajouter les labels et le titre
    plt.xlabel('Intensité')
    plt.ylabel('CDF')
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()

def equalize_histogram_builtin(image):
    """
    Applique l'égalisation d'histogramme à l'image en utilisant les fonctions intégrées de scikit-image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        
    Returns:
        ndarray: Image après égalisation d'histogramme
    """
    # Appliquer l'égalisation d'histogramme
    equalized = exposure.equalize_hist(image)
    
    return equalized

def equalize_histogram_custom(image, bins=256):
    """
    Implémentation personnalisée de l'égalisation d'histogramme.
    
    La transformation est définie par:
    T(x_k) = L * cdf_I(k)
    
    où:
    - x_k est la valeur d'intensité k
    - L est la valeur maximale d'intensité (1.0 pour les images normalisées)
    - cdf_I(k) est la fonction de distribution cumulative de l'image
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        ndarray: Image après égalisation d'histogramme
    """
    # Vérifier que l'image est bien en float avec des valeurs entre 0 et 1
    if image.min() < 0 or image.max() > 1:
        print("Attention: L'image doit avoir des valeurs entre 0 et 1. Normalisation appliquée.")
        image = (image - image.min()) / (image.max() - image.min())
    
    # Calculer l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    
    # Calculer la CDF
    cdf = hist.cumsum()
    
    # Normaliser la CDF
    cdf = cdf / cdf[-1]
    
    # Créer la LUT (Look-Up Table)
    lut = np.interp(np.linspace(0, 1, bins), bin_edges[:-1], cdf)
    
    # Appliquer la transformation
    # Pour chaque pixel, on trouve la valeur correspondante dans la LUT
    equalized = np.interp(image.ravel(), np.linspace(0, 1, bins), lut).reshape(image.shape)
    
    return equalized

def visualize_equalization_lut(image, bins=256):
    """
    Visualise la LUT (Look-Up Table) correspondant à l'égalisation d'histogramme.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
    """
    # Calculer l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    
    # Calculer la CDF
    cdf = hist.cumsum()
    
    # Normaliser la CDF
    cdf = cdf / cdf[-1]
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer la LUT (qui est la CDF normalisée)
    plt.plot(bin_edges[:-1], cdf, 'r-', linewidth=2, label='LUT (CDF normalisée)')
    
    # Tracer la ligne diagonale (transformation identité)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Identité')
    
    # Ajouter les labels et le titre
    plt.xlabel('Intensité d\'entrée')
    plt.ylabel('Intensité de sortie')
    plt.title('LUT pour l\'égalisation d\'histogramme')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()

def test_histogram_equalization(image=None):
    """
    Teste l'égalisation d'histogramme sur l'image d'ostéoblastes.
    
    Args:
        image (ndarray, optional): Image à traiter. Si None, charge l'image par défaut.
    """
    if image is None:
        image = load_image()
    
    # Calculer et visualiser l'histogramme de l'image originale
    print("Calcul et visualisation de l'histogramme de l'image originale...")
    visualize_histogram(image, title="Histogramme de l'image originale")
    visualize_cdf(image, title="CDF de l'image originale")
    
    # Appliquer l'égalisation d'histogramme avec les fonctions intégrées
    print("Application de l'égalisation d'histogramme avec les fonctions intégrées...")
    equalized_builtin = equalize_histogram_builtin(image)
    
    # Appliquer l'égalisation d'histogramme avec notre implémentation personnalisée
    print("Application de l'égalisation d'histogramme avec notre implémentation personnalisée...")
    equalized_custom = equalize_histogram_custom(image)
    
    # Visualiser les résultats
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Image égalisée (fonctions intégrées)
    axes[1].imshow(equalized_builtin, cmap='gray')
    axes[1].set_title('Égalisation (builtin)')
    axes[1].axis('off')
    
    # Image égalisée (implémentation personnalisée)
    axes[2].imshow(equalized_custom, cmap='gray')
    axes[2].set_title('Égalisation (custom)')
    axes[2].axis('off')
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    # Visualiser les histogrammes des images égalisées
    visualize_histogram(equalized_builtin, title="Histogramme après égalisation (builtin)")
    visualize_histogram(equalized_custom, title="Histogramme après égalisation (custom)")
    
    # Visualiser les CDFs des images égalisées
    visualize_cdf(equalized_builtin, title="CDF après égalisation (builtin)")
    visualize_cdf(equalized_custom, title="CDF après égalisation (custom)")
    
    # Visualiser la LUT correspondant à l'égalisation d'histogramme
    print("Visualisation de la LUT correspondant à l'égalisation d'histogramme...")
    visualize_equalization_lut(image)
    
    return equalized_builtin, equalized_custom

def main():
    """
    Fonction principale qui démontre l'égalisation d'histogramme.
    """
    # Charger l'image
    image = load_image()
    
    # Tester l'égalisation d'histogramme
    test_histogram_equalization(image)

if __name__ == "__main__":
    main()
