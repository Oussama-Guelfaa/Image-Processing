#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Histogram

Histogram equalization, histogram matching, and other histogram-based techniques.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Appariement d'histogramme (Histogram Matching)

Ce module implémente l'appariement d'histogramme, une technique qui transforme
l'image de sorte que son histogramme corresponde à un histogramme modèle.

L'appariement d'histogramme est défini par la transformation:
x2 = cdf2^(-1)(cdf1(x1))

où:
- x1 est la valeur d'intensité dans l'image source
- x2 est la valeur d'intensité correspondante dans l'image cible
- cdf1 est la fonction de distribution cumulative de l'image source
- cdf2 est la fonction de distribution cumulative de l'histogramme modèle

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
PHOBOS_IMAGE = "phobos.jpg"  # Image de Phobos (lune de Mars)

def load_image():
    """
    Charge l'image de Phobos pour les transformations.
    
    Returns:
        ndarray: Image en niveaux de gris
    """
    # Charger l'image
    image_path = get_data_path(PHOBOS_IMAGE)
    image = img_as_float(io.imread(image_path, as_gray=True))
    print(f"Image chargée: {PHOBOS_IMAGE}")
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
        tuple: (cdf, bin_edges) où cdf est la fonction de distribution cumulative
               et bin_edges sont les bords des bins
    """
    # Calculer l'histogramme
    hist, bin_edges = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    
    # Calculer la CDF
    cdf = hist.cumsum()
    
    # Normaliser la CDF
    cdf = cdf / cdf[-1]
    
    return cdf, bin_edges

def visualize_cdf(image, bins=256, title="Fonction de distribution cumulative"):
    """
    Visualise la fonction de distribution cumulative (CDF) de l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        title (str): Titre du graphique
    """
    # Calculer la CDF
    cdf, bin_edges = compute_cdf(image, bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
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

def equalize_histogram(image, bins=256):
    """
    Applique l'égalisation d'histogramme à l'image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        ndarray: Image après égalisation d'histogramme
    """
    # Appliquer l'égalisation d'histogramme
    equalized = exposure.equalize_hist(image)
    
    return equalized

def create_bimodal_histogram(bins=256, peak1=0.25, peak2=0.75, sigma1=0.05, sigma2=0.05, weight1=0.5, weight2=0.5):
    """
    Crée un histogramme bimodal de référence.
    
    Args:
        bins (int): Nombre de bins pour l'histogramme
        peak1 (float): Position du premier pic (entre 0 et 1)
        peak2 (float): Position du deuxième pic (entre 0 et 1)
        sigma1 (float): Écart-type du premier pic
        sigma2 (float): Écart-type du deuxième pic
        weight1 (float): Poids du premier pic (entre 0 et 1)
        weight2 (float): Poids du deuxième pic (entre 0 et 1)
        
    Returns:
        tuple: (reference_hist, bin_centers) où reference_hist est l'histogramme bimodal
               et bin_centers sont les centres des bins
    """
    # Normaliser les poids
    total_weight = weight1 + weight2
    weight1 = weight1 / total_weight
    weight2 = weight2 / total_weight
    
    # Créer les bins
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Créer l'histogramme bimodal (somme de deux gaussiennes)
    reference_hist = weight1 * np.exp(-0.5 * ((bin_centers - peak1) / sigma1) ** 2) / (sigma1 * np.sqrt(2 * np.pi))
    reference_hist += weight2 * np.exp(-0.5 * ((bin_centers - peak2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))
    
    # Normaliser l'histogramme
    reference_hist = reference_hist / np.sum(reference_hist)
    
    return reference_hist, bin_centers

def visualize_bimodal_histogram(bins=256, peak1=0.25, peak2=0.75, sigma1=0.05, sigma2=0.05, weight1=0.5, weight2=0.5, title="Histogramme bimodal de référence"):
    """
    Visualise l'histogramme bimodal de référence.
    
    Args:
        bins (int): Nombre de bins pour l'histogramme
        peak1 (float): Position du premier pic (entre 0 et 1)
        peak2 (float): Position du deuxième pic (entre 0 et 1)
        sigma1 (float): Écart-type du premier pic
        sigma2 (float): Écart-type du deuxième pic
        weight1 (float): Poids du premier pic (entre 0 et 1)
        weight2 (float): Poids du deuxième pic (entre 0 et 1)
        title (str): Titre du graphique
    """
    # Créer l'histogramme bimodal
    reference_hist, bin_centers = create_bimodal_histogram(bins, peak1, peak2, sigma1, sigma2, weight1, weight2)
    
    # Créer la figure
    plt.figure(figsize=(10, 6))
    
    # Tracer l'histogramme
    plt.bar(bin_centers, reference_hist * bins, width=1/bins, alpha=0.7)  # Multiplier par bins pour normaliser la hauteur
    
    # Ajouter les labels et le titre
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    plt.title(title)
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    return reference_hist, bin_centers

def compute_cdf_from_hist(hist):
    """
    Calcule la fonction de distribution cumulative (CDF) à partir d'un histogramme.
    
    Args:
        hist (ndarray): Histogramme
        
    Returns:
        ndarray: Fonction de distribution cumulative
    """
    # Calculer la CDF
    cdf = hist.cumsum()
    
    # Normaliser la CDF
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    
    return cdf

def match_histogram_custom(image, reference_hist, bins=256):
    """
    Implémentation personnalisée de l'appariement d'histogramme.
    
    La transformation est définie par:
    x2 = cdf2^(-1)(cdf1(x1))
    
    où:
    - x1 est la valeur d'intensité dans l'image source
    - x2 est la valeur d'intensité correspondante dans l'image cible
    - cdf1 est la fonction de distribution cumulative de l'image source
    - cdf2 est la fonction de distribution cumulative de l'histogramme modèle
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        reference_hist (ndarray): Histogramme de référence
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        ndarray: Image après appariement d'histogramme
    """
    # Vérifier que l'image est bien en float avec des valeurs entre 0 et 1
    if image.min() < 0 or image.max() > 1:
        print("Attention: L'image doit avoir des valeurs entre 0 et 1. Normalisation appliquée.")
        image = (image - image.min()) / (image.max() - image.min())
    
    # Calculer l'histogramme de l'image source
    hist_source, bin_edges_source = np.histogram(image.ravel(), bins=bins, range=(0, 1))
    
    # Calculer la CDF de l'image source
    cdf_source = compute_cdf_from_hist(hist_source)
    
    # Calculer la CDF de l'histogramme de référence
    cdf_reference = compute_cdf_from_hist(reference_hist)
    
    # Créer la LUT (Look-Up Table) pour la transformation
    # Pour chaque valeur de cdf_source, trouver la valeur correspondante dans cdf_reference
    bin_centers = (bin_edges_source[:-1] + bin_edges_source[1:]) / 2
    
    # Créer un tableau pour stocker les valeurs transformées
    matched = np.zeros_like(image)
    
    # Pour chaque pixel de l'image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Trouver l'index du bin correspondant à la valeur du pixel
            pixel_value = image[i, j]
            bin_index = min(int(pixel_value * bins), bins - 1)
            
            # Obtenir la valeur de la CDF source pour ce pixel
            cdf_value = cdf_source[bin_index]
            
            # Trouver l'index dans la CDF de référence qui correspond le mieux à cette valeur
            idx = np.argmin(np.abs(cdf_reference - cdf_value))
            
            # Convertir l'index en valeur d'intensité
            matched[i, j] = bin_centers[idx]
    
    return matched

def match_histogram_builtin(image, reference_hist, bins=256):
    """
    Applique l'appariement d'histogramme à l'image en utilisant les fonctions intégrées de scikit-image.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        reference_hist (ndarray): Histogramme de référence
        bins (int): Nombre de bins pour l'histogramme
        
    Returns:
        ndarray: Image après appariement d'histogramme
    """
    # Créer une image de référence à partir de l'histogramme de référence
    # Pour cela, nous générons des valeurs aléatoires suivant la distribution de l'histogramme de référence
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normaliser l'histogramme de référence
    reference_hist_norm = reference_hist / np.sum(reference_hist)
    
    # Générer des valeurs aléatoires suivant la distribution de l'histogramme de référence
    reference_values = np.random.choice(bin_centers, size=10000, p=reference_hist_norm)
    reference_image = np.reshape(reference_values, (100, 100))
    
    # Appliquer l'appariement d'histogramme
    matched = exposure.match_histograms(image, reference_image)
    
    return matched

def visualize_matching_results(image, reference_hist, bins=256):
    """
    Visualise les résultats de l'appariement d'histogramme.
    
    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        reference_hist (ndarray): Histogramme de référence
        bins (int): Nombre de bins pour l'histogramme
    """
    # Appliquer l'égalisation d'histogramme
    equalized = equalize_histogram(image)
    
    # Appliquer l'appariement d'histogramme avec notre implémentation personnalisée
    matched_custom = match_histogram_custom(image, reference_hist, bins)
    
    # Appliquer l'appariement d'histogramme avec les fonctions intégrées
    matched_builtin = match_histogram_builtin(image, reference_hist, bins)
    
    # Créer une figure pour afficher les images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Afficher l'image originale
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Image originale')
    axes[0, 0].axis('off')
    
    # Afficher l'image après égalisation d'histogramme
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('Égalisation d\'histogramme')
    axes[0, 1].axis('off')
    
    # Afficher l'image après appariement d'histogramme (implémentation personnalisée)
    axes[1, 0].imshow(matched_custom, cmap='gray')
    axes[1, 0].set_title('Appariement d\'histogramme (custom)')
    axes[1, 0].axis('off')
    
    # Afficher l'image après appariement d'histogramme (fonctions intégrées)
    axes[1, 1].imshow(matched_builtin, cmap='gray')
    axes[1, 1].set_title('Appariement d\'histogramme (builtin)')
    axes[1, 1].axis('off')
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    # Visualiser les histogrammes
    plt.figure(figsize=(12, 8))
    
    # Histogramme de l'image originale
    plt.subplot(2, 2, 1)
    hist_original, bin_centers = compute_histogram(image, bins)
    plt.bar(bin_centers, hist_original, width=1/bins, alpha=0.7)
    plt.title('Histogramme original')
    plt.xlim(0, 1)
    
    # Histogramme de l'image après égalisation
    plt.subplot(2, 2, 2)
    hist_equalized, bin_centers = compute_histogram(equalized, bins)
    plt.bar(bin_centers, hist_equalized, width=1/bins, alpha=0.7)
    plt.title('Histogramme après égalisation')
    plt.xlim(0, 1)
    
    # Histogramme de l'image après appariement (implémentation personnalisée)
    plt.subplot(2, 2, 3)
    hist_matched_custom, bin_centers = compute_histogram(matched_custom, bins)
    plt.bar(bin_centers, hist_matched_custom, width=1/bins, alpha=0.7)
    plt.title('Histogramme après appariement (custom)')
    plt.xlim(0, 1)
    
    # Histogramme de l'image après appariement (fonctions intégrées)
    plt.subplot(2, 2, 4)
    hist_matched_builtin, bin_centers = compute_histogram(matched_builtin, bins)
    plt.bar(bin_centers, hist_matched_builtin, width=1/bins, alpha=0.7)
    plt.title('Histogramme après appariement (builtin)')
    plt.xlim(0, 1)
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    # Visualiser l'histogramme de référence
    plt.figure(figsize=(10, 6))
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, reference_hist * bins, width=1/bins, alpha=0.7)  # Multiplier par bins pour normaliser la hauteur
    plt.title('Histogramme de référence')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Visualiser les CDFs
    plt.figure(figsize=(10, 6))
    
    # CDF de l'image originale
    cdf_original, bin_edges = compute_cdf(image, bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, cdf_original, 'b-', linewidth=2, label='CDF originale')
    
    # CDF de l'histogramme de référence
    cdf_reference = compute_cdf_from_hist(reference_hist)
    plt.plot(bin_centers, cdf_reference, 'r-', linewidth=2, label='CDF de référence')
    
    # CDF de l'image après appariement (implémentation personnalisée)
    cdf_matched_custom, _ = compute_cdf(matched_custom, bins)
    plt.plot(bin_centers, cdf_matched_custom, 'g--', linewidth=2, label='CDF après appariement (custom)')
    
    # CDF de l'image après appariement (fonctions intégrées)
    cdf_matched_builtin, _ = compute_cdf(matched_builtin, bins)
    plt.plot(bin_centers, cdf_matched_builtin, 'm--', linewidth=2, label='CDF après appariement (builtin)')
    
    # Ajouter les labels et le titre
    plt.xlabel('Intensité')
    plt.ylabel('CDF')
    plt.title('Comparaison des CDFs')
    plt.xlim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Afficher la figure
    plt.tight_layout()
    plt.show()
    
    return equalized, matched_custom, matched_builtin

def main():
    """
    Fonction principale qui démontre l'appariement d'histogramme.
    """
    # Charger l'image
    image = load_image()
    
    # Visualiser l'histogramme de l'image
    print("Visualisation de l'histogramme de l'image...")
    visualize_histogram(image, title=f"Histogramme de l'image '{PHOBOS_IMAGE}'")
    
    # Appliquer l'égalisation d'histogramme
    print("Application de l'égalisation d'histogramme...")
    equalized = equalize_histogram(image)
    
    # Visualiser l'image après égalisation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image originale')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(equalized, cmap='gray')
    plt.title('Image après égalisation')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Visualiser l'histogramme après égalisation
    visualize_histogram(equalized, title="Histogramme après égalisation")
    
    # Créer un histogramme bimodal de référence
    print("Création d'un histogramme bimodal de référence...")
    reference_hist, _ = visualize_bimodal_histogram(peak1=0.3, peak2=0.7, sigma1=0.05, sigma2=0.05, weight1=0.6, weight2=0.4)
    
    # Appliquer l'appariement d'histogramme et visualiser les résultats
    print("Application de l'appariement d'histogramme...")
    visualize_matching_results(image, reference_hist)

if __name__ == "__main__":
    main()
