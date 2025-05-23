#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformations

Intensity transformation techniques including gamma correction and contrast stretching.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

"""
Transformations d'intensité (LUT) pour l'amélioration d'images

Ce module implémente deux transformations d'intensité principales :
1. Correction gamma (γ correction) - modifie la luminosité globale de l'image
2. Étirement de contraste (contrast stretching) - améliore le contraste en utilisant
   la fonction de distribution cumulative (CDF)

Ces transformations permettent de modifier la dynamique des niveaux de gris
d'une image pour améliorer sa visualisation.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_float, img_as_ubyte
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

def apply_gamma_correction(image, gamma=1.0):
    """
    Applique une correction gamma à l'image.

    La correction gamma est définie par la formule:
    s = r^γ
    où r est la valeur d'entrée (entre 0 et 1) et s est la valeur de sortie.

    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        gamma (float): Valeur du paramètre gamma

    Returns:
        ndarray: Image après correction gamma
    """
    # Vérifier que l'image est bien en float avec des valeurs entre 0 et 1
    if image.min() < 0 or image.max() > 1:
        print("Attention: L'image doit avoir des valeurs entre 0 et 1. Normalisation appliquée.")
        image = (image - image.min()) / (image.max() - image.min())

    # Appliquer la correction gamma
    corrected = np.power(image, gamma)

    return corrected

def plot_gamma_lut():
    """
    Affiche les courbes de la fonction de correction gamma pour différentes valeurs de gamma.
    """
    # Créer une plage de valeurs d'entrée entre 0 et 1
    r = np.linspace(0, 1, 1000)

    # Valeurs de gamma à afficher
    gamma_values = [0.04, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 25.0]

    # Créer la figure
    plt.figure(figsize=(10, 8))

    # Tracer les courbes pour chaque valeur de gamma
    for gamma in gamma_values:
        s = np.power(r, gamma)
        if gamma < 1:
            plt.plot(r, s, '--', label=f'γ = {gamma}')
        else:
            plt.plot(r, s, '-', label=f'γ = {gamma}')

    # Ajouter les labels et la légende
    plt.xlabel('Entrée (r)')
    plt.ylabel('Sortie (s)')
    plt.title('Courbes de correction gamma')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Ajouter une ligne diagonale pour γ = 1
    plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)

    # Afficher la figure
    plt.tight_layout()
    plt.show()

def apply_contrast_stretching(image, E=4.0):
    """
    Applique un étirement de contraste à l'image en utilisant la formule:
    s = T(r) = 1 / (1 + (m/r)^E)

    où m est la valeur moyenne de gris de l'image et r est la valeur du pixel.

    Args:
        image (ndarray): Image en niveaux de gris (valeurs entre 0 et 1)
        E (float): Paramètre de contrôle de la pente de la transformation

    Returns:
        ndarray: Image après étirement de contraste
    """
    # Vérifier que l'image est bien en float avec des valeurs entre 0 et 1
    if image.min() < 0 or image.max() > 1:
        print("Attention: L'image doit avoir des valeurs entre 0 et 1. Normalisation appliquée.")
        image = (image - image.min()) / (image.max() - image.min())

    # Calculer la valeur moyenne de l'image
    m = np.mean(image)
    print(f"Valeur moyenne de l'image (m): {m:.4f}")

    # Éviter la division par zéro
    epsilon = 1e-10

    # Appliquer la transformation
    # Note: nous ajoutons epsilon pour éviter la division par zéro
    stretched = 1.0 / (1.0 + np.power(m / (image + epsilon), E))

    return stretched

def plot_contrast_stretching_lut(m=0.5, E_values=None):
    """
    Affiche les courbes de la fonction d'étirement de contraste pour différentes valeurs de E.

    Args:
        m (float): Valeur moyenne de gris (entre 0 et 1)
        E_values (list): Liste des valeurs de E à afficher
    """
    if E_values is None:
        E_values = [10, 20, 30, 40, 50, 1000]

    # Créer une plage de valeurs d'entrée entre 0 et 1
    r = np.linspace(0.01, 1, 1000)  # Commencer à 0.01 pour éviter la division par zéro

    # Créer la figure
    plt.figure(figsize=(10, 8))

    # Tracer les courbes pour chaque valeur de E
    for E in E_values:
        s = 1.0 / (1.0 + np.power(m / r, E))
        plt.plot(r, s, label=f'E = {E}')

    # Ajouter les labels et la légende
    plt.xlabel('Entrée (r)')
    plt.ylabel('Sortie (s)')
    plt.title(f'Courbes d\'étirement de contraste (m = {m})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Marquer la valeur moyenne m
    plt.axvline(x=m, color='k', linestyle='--', alpha=0.5)
    plt.text(m+0.02, 0.1, f'm = {m}', rotation=0)

    # Afficher la figure
    plt.tight_layout()
    plt.show()

def test_gamma_correction(image=None, gamma_values=None):
    """
    Teste la correction gamma sur l'image d'ostéoblastes avec différentes valeurs de gamma.

    Args:
        image (ndarray, optional): Image à traiter. Si None, charge l'image par défaut.
        gamma_values (list, optional): Liste des valeurs de gamma à tester.
    """
    if image is None:
        image = load_image()

    if gamma_values is None:
        gamma_values = [0.5, 1.0, 2.0]

    # Créer une figure pour afficher les résultats
    n_values = len(gamma_values)
    fig, axes = plt.subplots(1, n_values + 1, figsize=(4 * (n_values + 1), 4))

    # Afficher l'image originale
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    # Appliquer et afficher la correction gamma pour chaque valeur
    for i, gamma in enumerate(gamma_values):
        corrected = apply_gamma_correction(image, gamma)
        axes[i+1].imshow(corrected, cmap='gray')
        axes[i+1].set_title(f'Gamma = {gamma}')
        axes[i+1].axis('off')

    # Afficher la figure
    plt.tight_layout()
    plt.show()

    # Afficher les histogrammes
    fig, axes = plt.subplots(1, n_values + 1, figsize=(4 * (n_values + 1), 4))

    # Histogramme de l'image originale
    axes[0].hist(image.ravel(), bins=256, range=(0, 1), density=True, alpha=0.7)
    axes[0].set_title('Histogramme original')
    axes[0].set_xlim(0, 1)

    # Histogrammes des images corrigées
    for i, gamma in enumerate(gamma_values):
        corrected = apply_gamma_correction(image, gamma)
        axes[i+1].hist(corrected.ravel(), bins=256, range=(0, 1), density=True, alpha=0.7)
        axes[i+1].set_title(f'Histogramme (Gamma = {gamma})')
        axes[i+1].set_xlim(0, 1)

    # Afficher la figure
    plt.tight_layout()
    plt.show()

def test_contrast_stretching(image=None, E_values=None):
    """
    Teste l'étirement de contraste sur l'image d'ostéoblastes avec différentes valeurs de E.

    Args:
        image (ndarray, optional): Image à traiter. Si None, charge l'image par défaut.
        E_values (list, optional): Liste des valeurs de E à tester.
    """
    if image is None:
        image = load_image()

    if E_values is None:
        E_values = [10, 20, 40, 80]

    # Créer une figure pour afficher les résultats
    n_values = len(E_values)
    fig, axes = plt.subplots(1, n_values + 1, figsize=(4 * (n_values + 1), 4))

    # Afficher l'image originale
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    # Appliquer et afficher l'étirement de contraste pour chaque valeur
    for i, E in enumerate(E_values):
        stretched = apply_contrast_stretching(image, E)
        axes[i+1].imshow(stretched, cmap='gray')
        axes[i+1].set_title(f'E = {E}')
        axes[i+1].axis('off')

    # Afficher la figure
    plt.tight_layout()
    plt.show()

    # Afficher les histogrammes
    fig, axes = plt.subplots(1, n_values + 1, figsize=(4 * (n_values + 1), 4))

    # Histogramme de l'image originale
    axes[0].hist(image.ravel(), bins=256, range=(0, 1), density=True, alpha=0.7)
    axes[0].set_title('Histogramme original')
    axes[0].set_xlim(0, 1)

    # Histogrammes des images étirées
    for i, E in enumerate(E_values):
        stretched = apply_contrast_stretching(image, E)
        axes[i+1].hist(stretched.ravel(), bins=256, range=(0, 1), density=True, alpha=0.7)
        axes[i+1].set_title(f'Histogramme (E = {E})')
        axes[i+1].set_xlim(0, 1)

    # Afficher la figure
    plt.tight_layout()
    plt.show()

def main():
    """
    Fonction principale qui démontre les transformations d'intensité.
    """
    # Charger l'image
    image = load_image()

    # Afficher les courbes LUT
    print("Affichage des courbes de correction gamma...")
    plot_gamma_lut()

    print("Affichage des courbes d'étirement de contraste...")
    plot_contrast_stretching_lut()

    # Tester la correction gamma
    print("Test de la correction gamma...")
    test_gamma_correction(image, [0.5, 1.0, 2.0])

    # Tester l'étirement de contraste
    print("Test de l'étirement de contraste...")
    test_contrast_stretching(image, [10, 20, 40, 1000])

if __name__ == "__main__":
    main()
