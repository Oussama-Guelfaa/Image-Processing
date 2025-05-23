#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtering

Image filtering operations including low-pass, high-pass, derivative, and Wiener filters.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import os

from src.image_processing.damage_modeling import (
    load_image, generate_gaussian_psf, generate_motion_blur_psf,
    apply_damage, inverse_filter, wiener_filter, psf2otf
)


def calculate_power_spectrum(image):
    """
    Calcule le spectre de puissance d'une image.
    
    Args:
        image (ndarray): Image d'entrée
        
    Returns:
        ndarray: Spectre de puissance
    """
    # Calculer la transformée de Fourier
    f = np.fft.fft2(image)
    # Calculer le spectre de puissance (carré de la magnitude)
    power_spectrum = np.abs(f) ** 2
    return power_spectrum


def calculate_wiener_k(original_image, noise_level=0.01):
    """
    Calcule le paramètre K pour le filtre de Wiener basé sur le rapport
    entre le spectre de puissance du bruit et celui de l'image originale.
    
    Args:
        original_image (ndarray): Image originale
        noise_level (float): Niveau de bruit (écart-type du bruit gaussien)
        
    Returns:
        float: Valeur de K pour le filtre de Wiener
    """
    # Générer un bruit gaussien avec le même niveau que celui utilisé pour dégrader l'image
    noise = np.random.normal(0, noise_level, original_image.shape)
    
    # Calculer les spectres de puissance
    S_f = calculate_power_spectrum(original_image)
    S_n = calculate_power_spectrum(noise)
    
    # Calculer le rapport moyen
    K = np.mean(S_n) / np.mean(S_f)
    
    return K


def compare_restoration_methods(image_path=None, psf_type='gaussian', noise_level=0.01):
    """
    Compare différentes méthodes de restauration d'image avec différentes valeurs de K.
    
    Args:
        image_path (str): Chemin vers l'image (si None, utilise une image par défaut)
        psf_type (str): Type de PSF ('gaussian' ou 'motion')
        noise_level (float): Niveau de bruit
    """
    # Charger l'image
    original = load_image(image_path)
    
    # Générer la PSF
    if psf_type == 'gaussian':
        psf = generate_gaussian_psf(size=32, sigma=3)
        psf_title = "Gaussian PSF"
    else:  # motion
        psf = generate_motion_blur_psf(size=32, length=15, angle=45)
        psf_title = "Motion Blur PSF"
    
    # Appliquer les dommages à l'image
    damaged = apply_damage(original, psf, noise_level=noise_level)
    
    # Calculer le K optimal pour le filtre de Wiener
    optimal_k = calculate_wiener_k(original, noise_level)
    print(f"K optimal calculé: {optimal_k:.6f}")
    
    # Restaurer avec le filtre inverse
    restored_inverse = inverse_filter(damaged, psf, epsilon=1e-3)
    
    # Restaurer avec le filtre de Wiener pour différentes valeurs de K
    k_values = [0.0001, 0.001, 0.01, optimal_k, 0.1]
    restored_wiener = []
    
    for k in k_values:
        restored = wiener_filter(damaged, psf, K=k)
        restored_wiener.append(restored)
    
    # Visualiser les résultats
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Première ligne: original, endommagé, filtre inverse
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Image Originale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(damaged, cmap='gray')
    axes[0, 1].set_title('Image Endommagée')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(restored_inverse, cmap='gray')
    axes[0, 2].set_title('Filtre Inverse')
    axes[0, 2].axis('off')
    
    # Deuxième ligne: filtre de Wiener avec différentes valeurs de K
    for i, (k, restored) in enumerate(zip(k_values[:4], restored_wiener[:4])):
        axes[1, i].imshow(restored, cmap='gray')
        if k == optimal_k:
            axes[1, i].set_title(f'Wiener (K={k:.6f}, optimal)')
        else:
            axes[1, i].set_title(f'Wiener (K={k})')
        axes[1, i].axis('off')
    
    # Ajouter des informations sur le PSF et le bruit
    axes[0, 3].text(0.5, 0.5, f"PSF: {psf_title}\nBruit: {noise_level}\nK optimal: {optimal_k:.6f}",
                   horizontalalignment='center', verticalalignment='center',
                   transform=axes[0, 3].transAxes, fontsize=12)
    axes[0, 3].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Comparaison des méthodes de restauration d'image", fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Sauvegarder l'image
    base_name = os.path.basename(image_path) if image_path else "default"
    output_path = f"{base_name.split('.')[0]}_restoration_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Comparaison sauvegardée dans: {output_path}")
    
    plt.show()
    
    return original, damaged, restored_inverse, restored_wiener, optimal_k


def main():
    """Fonction principale."""
    # Tester avec l'image de Jupiter
    compare_restoration_methods(image_path="data/jupiter.png", psf_type='gaussian', noise_level=0.01)
    
    # Tester avec l'image de Saturne
    compare_restoration_methods(image_path="data/saturn.png", psf_type='motion', noise_level=0.01)


if __name__ == "__main__":
    main()
