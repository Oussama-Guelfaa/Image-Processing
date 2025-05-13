#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour comparer les résultats de restauration avec différentes valeurs de K
pour le filtre de Wiener sur les images de Jupiter et Saturne.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, metrics

def compare_images(original_path, restored_paths, titles, output_path):
    """
    Compare l'image originale avec plusieurs images restaurées.
    
    Args:
        original_path (str): Chemin vers l'image originale
        restored_paths (list): Liste des chemins vers les images restaurées
        titles (list): Liste des titres pour les images restaurées
        output_path (str): Chemin pour sauvegarder l'image de comparaison
    """
    # Charger l'image originale
    original = io.imread(original_path)
    if len(original.shape) == 3 and original.shape[2] == 4:  # Si l'image a un canal alpha
        original = original[:, :, :3]
    if len(original.shape) == 3:
        original = color.rgb2gray(original)
    
    # Charger les images restaurées
    restored_images = []
    for path in restored_paths:
        if os.path.exists(path):
            img = io.imread(path)
            if len(img.shape) == 3 and img.shape[2] == 4:  # Si l'image a un canal alpha
                img = img[:, :, :3]
            if len(img.shape) == 3:
                img = color.rgb2gray(img)
            restored_images.append(img)
        else:
            print(f"Attention: {path} n'existe pas.")
            restored_images.append(np.zeros_like(original))
    
    # Calculer les métriques de qualité
    psnr_values = []
    ssim_values = []
    for img in restored_images:
        try:
            psnr = metrics.peak_signal_noise_ratio(original, img)
            ssim = metrics.structural_similarity(original, img)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
        except Exception as e:
            print(f"Erreur lors du calcul des métriques: {e}")
            psnr_values.append(0)
            ssim_values.append(0)
    
    # Créer la figure
    n_images = len(restored_images) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 4, 4))
    
    # Afficher l'image originale
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    # Afficher les images restaurées avec leurs métriques
    for i, (img, title, psnr, ssim) in enumerate(zip(restored_images, titles, psnr_values, ssim_values)):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"{title}\nPSNR: {psnr:.2f} dB\nSSIM: {ssim:.4f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Comparaison sauvegardée dans: {output_path}")
    
    # Retourner les métriques pour analyse
    return psnr_values, ssim_values

def main():
    """Fonction principale."""
    # Comparer les résultats pour Jupiter
    original_path = "data/jupiter.png"
    jupiter_restored_paths = [
        "jupiter_restored_k0.0001.png",
        "jupiter_restored_k0.001.png",
        "jupiter_restored_k0.01.png",
        "jupiter_restored_k0.1.png"
    ]
    jupiter_titles = [
        "K=0.0001",
        "K=0.001",
        "K=0.01",
        "K=0.1"
    ]
    jupiter_psnr, jupiter_ssim = compare_images(
        original_path, 
        jupiter_restored_paths, 
        jupiter_titles, 
        "jupiter_wiener_comparison.png"
    )
    
    # Comparer les résultats pour Saturne
    original_path = "data/saturn.png"
    saturn_restored_paths = [
        "saturn_restored_k0.0001.png",
        "saturn_restored_k0.001.png",
        "saturn_restored_k0.01.png",
        "saturn_restored_k0.1.png"
    ]
    saturn_titles = [
        "K=0.0001",
        "K=0.001",
        "K=0.01",
        "K=0.1"
    ]
    saturn_psnr, saturn_ssim = compare_images(
        original_path, 
        saturn_restored_paths, 
        saturn_titles, 
        "saturn_wiener_comparison.png"
    )
    
    # Afficher un résumé des métriques
    print("\nRésumé des métriques pour Jupiter:")
    for k, psnr, ssim in zip(["0.0001", "0.001", "0.01", "0.1"], jupiter_psnr, jupiter_ssim):
        print(f"K={k}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    print("\nRésumé des métriques pour Saturne:")
    for k, psnr, ssim in zip(["0.0001", "0.001", "0.01", "0.1"], saturn_psnr, saturn_ssim):
        print(f"K={k}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
    
    # Trouver la meilleure valeur de K pour chaque image
    best_k_jupiter = ["0.0001", "0.001", "0.01", "0.1"][np.argmax(jupiter_ssim)]
    best_k_saturn = ["0.0001", "0.001", "0.01", "0.1"][np.argmax(saturn_ssim)]
    
    print(f"\nMeilleure valeur de K pour Jupiter (basée sur SSIM): {best_k_jupiter}")
    print(f"Meilleure valeur de K pour Saturne (basée sur SSIM): {best_k_saturn}")

if __name__ == "__main__":
    main()
