# Calcul de la valeur optimale de K pour le filtre de Wiener

Ce document explique comment calculer la valeur optimale du paramètre K pour le filtre de Wiener, en se basant sur les caractéristiques du bruit et de l'image originale.

## Théorie

Le filtre de Wiener est une méthode optimale qui minimise l'erreur quadratique moyenne entre l'image originale et l'image restaurée. Dans le domaine fréquentiel, la solution est donnée par :

```
F = (H* / (|H|² + K)) · G
```

Où :
- H* est le conjugué complexe de H (l'OTF)
- |H|² est le carré de la magnitude de H
- K est un paramètre lié au rapport signal/bruit
- G est la transformée de Fourier de l'image dégradée

Le paramètre K peut être calculé comme le rapport entre le spectre de puissance du bruit et celui de l'image originale :

```
K = S_n / S_f
```

Où S_n est le spectre de puissance du bruit et S_f est le spectre de puissance de l'image originale.

## Calcul pratique

En pratique, si l'on connaît le niveau de bruit (σ_n) et les caractéristiques de l'image originale (σ_f), on peut calculer une valeur approximative de K comme le rapport entre la variance du bruit et la variance de l'image :

```
K ≈ σ²_n / σ²_f
```

### Exemple de code Python

Voici un exemple de code Python pour calculer la valeur optimale de K :

```python
import numpy as np
from skimage import io, color

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

# Exemple d'utilisation
image = io.imread('path/to/image.jpg')
if len(image.shape) == 3:  # Si l'image est en couleur
    image = color.rgb2gray(image)  # Convertir en niveaux de gris

noise_level = 0.01  # Niveau de bruit utilisé pour dégrader l'image
optimal_k = calculate_wiener_k(image, noise_level)
print(f"Valeur optimale de K pour le filtre de Wiener : {optimal_k:.6f}")
```

## Considérations pratiques

1. **Estimation du niveau de bruit** : Dans les applications réelles, le niveau de bruit n'est souvent pas connu à l'avance. Il peut être estimé à partir de régions homogènes de l'image dégradée.

2. **Variation spatiale** : Le rapport signal/bruit peut varier spatialement dans l'image. Dans ce cas, un filtre de Wiener adaptatif qui utilise différentes valeurs de K pour différentes régions de l'image peut être plus efficace.

3. **Approche empirique** : En pratique, il est souvent utile de tester plusieurs valeurs de K et de choisir celle qui donne le meilleur résultat visuel ou qui maximise une métrique de qualité comme le PSNR (Peak Signal-to-Noise Ratio) ou le SSIM (Structural Similarity Index).

## Valeurs typiques de K

- **K ≈ 0.0001 - 0.001** : Pour les images avec peu de bruit et beaucoup de détails fins.
- **K ≈ 0.01** : Valeur moyenne qui offre un bon compromis pour la plupart des images.
- **K ≈ 0.1 - 1.0** : Pour les images très bruitées où la préservation des détails est moins importante que la réduction du bruit.

## Auteur

Oussama GUELFAA
Date: 01-04-2025
