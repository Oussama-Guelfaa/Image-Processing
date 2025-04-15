# Modélisation des dommages et restauration d'image

Ce module permet de simuler des dommages sur des images et de les restaurer en utilisant différentes techniques de déconvolution.

## Théorie

### Modèle de dommage

Un processus de dommage peut être modélisé par une fonction de dommage D et un bruit additif n, agissant sur une image d'entrée f pour produire l'image endommagée g :

```
g = D(f) + n
```

Si D est un processus linéaire, invariant dans l'espace, alors l'équation peut être simplifiée comme :

```
g = h * f + n
```

où h est la représentation spatiale de la fonction de dommage (appelée Point Spread Function - PSF), et * désigne l'opération de convolution.

Dans le domaine fréquentiel, cette équation devient :

```
G = H·F + N
```

où les lettres majuscules sont les transformées de Fourier des termes correspondants.

### Restauration d'image

L'objectif de la restauration est d'obtenir une estimation f̂ (l'image restaurée) de l'image originale f. Plus vous avez de connaissances sur la fonction H et le bruit n, plus f̂ sera proche de f.

## Fonctionnalités

### 1. Génération d'images de test

- **Damier (Checkerboard)** : Génère une image en damier, utile pour tester les algorithmes de restauration.

### 2. Génération de PSF (Point Spread Function)

- **PSF Gaussienne** : Simule un flou isotrope, commun dans de nombreux systèmes d'imagerie.
- **PSF de flou de mouvement** : Simule le flou causé par le mouvement de la caméra ou de l'objet pendant l'exposition.

### 3. Application de dommages

- **Convolution spatiale** : Applique la PSF à l'image dans le domaine spatial.
- **Convolution fréquentielle** : Applique la PSF à l'image dans le domaine fréquentiel.
- **Ajout de bruit** : Ajoute un bruit gaussien à l'image floutée.

### 4. Méthodes de restauration

- **Filtre inverse** : La méthode la plus simple, mais sensible au bruit.
- **Filtre de Wiener** : Plus robuste au bruit que le filtre inverse.

## Utilisation

### En ligne de commande

#### Génération d'une image en damier

```bash
imgproc checkerboard --size 8 --square_size 32 --output checkerboard.png
```

#### Application de dommages à une image

```bash
# Avec une PSF gaussienne
imgproc damage --psf gaussian --sigma 3.0 --noise 0.01 --image path/to/image.jpg --output damaged.png

# Avec une PSF de flou de mouvement
imgproc damage --psf motion --length 15 --angle 45 --noise 0.01 --image path/to/image.jpg --output damaged.png
```

#### Restauration d'une image endommagée

```bash
# Avec le filtre inverse
imgproc restore --method inverse --psf gaussian --sigma 3.0 --image damaged.png --output restored.png

# Avec le filtre de Wiener
imgproc restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image damaged.png --output restored.png

# Comparaison des différentes méthodes
imgproc restore --method compare --psf gaussian --sigma 3.0 --image damaged.png --output restored.png
```

### Dans un script Python

```python
from src.image_processing import damage_modeling

# Charger une image
image = damage_modeling.load_image('path/to/image.jpg')

# Générer une PSF
psf = damage_modeling.generate_gaussian_psf(size=64, sigma=3)
# ou
psf = damage_modeling.generate_motion_blur_psf(size=64, length=15, angle=45)

# Visualiser la PSF
damage_modeling.visualize_psf(psf, title="Point Spread Function")

# Appliquer des dommages à l'image
damaged = damage_modeling.apply_damage(image, psf, noise_level=0.01)

# Restaurer l'image avec le filtre inverse
restored_inverse = damage_modeling.inverse_filter(damaged, psf, epsilon=1e-3)

# Restaurer l'image avec le filtre de Wiener
restored_wiener = damage_modeling.wiener_filter(damaged, psf, K=0.01)

# Visualiser les résultats
damage_modeling.visualize_restoration_results(image, damaged, restored_wiener,
                                             titles=["Image originale", "Image endommagée", "Image restaurée"])
```

## Paramètres importants

### PSF Gaussienne
- **size** : Taille de la PSF en pixels (défaut : 64)
- **sigma** : Écart-type de la gaussienne (défaut : 3)

### PSF de flou de mouvement
- **size** : Taille de la PSF en pixels (défaut : 64)
- **length** : Longueur du flou de mouvement en pixels (défaut : 15)
- **angle** : Angle du flou de mouvement en degrés (défaut : 45)

### Application de dommages
- **noise_level** : Niveau de bruit gaussien (défaut : 0.01)

### Filtre inverse
- **epsilon** : Petite valeur pour éviter la division par zéro (défaut : 1e-3)

### Filtre de Wiener
- **K** : Paramètre lié au rapport signal/bruit (défaut : 0.01)
  - Valeurs plus petites (ex : 0.001) : Meilleure restauration des détails, mais plus sensible au bruit
  - Valeurs plus grandes (ex : 0.1) : Moins sensible au bruit, mais perte de détails

## Exemples de résultats

### Image originale vs. Image endommagée vs. Image restaurée
![Exemple de restauration](../figures/restoration_example.png)

### Comparaison des méthodes de restauration
![Comparaison des méthodes](../figures/restoration_comparison.png)

## Limitations

- Le filtre inverse est très sensible au bruit et peut amplifier le bruit dans l'image restaurée.
- Le filtre de Wiener nécessite une estimation du rapport signal/bruit (paramètre K).
- Les deux méthodes supposent que la PSF est connue avec précision, ce qui n'est pas toujours le cas dans les applications réelles.

## Références

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
2. Jain, A. K. (1989). Fundamentals of Digital Image Processing. Prentice-Hall.

## Auteur

Oussama GUELFAA
Date: 01-04-2025
