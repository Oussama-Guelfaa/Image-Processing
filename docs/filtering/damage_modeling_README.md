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

### Fonction de Transfert Optique (OTF)

La Fonction de Transfert Optique (OTF) est la transformée de Fourier centrée de la PSF. Elle est utilisée pour effectuer des calculs de convolution dans le domaine fréquentiel, ce qui est souvent plus efficace que la convolution dans le domaine spatial, surtout pour les grandes PSF.

La relation entre la PSF et l'OTF est donnée par :

```
OTF = F{PSF centrée}
```

où F{} représente la transformée de Fourier.

### Modèle de dommage dans le domaine fréquentiel

Dans le domaine fréquentiel, le modèle de dommage devient :

```
G = H·F + N
```

où les lettres majuscules sont les transformées de Fourier des termes correspondants, et H est l'OTF.

### Restauration d'image

L'objectif de la restauration est d'obtenir une estimation f̂ (l'image restaurée) de l'image originale f. Plus vous avez de connaissances sur la fonction H et le bruit n, plus f̂ sera proche de f.

## Fonctionnalités

### 1. Génération d'images de test

- **Damier (Checkerboard)** : Génère une image en damier, utile pour tester les algorithmes de restauration.

### 2. Génération de PSF (Point Spread Function) et OTF (Optical Transfer Function)

- **PSF Gaussienne** : Simule un flou isotrope, commun dans de nombreux systèmes d'imagerie.
- **PSF de flou de mouvement** : Simule le flou causé par le mouvement de la caméra ou de l'objet pendant l'exposition.
- **Fonction de Transfert Optique (OTF)** : La transformée de Fourier centrée de la PSF, utilisée pour les calculs dans le domaine fréquentiel.

### 3. Application de dommages

- **Convolution spatiale** : Applique la PSF à l'image dans le domaine spatial.
- **Convolution fréquentielle** : Applique la PSF à l'image dans le domaine fréquentiel.
- **Ajout de bruit** : Ajoute un bruit gaussien à l'image floutée.

### 4. Méthodes de restauration

- **Filtre inverse** : La méthode la plus simple, mais sensible au bruit.
- **Filtre de Wiener** : Plus robuste au bruit que le filtre inverse.

#### Filtre inverse (cas simple sans bruit)

Dans le cas où il n'y a pas de bruit (n = 0), le modèle de dégradation devient simplement :

```
g = h * f
```

Dans le domaine fréquentiel, cela devient :

```
G = H · F
```

Où G est la transformée de Fourier de g, H est l'OTF (transformée de Fourier de h), et F est la transformée de Fourier de f.

La restauration par filtre inverse consiste alors à estimer F en divisant G par H :

```
F = G / H
```

Et l'image restaurée f est obtenue par transformée de Fourier inverse :

```
f = FT^(-1){F} = FT^(-1){G / H}
```

Cependant, cette approche pose un problème lorsque H contient des valeurs nulles ou proches de zéro, car la division devient instable. Pour éviter ce problème, on ajoute une petite constante ε au dénominateur :

```
F = G / (H + ε)
```

Où ε est une petite valeur positive (par exemple 0.001).

#### Filtre de Wiener

Le filtre de Wiener est une méthode optimale qui minimise l'erreur quadratique moyenne entre l'image originale et l'image restaurée :

```
E[|f - f'|^2]
```

Où E[] dénote l'espérance mathématique, f est l'image originale et f' est l'image restaurée.

Dans le domaine fréquentiel, la solution est donnée par :

```
F = (H* / (|H|^2 + K)) · G
```

Où :
- H* est le conjugué complexe de H
- |H|^2 est le carré de la magnitude de H
- K est un paramètre lié au rapport signal/bruit

Le paramètre K peut être fixé arbitrairement ou calculé comme le rapport entre le spectre de puissance du bruit et celui de l'image originale :

```
K = S_n / S_f
```

Où S_n est le spectre de puissance du bruit et S_f est le spectre de puissance de l'image originale.

##### Cas sans bruit

Dans le cas où il n'y a pas de bruit, le filtre de Wiener se réduit au filtre inverse :

```
H_w = { 1/H(u,v)  si H(u,v) ≠ 0
        0         sinon
```

##### Cas avec bruit

Dans le cas général avec bruit, le rapport S_n/S_f est remplacé par une constante K qui représente le rapport des spectres de puissance moyens :

```
K = (1/PQ) ∑∑ S_n(u,v) / (1/PQ) ∑∑ S_f(u,v)
```

Où PQ est la taille de la matrice (nombre d'éléments).

En pratique, comme nous ne connaissons généralement pas les spectres de puissance exacts, nous utilisons une valeur constante pour K qui peut être ajustée empiriquement :
- Des valeurs plus petites de K (ex : 0.0001 ou 0.001) donnent une meilleure restauration des détails, mais sont plus sensibles au bruit.
- Des valeurs moyennes de K (ex : 0.01) offrent un bon compromis entre restauration des détails et réduction du bruit.
- Des valeurs plus grandes de K (ex : 0.1) réduisent l'amplification du bruit, mais peuvent lisser les détails de l'image.

#### Choix de la valeur optimale de K

Le choix de la valeur optimale de K dépend de plusieurs facteurs :

1. **Niveau de bruit** : Plus le niveau de bruit est élevé, plus K devrait être grand.
2. **Type de PSF** : Certaines PSF (comme les PSF de mouvement) peuvent nécessiter des valeurs de K différentes.
3. **Contenu de l'image** : Les images avec beaucoup de détails fins peuvent nécessiter des valeurs de K plus petites.

Si l'on connaît le niveau de bruit et les caractéristiques de l'image originale, on peut calculer une valeur approximative de K comme le rapport entre la puissance moyenne du bruit et la puissance moyenne de l'image :

```
K ≈ σ²_n / σ²_f
```

Où σ²_n est la variance du bruit et σ²_f est la variance de l'image originale.

En pratique, il est souvent utile de tester plusieurs valeurs de K et de choisir celle qui donne le meilleur résultat visuel ou qui maximise une métrique de qualité comme le PSNR (Peak Signal-to-Noise Ratio) ou le SSIM (Structural Similarity Index).

Pour plus de détails sur le calcul de la valeur optimale de K, consultez le document [wiener_filter_optimal_k.md](wiener_filter_optimal_k.md).

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

# Test de différentes valeurs de K pour le filtre de Wiener
imgproc restore --method wiener --k 0.0001 --psf gaussian --sigma 3.0 --image damaged.png --output restored_k0.0001.png
imgproc restore --method wiener --k 0.001 --psf gaussian --sigma 3.0 --image damaged.png --output restored_k0.001.png
imgproc restore --method wiener --k 0.01 --psf gaussian --sigma 3.0 --image damaged.png --output restored_k0.01.png
imgproc restore --method wiener --k 0.1 --psf gaussian --sigma 3.0 --image damaged.png --output restored_k0.1.png
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

# Visualiser la PSF et son OTF
damage_modeling.visualize_otf(psf, title="PSF et sa Fonction de Transfert Optique")

# Appliquer des dommages à l'image (domaine spatial)
damaged = damage_modeling.apply_damage(image, psf, noise_level=0.01)

# Appliquer des dommages à l'image (domaine fréquentiel avec OTF)
damaged_freq = damage_modeling.apply_damage_frequency(image, psf, noise_level=0.01)

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
