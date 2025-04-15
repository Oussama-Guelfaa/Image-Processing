# Égalisation d'Histogramme

Ce module implémente l'égalisation d'histogramme, une technique qui transforme l'image de sorte que son histogramme soit constant (et sa fonction de distribution cumulative soit linéaire).

## Théorie

L'égalisation d'histogramme est définie par la transformation :

```
T(x_k) = L * cdf_I(k)
```

où :
- `x_k` est la valeur d'intensité k
- `L` est la valeur maximale d'intensité (255 pour les images 8 bits)
- `cdf_I(k)` est la fonction de distribution cumulative de l'image

### Notations

- `I` est l'image de `n` pixels, avec des intensités entre 0 et `L` (pour les images 8 bits, `L = 255`).
- `h` est l'histogramme, défini par :

```
h_I(k) = p(x = k) = n_k / n,   0 ≤ k ≤ L
```

où `n_k` est le nombre de pixels ayant l'intensité `k`.

### Fonction de Distribution Cumulative (CDF)

La CDF est définie comme la somme cumulative de l'histogramme :

```
cdf_I(k) = Σ p(x_j)
           j=0
```

L'égalisation d'histogramme utilise cette CDF comme fonction de transformation pour redistribuer les intensités de l'image.

## Implémentation

Deux implémentations sont fournies :

1. **Implémentation intégrée** : Utilise la fonction `equalize_hist` de scikit-image
2. **Implémentation personnalisée** : Implémente l'algorithme d'égalisation d'histogramme à partir de zéro

## Utilisation en ligne de commande

### Égalisation d'histogramme avec la fonction intégrée

```bash
python main.py histogram --method builtin
```

Options :
- `--method builtin` : Utilise la fonction intégrée de scikit-image
- `--image chemin/vers/image.jpg` : Spécifie une image d'entrée (par défaut : osteoblaste.jpg)
- `--bins 256` : Nombre de bins pour l'histogramme (par défaut : 256)
- `--output resultat.jpg` : Sauvegarde l'image égalisée

### Égalisation d'histogramme avec l'implémentation personnalisée

```bash
python main.py histogram --method custom
```

Options :
- `--method custom` : Utilise l'implémentation personnalisée
- `--image chemin/vers/image.jpg` : Spécifie une image d'entrée (par défaut : osteoblaste.jpg)
- `--bins 256` : Nombre de bins pour l'histogramme (par défaut : 256)
- `--output resultat.jpg` : Sauvegarde l'image égalisée

### Comparaison des deux méthodes

```bash
python main.py histogram --method both
```

Cette commande exécute les deux méthodes et affiche les résultats côte à côte pour comparaison.

## Résultats

L'égalisation d'histogramme améliore généralement le contraste global de l'image en redistribuant les intensités de manière uniforme sur toute la plage disponible. Cela peut faire ressortir des détails qui étaient auparavant difficiles à distinguer dans les zones sombres ou claires de l'image.

### Avantages

- Améliore automatiquement le contraste
- Fait ressortir les détails dans les zones de faible contraste
- Simple à implémenter et à utiliser

### Inconvénients

- Peut amplifier le bruit dans l'image
- Peut produire des artefacts non naturels
- Ne prend pas en compte le contenu sémantique de l'image

## Auteur

Oussama GUELFAA
Date: 01-04-2025
