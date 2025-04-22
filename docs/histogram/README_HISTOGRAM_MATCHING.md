# Appariement d'Histogramme (Histogram Matching)

Ce module implémente l'appariement d'histogramme, une technique qui transforme l'image de sorte que son histogramme corresponde à un histogramme modèle.

## Théorie

L'appariement d'histogramme est défini par la transformation :

```
x2 = cdf2^(-1)(cdf1(x1))
```

où :
- `x1` est la valeur d'intensité dans l'image source
- `x2` est la valeur d'intensité correspondante dans l'image cible
- `cdf1` est la fonction de distribution cumulative de l'image source
- `cdf2` est la fonction de distribution cumulative de l'histogramme modèle

### Principe

1. Calculer l'histogramme et la CDF de l'image source
2. Définir un histogramme de référence (dans notre cas, un histogramme bimodal)
3. Calculer la CDF de l'histogramme de référence
4. Pour chaque niveau d'intensité `x1` dans l'image source :
   - Trouver la valeur de `cdf1(x1)`
   - Trouver la valeur `x2` telle que `cdf2(x2) = cdf1(x1)`
   - Remplacer `x1` par `x2` dans l'image résultante

Comme les valeurs d'intensité sont discrètes, une interpolation est nécessaire pour trouver la valeur exacte de `x2`.

## Implémentation

Deux implémentations sont fournies :

1. **Implémentation intégrée** : Utilise la fonction `match_histograms` de scikit-image
2. **Implémentation personnalisée** : Implémente l'algorithme d'appariement d'histogramme à partir de zéro

De plus, une fonction pour générer un histogramme bimodal de référence est fournie. Cet histogramme est créé en combinant deux distributions gaussiennes avec des paramètres ajustables.

## Utilisation en ligne de commande

### Appariement d'histogramme avec la fonction intégrée

```bash
python main.py matching --method builtin
```

Options :
- `--method builtin` : Utilise la fonction intégrée de scikit-image
- `--image chemin/vers/image.jpg` : Spécifie une image d'entrée (par défaut : phobos.jpg)
- `--bins 256` : Nombre de bins pour l'histogramme (par défaut : 256)
- `--peak1 0.3` : Position du premier pic dans l'histogramme bimodal (par défaut : 0.3)
- `--peak2 0.7` : Position du deuxième pic dans l'histogramme bimodal (par défaut : 0.7)
- `--sigma1 0.05` : Écart-type du premier pic (par défaut : 0.05)
- `--sigma2 0.05` : Écart-type du deuxième pic (par défaut : 0.05)
- `--weight1 0.6` : Poids du premier pic (par défaut : 0.6)
- `--weight2 0.4` : Poids du deuxième pic (par défaut : 0.4)
- `--output resultat.jpg` : Sauvegarde l'image après appariement

### Appariement d'histogramme avec l'implémentation personnalisée

```bash
python main.py matching --method custom
```

Options identiques à celles de la méthode intégrée.

### Comparaison des deux méthodes

```bash
python main.py matching --method both
```

Cette commande exécute les deux méthodes et affiche les résultats côte à côte pour comparaison.

## Résultats

L'appariement d'histogramme permet de transformer l'image pour qu'elle ait un histogramme spécifique. Contrairement à l'égalisation d'histogramme qui vise à obtenir un histogramme uniforme, l'appariement d'histogramme permet de cibler n'importe quelle distribution.

Dans notre implémentation, nous utilisons un histogramme bimodal comme référence, ce qui crée un effet de "posterisation" où les pixels sont regroupés autour de deux valeurs d'intensité principales.

### Avantages

- Plus flexible que l'égalisation d'histogramme
- Permet de cibler des distributions spécifiques
- Utile pour la normalisation d'images provenant de différentes sources

### Inconvénients

- Peut produire des artefacts non naturels
- Nécessite de définir un histogramme de référence approprié
- Plus complexe à implémenter que l'égalisation d'histogramme

## Auteur

Oussama GUELFAA
Date: 01-04-2025
