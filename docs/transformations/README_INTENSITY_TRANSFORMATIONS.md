# Transformations d'Intensité (LUT)

Ce module implémente deux transformations d'intensité principales pour l'amélioration d'images :

1. **Correction gamma (γ correction)** - modifie la luminosité globale de l'image
2. **Étirement de contraste (contrast stretching)** - améliore le contraste en utilisant une fonction non linéaire

## Théorie

### Correction Gamma

La correction gamma est définie par la formule :

```
s = r^γ
```

où :
- `r` est la valeur d'entrée (entre 0 et 1)
- `s` est la valeur de sortie
- `γ` est le paramètre gamma

Effets du paramètre gamma :
- `γ < 1` : Éclaircit l'image, améliore les détails dans les zones sombres
- `γ = 1` : Aucun changement
- `γ > 1` : Assombrit l'image, améliore les détails dans les zones claires

### Étirement de Contraste

L'étirement de contraste est implémenté avec la fonction de distribution cumulative suivante :

```
s = T(r) = 1 / (1 + (m/r)^E)
```

où :
- `r` est la valeur d'entrée (entre 0 et 1)
- `s` est la valeur de sortie
- `m` est la valeur moyenne de gris de l'image
- `E` est un paramètre qui contrôle la pente de la transformation

Effets du paramètre E :
- Plus `E` est grand, plus la transition entre les zones sombres et claires est abrupte
- Plus `E` est petit, plus la transition est douce

## Utilisation en ligne de commande

### Correction Gamma

```bash
python main.py intensity --type gamma --gamma 0.5
```

Options :
- `--type gamma` : Applique uniquement la correction gamma
- `--gamma 0.5` : Définit la valeur du paramètre gamma (valeur par défaut : 2.0)
- `--image chemin/vers/image.jpg` : Spécifie une image d'entrée (par défaut : osteoblaste.jpg)
- `--output resultat.jpg` : Sauvegarde l'image transformée

### Étirement de Contraste

```bash
python main.py intensity --type contrast --E 4.0
```

Options :
- `--type contrast` : Applique uniquement l'étirement de contraste
- `--E 4.0` : Définit la valeur du paramètre E (valeur par défaut : 4.0)
- `--image chemin/vers/image.jpg` : Spécifie une image d'entrée (par défaut : osteoblaste.jpg)
- `--output resultat.jpg` : Sauvegarde l'image transformée

### Les deux transformations

```bash
python main.py intensity --type both --gamma 0.5 --E 4.0
```

Options :
- `--type both` : Applique les deux transformations et affiche les résultats
- `--gamma 0.5` : Définit la valeur du paramètre gamma
- `--E 4.0` : Définit la valeur du paramètre E

## Exemples de résultats

### Correction Gamma

- `γ = 0.5` : Éclaircit l'image, améliore les détails dans les zones sombres
- `γ = 1.0` : Aucun changement
- `γ = 2.0` : Assombrit l'image, améliore les détails dans les zones claires

### Étirement de Contraste

- `E = 1.0` : Transition douce entre les zones sombres et claires
- `E = 4.0` : Transition plus abrupte, meilleur contraste
- `E = 8.0` : Transition très abrupte, effet presque binaire

## Auteur

Oussama GUELFAA
Date: 01-04-2025
