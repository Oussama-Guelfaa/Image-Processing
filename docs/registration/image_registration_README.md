# Image Registration avec la méthode ICP (Iterative Closest Point)

Ce document explique la théorie et l'implémentation de la méthode ICP (Iterative Closest Point) pour l'enregistrement d'images, en particulier pour estimer une transformation rigide (translation + rotation sans mise à l'échelle) entre deux images.

## Table des matières

1. [Introduction](#introduction)
2. [Théorie](#théorie)
   - [Transformation rigide](#transformation-rigide)
   - [Estimation de la transformation](#estimation-de-la-transformation)
   - [Méthode ICP](#méthode-icp)
3. [Implémentation](#implémentation)
   - [Estimation de la transformation rigide](#estimation-de-la-transformation-rigide)
   - [Application de la transformation](#application-de-la-transformation)
   - [Algorithme ICP](#algorithme-icp)
   - [Détection automatique des points de contrôle](#détection-automatique-des-points-de-contrôle)
   - [Sélection manuelle de points correspondants](#sélection-manuelle-de-points-correspondants)
   - [Superposition d'images](#superposition-dimages)
4. [Résultats](#résultats)
5. [Conclusion](#conclusion)
6. [Références](#références)

## Introduction

L'enregistrement d'images est le processus d'alignement de deux ou plusieurs images de la même scène. Cette technique est largement utilisée dans le traitement d'images médicales, la vision par ordinateur, et la télédétection. Dans ce projet, nous nous concentrons sur l'enregistrement d'images IRM du cerveau en utilisant la méthode ICP.

L'objectif est d'estimer une transformation rigide (rotation + translation sans mise à l'échelle) qui aligne au mieux une image source (mobile) avec une image cible.

## Théorie

### Transformation rigide

Une transformation rigide préserve les distances entre les points et est composée d'une rotation et d'une translation. Mathématiquement, pour un point p, la transformation rigide T est définie comme :

```
T(p) = R·p + t
```

où R est une matrice de rotation 2×2 et t est un vecteur de translation.

### Estimation de la transformation

Pour estimer la transformation rigide optimale entre deux ensembles de points correspondants {p_i} et {q_i}, nous minimisons le critère des moindres carrés :

```
C(R, t) = ∑_i ||q_i - R·p_i - t||²
```

#### Calcul de la translation

La translation optimale est caractérisée par une dérivée nulle du critère :

```
∂C/∂t = -2·∑_i (q_i - R·p_i - t)ᵀ = 0
```

Ce qui donne :

```
∑_i q_i - R·(∑_i p_i) = N·t
```

En définissant les centres de masse p̄ = (1/N)·∑_i p_i et q̄ = (1/N)·∑_i q_i, la translation estimée est :

```
t̂ = q̄ - R̂·p̄
```

#### Calcul de la rotation par la méthode SVD

Pour calculer la rotation optimale, nous utilisons la décomposition en valeurs singulières (SVD). Après avoir centré les points (p_i' = p_i - p̄, q_i' = q_i - q̄), nous calculons la matrice K = q'ᵀ·p'.

Soit U·D·Vᵀ la SVD de K. La rotation optimale est donnée par :

```
R̂ = U·S·Vᵀ
```

où S est une matrice diagonale définie comme :

```
S = [1    0]
    [0  det(U)·det(V)]
```

Cette formulation garantit que R̂ est une matrice de rotation propre (déterminant = 1).

### Méthode ICP

Lorsque les points ne sont pas correctement appariés, il est nécessaire de les réordonner avant d'estimer la transformation. La méthode ICP (Iterative Closest Point) consiste en un processus itératif :

1. Trouver la correspondance entre les points (plus proches voisins)
2. Estimer la transformation rigide
3. Appliquer la transformation
4. Répéter jusqu'à convergence

Cette méthode permet de trouver automatiquement les correspondances entre deux ensembles de points et d'estimer la transformation optimale.

## Implémentation

### Estimation de la transformation rigide

La fonction `estimate_rigid_transform` implémente l'estimation de la transformation rigide entre deux ensembles de points :

```python
def estimate_rigid_transform(source_points, target_points):
    # 1. Compute the center of each set of points
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)

    # 2. Subtract centers to get centered coordinates
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # 3. Compute the matrix K = target_centered.T @ source_centered
    K = target_centered.T @ source_centered

    # 4. Use SVD decomposition
    U, _, Vt = linalg.svd(K)

    # 5. Evaluate the rotation matrix
    # Ensure proper rotation matrix (right-handed coordinate system)
    S = np.eye(2)
    S[1, 1] = np.linalg.det(U) * np.linalg.det(Vt.T)

    R = U @ S @ Vt

    # 6. Evaluate the translation
    t = target_center - R @ source_center

    return R, t
```

### Application de la transformation

La fonction `apply_rigid_transform` applique la transformation rigide à une image en utilisant la fonction `cv2.warpAffine` :

```python
def apply_rigid_transform(image, R, t):
    # Create affine transformation matrix
    M = np.zeros((2, 3))
    M[:2, :2] = R
    M[:, 2] = t

    # Apply transformation
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, M, (cols, rows))

    return transformed_image
```

### Algorithme ICP

La fonction `icp_registration` implémente l'algorithme ICP :

```python
def icp_registration(source_points, target_points, max_iterations=20, tolerance=1e-6):
    # Initialize
    current_points = source_points.copy()
    prev_error = float('inf')

    for i in range(max_iterations):
        # 1. Find nearest neighbors
        matched_source, matched_target = find_nearest_neighbors(current_points, target_points)

        # 2. Estimate transformation
        R, t = estimate_rigid_transform(matched_source, matched_target)

        # 3. Apply transformation
        current_points = transform_points(current_points, R, t)

        # 4. Compute error
        error = np.mean(np.sum((matched_target - current_points) ** 2, axis=1))

        # 5. Check convergence
        if abs(prev_error - error) < tolerance:
            break

        prev_error = error

    # Compute final transformation (from original source to final position)
    final_R, final_t = estimate_rigid_transform(source_points, current_points)

    return final_R, final_t, current_points, error
```

La fonction `find_nearest_neighbors` utilise un arbre KD (KD-Tree) pour trouver efficacement les plus proches voisins :

```python
def find_nearest_neighbors(source_points, target_points):
    # Build KD-Tree for target points
    tree = cKDTree(target_points)

    # Find nearest neighbors
    distances, indices = tree.query(source_points)

    # Get matched points
    matched_target = target_points[indices]

    return source_points, matched_target
```

### Détection automatique des points de contrôle

La fonction `detect_corners` utilise le détecteur de coins de Shi-Tomasi (implémenté dans OpenCV via `goodFeaturesToTrack`) pour détecter automatiquement les points caractéristiques dans les images :

```python
def detect_corners(image, max_corners=50, quality_level=0.01, min_distance=10):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    # Ensure image is in the right format
    gray = (gray * 255).astype(np.uint8)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    # Reshape to (N, 2)
    if corners is not None:
        corners = corners.reshape(-1, 2)
    else:
        corners = np.array([])

    return corners
```

### Superposition d'images

La fonction `superimpose` permet de visualiser la superposition de deux images en niveaux de gris, ce qui est utile pour évaluer visuellement la qualité de l'enregistrement :

```python
def superimpose(G1, G2, filename=None):
    """
    Superimpose 2 images, supposing they are grayscale images and of same shape.
    For display purposes.
    """
    r, c = G1.shape
    S = np.zeros((r, c, 3))

    S[:,:,0] = np.maximum(G1-G2, 0) + G1
    S[:,:,1] = np.maximum(G2-G1, 0) + G2
    S[:,:,2] = (G1+G2) / 2

    S = 255 * S / np.max(S)
    S = S.astype('uint8')

    plt.figure(figsize=(10, 8))
    plt.imshow(S)
    plt.title("Superimposed Images")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if filename is not None:
        cv2.imwrite(filename, cv2.cvtColor(S, cv2.COLOR_RGB2BGR))

    return S
```

Cette fonction crée une image RGB où :
- Le canal rouge (R) contient les parties de la première image qui sont plus intenses que la seconde, plus la première image
- Le canal vert (G) contient les parties de la seconde image qui sont plus intenses que la première, plus la seconde image
- Le canal bleu (B) contient la moyenne des deux images

Cette représentation permet de visualiser facilement les différences entre les deux images et d'évaluer la qualité de l'alignement.

### Sélection manuelle de points correspondants

La sélection manuelle de points correspondants est implémentée à l'aide d'OpenCV. Cette approche permet à l'utilisateur de sélectionner interactivement des paires de points de contrôle sur les deux images :

```python
def on_mouse(event, x, y, flags, param):
    """
    Callback method for detecting click on image.
    It draws a circle on the global variable image I.
    """
    global pts, I
    if event == cv2.EVENT_LBUTTONUP:
        pts.append((x, y))
        cv2.circle(I, (x, y), 2, (0, 0, 255), -1)


def cpselect(image, title="Select Points"):
    """
    Method for manually selecting the control points.
    It waits until 'q' key is pressed.
    """
    global pts, I

    # Reset points list
    pts = []

    # Make a copy of the image for display
    if len(image.shape) == 2:  # Grayscale image
        I = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:  # Color image
        I = image.copy()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)

    print(f"Select points on '{title}' and press 'q' when finished")

    # Keep looping until the 'q' key is pressed
    while True:
        # Display the image and wait for a keypress
        cv2.imshow(title, I)
        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

    # Close all open windows
    cv2.destroyAllWindows()

    return pts
```

La fonction `rigid_registration` estime la transformation rigide à partir des points correspondants :

```python
def rigid_registration(data1, data2):
    """
    Rigid transformation estimation between n pairs of points.
    This function returns a rotation R and a translation t.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Computes barycenters, and recenters the points
    m1 = np.mean(data1, 0)
    m2 = np.mean(data2, 0)
    data1_inv_shifted = data1 - m1
    data2_inv_shifted = data2 - m2

    # Evaluates SVD
    K = np.matmul(np.transpose(data2_inv_shifted), data1_inv_shifted)
    U, S, V = np.linalg.svd(K)

    # Computes Rotation
    S = np.eye(2)
    S[1, 1] = np.linalg.det(U) * np.linalg.det(V)
    R = np.matmul(U, S)
    R = np.matmul(R, np.transpose(V))

    # Computes Translation
    t = m2 - np.matmul(R, m1)

    T = np.zeros((2, 3))
    T[0:2, 0:2] = R
    T[0:2, 2] = t

    return T
```

La transformation est ensuite appliquée à l'image source avec OpenCV :

```python
def applyTransform(points, T):
    """
    Apply transform to a list of points.
    """
    dataA = np.array(points)
    src = np.array([dataA])
    data_dest = cv2.transform(src, T)
    a, b, c = data_dest.shape
    data_dest = np.reshape(data_dest, (b, c))

    return data_dest

# Application de la transformation à l'image
rows, cols = target_image.shape[:2]
registered_image = cv2.warpAffine(source_image, T, (cols, rows))
```

## Résultats

Nous avons testé notre implémentation sur des images IRM du cerveau (brain1.png et brain2.png). Voici les résultats obtenus :

### 1. Estimation directe de la transformation rigide

Nous avons d'abord utilisé des points de contrôle prédéfinis pour estimer directement la transformation rigide :

```
A_points = np.array([[136, 100], [127, 153], [96, 156], [87, 99]])
B_points = np.array([[144, 99], [109, 140], [79, 128], [100, 74]])
```

La transformation estimée a été appliquée à l'image source, ce qui a donné un bon alignement avec l'image cible.

### 2. Méthode ICP avec points mélangés

Pour tester la robustesse de la méthode ICP, nous avons mélangé aléatoirement les points de contrôle de l'image source. Malgré ce mélange, l'algorithme ICP a réussi à trouver les bonnes correspondances et à estimer une transformation similaire à celle obtenue avec les points correctement appariés.

### 3. Détection automatique des points de contrôle

Nous avons également testé la détection automatique des points de contrôle en utilisant le détecteur de coins de Shi-Tomasi. Cette approche a permis d'éviter la sélection manuelle des points, mais a nécessité l'utilisation de l'algorithme ICP pour trouver les correspondances entre les points détectés dans les deux images.

## Conclusion

L'enregistrement d'images par la méthode ICP est une technique puissante pour aligner des images, en particulier lorsque les correspondances exactes entre les points ne sont pas connues à l'avance. Notre implémentation a démontré l'efficacité de cette méthode pour l'enregistrement d'images IRM du cerveau.

Les principales contributions de ce projet sont :

1. Une implémentation complète de l'estimation de la transformation rigide par la méthode SVD
2. Une implémentation de l'algorithme ICP pour l'enregistrement d'images
3. L'intégration de la détection automatique des points de contrôle

Ces techniques peuvent être étendues à d'autres types d'images et à des transformations plus complexes (affines, projectives, etc.).

## Références

1. Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2), 239-256.
2. Arun, K. S., Huang, T. S., & Blostein, S. D. (1987). Least-squares fitting of two 3-D point sets. IEEE Transactions on Pattern Analysis and Machine Intelligence, 9(5), 698-700.
3. Shi, J., & Tomasi, C. (1994). Good features to track. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (pp. 593-600).
4. OpenCV Documentation: https://docs.opencv.org/
5. SciPy Documentation: https://docs.scipy.org/

## Auteur

Oussama GUELFAA
Date: 01-04-2025
