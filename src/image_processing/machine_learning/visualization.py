#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine_learning

Machine learning techniques for image processing and analysis.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8), cmap='Blues', normalize=None):
    """
    Plot a confusion matrix for classification results.

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    class_names : list, optional
        List of class names.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Colormap for the confusion matrix.
    normalize : str, optional
        Normalization method. Options: None, 'true', 'pred', 'all'

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=cmap, ax=ax)

    # Set title based on normalization
    if normalize is None:
        ax.set_title('Confusion Matrix')
    elif normalize == 'true':
        ax.set_title('Confusion Matrix (Normalized by True Labels)')
    elif normalize == 'pred':
        ax.set_title('Confusion Matrix (Normalized by Predicted Labels)')
    elif normalize == 'all':
        ax.set_title('Confusion Matrix (Normalized by All)')

    plt.tight_layout()

    return fig, ax

def plot_feature_importance(classifier, feature_names=None, figsize=(12, 8)):
    """
    Plot feature importance for tree-based classifiers.

    Parameters
    ----------
    classifier : object
        Trained classifier with feature_importances_ attribute.
    feature_names : list, optional
        List of feature names.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    # Check if classifier has feature_importances_ attribute
    if not hasattr(classifier, 'feature_importances_'):
        raise ValueError("Classifier does not have feature_importances_ attribute")

    # Get feature importances
    importances = classifier.feature_importances_

    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = [feature_names[i] for i in indices]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot feature importances
    ax.bar(range(len(sorted_importances)), sorted_importances)
    ax.set_xticks(range(len(sorted_importances)))
    ax.set_xticklabels(sorted_feature_names, rotation=90)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')

    plt.tight_layout()

    return fig, ax

def visualize_dataset(features, labels, class_names=None, method='pca', figsize=(10, 8)):
    """
    Visualize dataset using dimensionality reduction.

    Parameters
    ----------
    features : ndarray
        Feature matrix.
    labels : ndarray
        Label vector.
    class_names : list, optional
        List of class names.
    method : str, optional
        Dimensionality reduction method. Options: 'pca', 'tsne'
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in np.unique(labels)]

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(features)
        title = 'PCA Visualization'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        title = 't-SNE Visualization'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot reduced features
    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax.scatter(reduced_features[mask, 0], reduced_features[mask, 1], label=class_name, alpha=0.7)

    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    return fig, ax

def visualize_classification_results(images, y_true, y_pred, class_names, figsize=(15, 10), max_images=10):
    """
    Visualize classification results by showing images and their predicted labels.

    Parameters
    ----------
    images : list
        List of images.
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    class_names : list
        List of class names.
    figsize : tuple, optional
        Figure size.
    max_images : int, optional
        Maximum number of images to display.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    # Find misclassified images
    misclassified = np.where(y_true != y_pred)[0]

    # Limit the number of images to display
    if len(misclassified) > max_images:
        misclassified = np.random.choice(misclassified, max_images, replace=False)

    # Calculate grid size
    n_images = len(misclassified)
    if n_images == 0:
        print("No misclassified images found.")
        return None

    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot misclassified images
    for i, idx in enumerate(misclassified):
        if i < len(axes):
            axes[i].imshow(images[idx], cmap='gray')
            axes[i].set_title(f'True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}')
            axes[i].axis('off')

    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    return fig

def save_visualization(fig, filename, output_dir='output', dpi=300):
    """
    Save visualization figure to file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    filename : str
        Output filename.
    output_dir : str, optional
        Output directory.
    dpi : int, optional
        DPI for the output image.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

    return output_path

def main():
    """
    Main function to demonstrate visualization capabilities.
    This function can be run directly to test the visualization module.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Create output directory
    output_dir = 'output/machine_learning'
    os.makedirs(output_dir, exist_ok=True)

    print("Visualization Module Demo")
    print("========================")

    # Generate synthetic data for demonstration
    print("\nGenerating synthetic data...")
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        random_state=42
    )

    # Create class names
    class_names = [f'Class {i}' for i in range(5)]
    print(f"Generated dataset with {X.shape[0]} samples, {X.shape[1]} features, and {len(class_names)} classes")

    # Train a simple classifier
    print("\nTraining a Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict(X)

    # 1. Visualize dataset using PCA
    print("\nVisualizing dataset using PCA...")
    fig_pca, ax_pca = visualize_dataset(X, y, class_names=class_names, method='pca')
    save_visualization(fig_pca, 'pca_visualization.png', output_dir=output_dir)
    plt.close(fig_pca)

    # 2. Visualize dataset using t-SNE
    print("\nVisualizing dataset using t-SNE...")
    fig_tsne, ax_tsne = visualize_dataset(X, y, class_names=class_names, method='tsne')
    save_visualization(fig_tsne, 'tsne_visualization.png', output_dir=output_dir)
    plt.close(fig_tsne)

    # 3. Plot confusion matrix
    print("\nPlotting confusion matrix...")
    fig_cm, ax_cm = plot_confusion_matrix(y, y_pred, class_names=class_names)
    save_visualization(fig_cm, 'confusion_matrix.png', output_dir=output_dir)
    plt.close(fig_cm)

    # 4. Plot feature importance
    print("\nPlotting feature importance...")
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    fig_fi, ax_fi = plot_feature_importance(clf, feature_names=feature_names)
    save_visualization(fig_fi, 'feature_importance.png', output_dir=output_dir)
    plt.close(fig_fi)

    # 5. Generate some synthetic images for visualization
    print("\nGenerating synthetic images for visualization...")
    images = []
    for i in range(20):
        # Create a simple image with a shape
        img = np.zeros((50, 50))
        if i % 5 == 0:  # Circle
            rr, cc = np.ogrid[:50, :50]
            circle = (rr - 25) ** 2 + (cc - 25) ** 2 <= 15 ** 2
            img[circle] = 1
        elif i % 5 == 1:  # Square
            img[15:35, 15:35] = 1
        elif i % 5 == 2:  # Triangle
            for j in range(20):
                width = 2 * j
                if 15 + j < 35:
                    img[15 + j, 25 - width//2:25 + width//2] = 1
        elif i % 5 == 3:  # Cross
            img[20:30, 10:40] = 1
            img[10:40, 20:30] = 1
        else:  # Diamond
            for j in range(15):
                width = 2 * j
                img[25 - j, 25 - width//2:25 + width//2] = 1
                img[25 + j, 25 - width//2:25 + width//2] = 1
        images.append(img)

    # Create some misclassifications for demonstration
    y_true = np.array([i % 5 for i in range(20)])
    y_pred = y_true.copy()
    # Introduce some misclassifications
    y_pred[2] = (y_true[2] + 1) % 5
    y_pred[7] = (y_true[7] + 2) % 5
    y_pred[12] = (y_true[12] + 1) % 5
    y_pred[17] = (y_true[17] + 3) % 5

    # 6. Visualize classification results
    print("\nVisualizing classification results...")
    fig_results = visualize_classification_results(
        images, y_true, y_pred,
        class_names=['Circle', 'Square', 'Triangle', 'Cross', 'Diamond']
    )
    save_visualization(fig_results, 'classification_results.png', output_dir=output_dir)
    plt.close(fig_results)

    print("\nVisualization demo completed successfully!")
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
