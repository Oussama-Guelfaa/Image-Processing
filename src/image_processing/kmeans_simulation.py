"""
Simulation example for k-means clustering in 2D

This script demonstrates the application of k-means clustering on 2D random points.
It generates 3 clusters of random points around specified centers and applies
k-means clustering to separate them.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def generation(n, x, y):
    """
    Generate n random points around the center (x, y).

    Args:
        n (int): Number of points to generate
        x (float): x-coordinate of the center
        y (float): y-coordinate of the center

    Returns:
        ndarray: Array of shape (n, 2) containing the generated points
    """
    Y = np.random.randn(n, 2) + np.array([[x, y]])
    return Y

def main():
    """
    Main function to demonstrate k-means clustering on random 2D points.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate 3 sets of points around specified centers
    n_points = 100  # Number of points per cluster

    # Centers: (0, 0), (3, 4), (-5, -3)
    points1 = generation(n_points, 0, 0)    # Cluster 1 around (0, 0)
    points2 = generation(n_points, 3, 4)    # Cluster 2 around (3, 4)
    points3 = generation(n_points, -5, -3)  # Cluster 3 around (-5, -3)

    # Concatenate all points into a single array
    all_points = np.concatenate((points1, points2, points3))

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(all_points)

    # Get the cluster centers
    centers = kmeans.cluster_centers_

    # Visualize the results
    plt.figure(figsize=(10, 8))

    # Plot the points with different colors based on the assigned clusters
    colors = ['blue', 'red', 'green']
    for i in range(3):
        plt.scatter(all_points[labels == i, 0], all_points[labels == i, 1],
                   c=colors[i], label=f'Cluster {i+1}', alpha=0.7)

    # Plot the cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100,
               linewidths=3, label='Cluster Centers')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('K-means Clustering of Random Points (k=3)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set equal aspect ratio
    plt.axis('equal')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Print additional information
    print("Cluster centers:")
    for i, center in enumerate(centers):
        print(f"Cluster {i+1}: ({center[0]:.2f}, {center[1]:.2f})")

    print("\nNumber of points in each cluster:")
    for i in range(3):
        print(f"Cluster {i+1}: {np.sum(labels == i)}")

if __name__ == "__main__":
    main()
