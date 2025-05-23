#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for machine learning module.

This module provides utility functions for data loading,
preprocessing, and other common tasks.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import time

def create_output_directory(base_dir='output/machine_learning'):
    """
    Create an output directory for saving results.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for output.
        
    Returns
    -------
    output_dir : str
        Path to the created output directory.
    """
    # Create a timestamp-based directory name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_model(classifier, scaler=None, output_dir='output/machine_learning', filename='model.pkl'):
    """
    Save a trained model to disk.
    
    Parameters
    ----------
    classifier : object
        Trained classifier.
    scaler : object, optional
        Fitted scaler.
    output_dir : str, optional
        Output directory.
    filename : str, optional
        Output filename.
        
    Returns
    -------
    model_path : str
        Path to the saved model.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model dictionary
    model_dict = {
        'classifier': classifier,
        'scaler': scaler
    }
    
    # Save model
    model_path = os.path.join(output_dir, filename)
    with open(model_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model.
        
    Returns
    -------
    classifier : object
        Trained classifier.
    scaler : object or None
        Fitted scaler if available, otherwise None.
    """
    # Load model
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Extract classifier and scaler
    classifier = model_dict['classifier']
    scaler = model_dict.get('scaler', None)
    
    return classifier, scaler

def save_results(results, output_dir='output/machine_learning', filename='results.txt'):
    """
    Save classification results to a text file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing results.
    output_dir : str, optional
        Output directory.
    filename : str, optional
        Output filename.
        
    Returns
    -------
    results_path : str
        Path to the saved results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(output_dir, filename)
    with open(results_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}:\n")
            f.write(f"{value}\n\n")
    
    print(f"Results saved to {results_path}")
    
    return results_path

def plot_learning_curve(classifier, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5),
                       figsize=(10, 6), output_dir=None, filename=None):
    """
    Plot learning curve for a classifier.
    
    Parameters
    ----------
    classifier : object
        Classifier object.
    X : ndarray
        Feature matrix.
    y : ndarray
        Label vector.
    cv : int, optional
        Number of cross-validation folds.
    train_sizes : ndarray, optional
        Training set sizes to plot.
    figsize : tuple, optional
        Figure size.
    output_dir : str, optional
        Output directory for saving the plot.
    filename : str, optional
        Output filename for saving the plot.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    from sklearn.model_selection import learning_curve
    
    # Calculate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        classifier, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1)
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot learning curve
    ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    # Set labels and title
    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curve')
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output_dir and filename are provided
    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {output_path}")
    
    return fig

def plot_validation_curve(classifier, X, y, param_name, param_range, cv=5,
                         figsize=(10, 6), output_dir=None, filename=None):
    """
    Plot validation curve for a classifier.
    
    Parameters
    ----------
    classifier : object
        Classifier object.
    X : ndarray
        Feature matrix.
    y : ndarray
        Label vector.
    param_name : str
        Parameter name to vary.
    param_range : list
        Parameter values to try.
    cv : int, optional
        Number of cross-validation folds.
    figsize : tuple, optional
        Figure size.
    output_dir : str, optional
        Output directory for saving the plot.
    filename : str, optional
        Output filename for saving the plot.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    """
    from sklearn.model_selection import validation_curve
    
    # Calculate validation curve
    train_scores, test_scores = validation_curve(
        classifier, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot validation curve
    ax.plot(param_range, train_mean, 'o-', color='r', label='Training score')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    ax.plot(param_range, test_mean, 'o-', color='g', label='Cross-validation score')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
    
    # Set labels and title
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(f'Validation Curve ({param_name})')
    ax.legend(loc='best')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save figure if output_dir and filename are provided
    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Validation curve saved to {output_path}")
    
    return fig
