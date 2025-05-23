"""
Classification module for image processing.

This module contains functions for training and evaluating classifiers on image features.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_classifier(X_train, y_train, classifier_type='mlp', **kwargs):
    """
    Train a classifier on the given data.
    
    Parameters
    ----------
    X_train : ndarray
        Training features.
    y_train : ndarray
        Training targets.
    classifier_type : str, optional
        Type of classifier to use ('mlp' or 'svm').
    **kwargs : dict
        Additional parameters for the classifier.
        
    Returns
    -------
    classifier : object
        Trained classifier.
    scaler : object
        Fitted scaler.
    """
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train the classifier
    if classifier_type.lower() == 'mlp':
        # Default parameters for MLP
        params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'max_iter': 1000,
            'random_state': 42
        }
        # Update with user parameters
        params.update(kwargs)
        classifier = MLPClassifier(**params)
    elif classifier_type.lower() == 'svm':
        # Default parameters for SVM
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42
        }
        # Update with user parameters
        params.update(kwargs)
        classifier = SVC(**params)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Train the classifier
    classifier.fit(X_train_scaled, y_train)
    
    return classifier, scaler

def evaluate_classifier(classifier, scaler, X_test, y_test, class_names=None):
    """
    Evaluate a classifier on test data.
    
    Parameters
    ----------
    classifier : object
        Trained classifier.
    scaler : object
        Fitted scaler.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test targets.
    class_names : list, optional
        Names of the classes.
        
    Returns
    -------
    accuracy : float
        Classification accuracy.
    cm : ndarray
        Confusion matrix.
    """
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    plt.tight_layout()
    plt.show()
    
    return accuracy, cm
