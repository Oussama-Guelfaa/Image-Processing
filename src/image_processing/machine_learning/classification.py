#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine_learning

Machine learning techniques for image processing and analysis.

Author: Oussama GUELFAA
Date: 01-04-2025
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_test_split_dataset(features, labels, test_size=0.25, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    features : ndarray
        Feature matrix.
    labels : ndarray
        Label vector.
    test_size : float, optional
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train : ndarray
        Training features.
    X_test : ndarray
        Testing features.
    y_train : ndarray
        Training labels.
    y_test : ndarray
        Testing labels.
    """
    return train_test_split(features, labels, test_size=test_size,
                           random_state=random_state, stratify=labels)

def train_classifier(X_train, y_train, classifier_type='mlp', scale=True, **kwargs):
    """
    Train a classifier on the given data.

    Parameters
    ----------
    X_train : ndarray
        Training features.
    y_train : ndarray
        Training labels.
    classifier_type : str, optional
        Type of classifier to use. Options: 'mlp', 'svm', 'rf', 'ensemble'
    scale : bool, optional
        Whether to scale the features.
    **kwargs : dict
        Additional arguments to pass to the classifier.

    Returns
    -------
    classifier : object
        Trained classifier.
    scaler : object or None
        Fitted scaler if scale=True, otherwise None.
    """
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        scaler = None
        X_train_scaled = X_train

    # Create classifier based on type
    if classifier_type == 'mlp':
        # Default parameters for MLP
        mlp_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 1000,
            'random_state': 42
        }
        # Update with user-provided parameters
        mlp_params.update(kwargs)
        classifier = MLPClassifier(**mlp_params)

    elif classifier_type == 'svm':
        # Default parameters for SVM
        svm_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        # Update with user-provided parameters
        svm_params.update(kwargs)
        classifier = SVC(**svm_params)

    elif classifier_type == 'rf':
        # Default parameters for Random Forest
        rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        # Update with user-provided parameters
        rf_params.update(kwargs)
        classifier = RandomForestClassifier(**rf_params)

    elif classifier_type == 'ensemble':
        # Create a voting ensemble of classifiers
        from sklearn.ensemble import VotingClassifier

        # Default parameters for ensemble components
        rf_params = {
            'n_estimators': 200,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        }
        svm_params = {
            'C': 10.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        mlp_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 1000,
            'random_state': 42
        }

        # Create individual classifiers
        rf = RandomForestClassifier(**rf_params)
        svm = SVC(**svm_params)
        mlp = MLPClassifier(**mlp_params)

        # Create voting classifier
        classifier = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm), ('mlp', mlp)],
            voting='soft'
        )

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Train the classifier
    classifier.fit(X_train_scaled, y_train)

    return classifier, scaler

def evaluate_classifier(classifier, X_test, y_test, scaler=None):
    """
    Evaluate a trained classifier on test data.

    Parameters
    ----------
    classifier : object
        Trained classifier.
    X_test : ndarray
        Test features.
    y_test : ndarray
        Test labels.
    scaler : object or None, optional
        Fitted scaler to transform test features.

    Returns
    -------
    accuracy : float
        Classification accuracy.
    y_pred : ndarray
        Predicted labels.
    report : str
        Classification report.
    conf_matrix : ndarray
        Confusion matrix.
    """
    # Scale features if scaler is provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    # Make predictions
    y_pred = classifier.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate classification report
    report = classification_report(y_test, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return accuracy, y_pred, report, conf_matrix

def cross_validate(features, labels, classifier_type='mlp', n_splits=5, scale=True, **kwargs):
    """
    Perform cross-validation on the dataset.

    Parameters
    ----------
    features : ndarray
        Feature matrix.
    labels : ndarray
        Label vector.
    classifier_type : str, optional
        Type of classifier to use. Options: 'mlp', 'svm', 'rf'
    n_splits : int, optional
        Number of folds for cross-validation.
    scale : bool, optional
        Whether to scale the features.
    **kwargs : dict
        Additional arguments to pass to the classifier.

    Returns
    -------
    cv_scores : ndarray
        Cross-validation scores.
    mean_score : float
        Mean cross-validation score.
    std_score : float
        Standard deviation of cross-validation scores.
    """
    # Scale features if requested
    if scale:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features

    # Create classifier based on type
    if classifier_type == 'mlp':
        # Default parameters for MLP
        mlp_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'max_iter': 1000,
            'random_state': 42
        }
        # Update with user-provided parameters
        mlp_params.update(kwargs)
        classifier = MLPClassifier(**mlp_params)

    elif classifier_type == 'svm':
        # Default parameters for SVM
        svm_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        # Update with user-provided parameters
        svm_params.update(kwargs)
        classifier = SVC(**svm_params)

    elif classifier_type == 'rf':
        # Default parameters for Random Forest
        rf_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        # Update with user-provided parameters
        rf_params.update(kwargs)
        classifier = RandomForestClassifier(**rf_params)

    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(classifier, features_scaled, labels, cv=cv, scoring='accuracy')

    # Calculate mean and standard deviation
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    return cv_scores, mean_score, std_score

def classify_image(image, classifier, scaler=None, feature_types=None):
    """
    Classify a single image using a trained classifier.

    Parameters
    ----------
    image : ndarray
        Input image.
    classifier : object
        Trained classifier.
    scaler : object or None, optional
        Fitted scaler to transform features.
    feature_types : list, optional
        List of feature types to extract.

    Returns
    -------
    label : int
        Predicted class label.
    probability : float
        Probability of the predicted class.
    """
    # Import feature extraction function
    from .feature_extraction import extract_features

    # Extract features
    features = extract_features(image, feature_types)
    features = features.reshape(1, -1)

    # Scale features if scaler is provided
    if scaler is not None:
        features = scaler.transform(features)

    # Make prediction
    label = classifier.predict(features)[0]

    # Get probability if available
    if hasattr(classifier, 'predict_proba'):
        probabilities = classifier.predict_proba(features)[0]
        probability = probabilities[label]
    else:
        probability = None

    return label, probability

def main():
    """
    Main function to demonstrate classification capabilities.
    This function can be run directly to test the classification module.
    """
    import os
    import pickle
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs('output/machine_learning', exist_ok=True)

    print("Classification Module Demo")
    print("=========================")

    # Import feature extraction functions
    from .feature_extraction import load_kimia_dataset, extract_dataset_features

    # Load dataset
    print("\nLoading dataset...")
    data_dir = 'data/images_Kimia'
    if not os.path.exists(data_dir):
        print(f"Dataset directory {data_dir} not found. Using synthetic data instead.")
        # Create synthetic data for demonstration
        features = np.random.rand(100, 10)
        labels = np.random.randint(0, 5, 100)
        class_names = [f'Class_{i}' for i in range(5)]
    else:
        # Load real dataset
        images, labels, class_names = load_kimia_dataset(data_dir, max_images_per_class=10)
        print(f"Loaded {len(images)} images from {len(class_names)} classes")

        # Extract features
        print("Extracting features...")
        features = extract_dataset_features(images, feature_types=['hu', 'zernike', 'geometric'], verbose=True)

    # Split dataset
    print("\nSplitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split_dataset(features, labels, test_size=0.3)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")

    # Train classifiers
    classifiers = ['mlp', 'svm', 'rf']
    results = {}

    for clf_type in classifiers:
        print(f"\nTraining {clf_type.upper()} classifier...")
        classifier, scaler = train_classifier(X_train, y_train, classifier_type=clf_type)

        # Evaluate classifier
        print(f"Evaluating {clf_type.upper()} classifier...")
        accuracy, y_pred, report, conf_matrix = evaluate_classifier(classifier, X_test, y_test, scaler)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

        results[clf_type] = {
            'accuracy': accuracy,
            'classifier': classifier,
            'scaler': scaler,
            'y_pred': y_pred
        }

    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores, mean_score, std_score = cross_validate(features, labels, classifier_type='rf', n_splits=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")

    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(10, 6))
    accuracies = [results[clf]['accuracy'] for clf in classifiers]
    plt.bar(classifiers, accuracies)
    plt.ylim(0, 1.0)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Performance Comparison')
    plt.savefig('output/machine_learning/classifier_comparison.png')
    plt.close()

    # Save best model
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest classifier: {best_clf.upper()} with accuracy {results[best_clf]['accuracy']:.4f}")

    model_path = 'output/machine_learning/best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'classifier': results[best_clf]['classifier'],
            'scaler': results[best_clf]['scaler'],
            'accuracy': results[best_clf]['accuracy'],
            'class_names': class_names
        }, f)

    print(f"Best model saved to {model_path}")
    print("\nClassification demo completed successfully!")

if __name__ == "__main__":
    main()
