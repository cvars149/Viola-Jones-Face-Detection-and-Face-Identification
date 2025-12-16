"""
AdaBoost Classifier Implementation
Adaptive boosting for selecting and combining weak classifiers
"""

import numpy as np
from features import WeakClassifier


class AdaBoostClassifier:
    """
    AdaBoost classifier that combines multiple weak classifiers
    """
    
    def __init__(self, n_estimators=10):
        """
        Initialize AdaBoost classifier
        
        Args:
            n_estimators: Number of weak classifiers to select (T)
        """
        self.n_estimators = n_estimators
        self.weak_classifiers = []
        self.alphas = []
    
    def train(self, features, integral_images, labels, verbose=True):
        """
        Train AdaBoost classifier using the full algorithm
        
        Args:
            features: Pool of HaarFeature objects
            integral_images: Training integral images
            labels: Training labels (0 or 1)
            verbose: Print progress
            
        Returns:
            Training error history
        """
        n_samples = len(integral_images)
        n_features = len(features)
        
        # Initialize weights uniformly
        # Positive and negative classes get equal total weight
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        weights = np.zeros(n_samples)
        weights[labels == 1] = 1.0 / (2 * n_pos)
        weights[labels == 0] = 1.0 / (2 * n_neg)
        
        error_history = []
        
        if verbose:
            print(f"\n   Training AdaBoost with {n_features} features...")
        
        for t in range(self.n_estimators):
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Find best weak classifier
            if verbose:
                print(f"   Round {t+1}/{self.n_estimators}: ", end='')
            
            best_classifier, best_error = self._select_best_feature(
                features, integral_images, labels, weights, verbose
            )
            
            # Check for perfect classification
            if best_error == 0:
                best_error = 1e-10
            if best_error >= 0.5:
                if verbose:
                    print(f"Stopping: error {best_error:.4f} >= 0.5")
                break
            
            # Calculate classifier weight
            beta = best_error / (1 - best_error)
            alpha = np.log(1 / beta)
            
            # Update weights
            predictions = best_classifier.classify_batch(integral_images)
            correct = (predictions == labels).astype(float)
            
            # Decrease weight of correctly classified samples
            weights = weights * np.power(beta, correct)
            
            # Store classifier and weight
            self.weak_classifiers.append(best_classifier)
            self.alphas.append(alpha)
            error_history.append(best_error)
            
            if verbose:
                print(f"Error: {best_error:.4f}, Alpha: {alpha:.4f}")
        
        return error_history
    
    def _select_best_feature(self, features, integral_images, labels, 
                            weights, verbose):
        """
        Select the feature with lowest weighted error
        
        This is the most computationally intensive step
        """
        best_error = float('inf')
        best_classifier = None
        
        # Sample a subset of features for efficiency (optional optimization)
        # For full implementation, evaluate all features
        n_features_to_check = min(len(features), 5000)  # Limit for speed
        
        if len(features) > n_features_to_check:
            feature_indices = np.random.choice(
                len(features), n_features_to_check, replace=False
            )
            features_to_check = [features[i] for i in feature_indices]
        else:
            features_to_check = features
        
        # Evaluate each feature
        for i, feature in enumerate(features_to_check):
            classifier, error = WeakClassifier.train(
                feature, integral_images, labels, weights
            )
            
            if error < best_error:
                best_error = error
                best_classifier = classifier
            
            # Progress indicator
            if verbose and (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(features_to_check)}...", end='', flush=True)
        
        if verbose:
            print(" ", end='')
        
        return best_classifier, best_error
    
    def predict(self, integral_image):
        """
        Predict single sample
        
        Args:
            integral_image: Integral image to classify
            
        Returns:
            1 (face) or 0 (non-face)
        """
        score = self.predict_score(integral_image)
        return 1 if score >= 0.5 else 0
    
    def predict_score(self, integral_image):
        """
        Get confidence score for single sample
        
        Args:
            integral_image: Integral image
            
        Returns:
            Confidence score (0 to 1)
        """
        if len(self.weak_classifiers) == 0:
            return 0.5
        
        total = 0.0
        for classifier, alpha in zip(self.weak_classifiers, self.alphas):
            total += alpha * classifier.classify(integral_image)
        
        # Normalize by sum of alphas
        score = total / np.sum(self.alphas)
        return score
    
    def predict_batch(self, integral_images):
        """
        Predict multiple samples
        
        Args:
            integral_images: List of integral images
            
        Returns:
            Array of predictions
        """
        return np.array([self.predict(ii) for ii in integral_images])
    
    def evaluate(self, integral_images, labels):
        """
        Evaluate classifier on dataset
        
        Args:
            integral_images: Test integral images
            labels: True labels
            
        Returns:
            Accuracy, TPR, FPR
        """
        predictions = self.predict_batch(integral_images)
        
        accuracy = np.mean(predictions == labels)
        
        # True Positive Rate (Detection Rate)
        positives = labels == 1
        tpr = np.mean(predictions[positives] == 1) if np.any(positives) else 0
        
        # False Positive Rate
        negatives = labels == 0
        fpr = np.mean(predictions[negatives] == 1) if np.any(negatives) else 0
        
        return accuracy, tpr, fpr