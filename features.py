"""
Haar Feature Generation Module
Implements all Haar-like features for Viola-Jones detector
"""

import numpy as np
from preprocessing import IntegralImage


class HaarFeature:
    """
    Represents a single Haar-like feature
    Consists of positive and negative rectangular regions
    """
    
    def __init__(self, feature_type, position, size):
        """
        Initialize Haar feature
        
        Args:
            feature_type: Type of feature ('two_horizontal', 'two_vertical', 
                         'three_horizontal', 'three_vertical', 'four_diagonal')
            position: (x, y) top-left position in window
            size: (w, h) base size of rectangles
        """
        self.feature_type = feature_type
        self.position = position
        self.size = size
        
        # Define positive and negative regions based on type
        self.positive_rects = []
        self.negative_rects = []
        
        self._define_rectangles()
    
    def _define_rectangles(self):
        """Define the positive and negative rectangles for this feature"""
        x, y = self.position
        w, h = self.size
        
        if self.feature_type == 'two_horizontal':
            # Left rect (negative), Right rect (positive)
            self.negative_rects = [(x, y, w, h)]
            self.positive_rects = [(x + w, y, w, h)]
        
        elif self.feature_type == 'two_vertical':
            # Top rect (negative), Bottom rect (positive)
            self.negative_rects = [(x, y, w, h)]
            self.positive_rects = [(x, y + h, w, h)]
        
        elif self.feature_type == 'three_horizontal':
            # Left and right (negative), Center (positive)
            self.negative_rects = [(x, y, w, h), (x + 2*w, y, w, h)]
            self.positive_rects = [(x + w, y, w, h)]
        
        elif self.feature_type == 'three_vertical':
            # Top and bottom (negative), Center (positive)
            self.negative_rects = [(x, y, w, h), (x, y + 2*h, w, h)]
            self.positive_rects = [(x, y + h, w, h)]
        
        elif self.feature_type == 'four_diagonal':
            # Diagonal checkerboard pattern
            self.positive_rects = [(x, y, w, h), (x + w, y + h, w, h)]
            self.negative_rects = [(x + w, y, w, h), (x, y + h, w, h)]
    
    def compute(self, integral_image):
        """
        Compute feature value using integral image
        
        Args:
            integral_image: Precomputed integral image
            
        Returns:
            Feature response (sum of positive regions - sum of negative regions)
        """
        positive_sum = sum([
            IntegralImage.sum_region(integral_image, x, y, w, h)
            for x, y, w, h in self.positive_rects
        ])
        
        negative_sum = sum([
            IntegralImage.sum_region(integral_image, x, y, w, h)
            for x, y, w, h in self.negative_rects
        ])
        
        return positive_sum - negative_sum
    
    def get_total_width(self):
        """Get total width of feature"""
        if self.feature_type in ['two_horizontal', 'four_diagonal']:
            return 2 * self.size[0]
        elif self.feature_type == 'three_horizontal':
            return 3 * self.size[0]
        else:
            return self.size[0]
    
    def get_total_height(self):
        """Get total height of feature"""
        if self.feature_type in ['two_vertical', 'four_diagonal']:
            return 2 * self.size[1]
        elif self.feature_type == 'three_vertical':
            return 3 * self.size[1]
        else:
            return self.size[1]


class HaarFeatureGenerator:
    """
    Generates the complete pool of Haar features for a detection window
    """
    
    def __init__(self, window_size=(16, 16), min_feature_size=(1, 1)):
        """
        Initialize feature generator
        
        Args:
            window_size: Size of detection window (w, h)
            min_feature_size: Minimum size of base rectangle (w, h)
        """
        self.window_width, self.window_height = window_size
        self.min_width, self.min_height = min_feature_size
        
        self.feature_types = [
            'two_horizontal',
            'two_vertical',
            'three_horizontal',
            'three_vertical',
            'four_diagonal'
        ]
    
    def generate_all_features(self):
        """
        Generate all possible Haar features for the window
        
        Returns:
            List of HaarFeature objects
        """
        features = []
        
        print("   Generating features by type:")
        
        for feature_type in self.feature_types:
            type_features = self._generate_features_of_type(feature_type)
            features.extend(type_features)
            print(f"     {feature_type}: {len(type_features)} features")
        
        return features
    
    def _generate_features_of_type(self, feature_type):
        """Generate all features of a specific type"""
        features = []
        
        # Iterate over all possible base rectangle sizes
        for w in range(self.min_width, self.window_width + 1):
            for h in range(self.min_height, self.window_height + 1):
                
                # Iterate over all possible positions
                for x in range(0, self.window_width):
                    for y in range(0, self.window_height):
                        
                        # Create feature and check if it fits in window
                        feature = HaarFeature(feature_type, (x, y), (w, h))
                        
                        if (x + feature.get_total_width() <= self.window_width and
                            y + feature.get_total_height() <= self.window_height):
                            features.append(feature)
        
        return features


class WeakClassifier:
    """
    A weak classifier based on a single Haar feature with a threshold
    """
    
    def __init__(self, feature, threshold, parity):
        """
        Initialize weak classifier
        
        Args:
            feature: HaarFeature object
            threshold: Decision threshold
            parity: +1 or -1 indicating direction of inequality
        """
        self.feature = feature
        self.threshold = threshold
        self.parity = parity
    
    def classify(self, integral_image):
        """
        Classify a single sample
        
        Args:
            integral_image: Integral image of sample
            
        Returns:
            1 (face) or 0 (non-face)
        """
        feature_value = self.feature.compute(integral_image)
        
        if self.parity * feature_value < self.parity * self.threshold:
            return 1
        else:
            return 0
    
    def classify_batch(self, integral_images):
        """
        Classify multiple samples
        
        Args:
            integral_images: List of integral images
            
        Returns:
            Array of predictions
        """
        return np.array([self.classify(ii) for ii in integral_images])
    
    @staticmethod
    def train(feature, integral_images, labels, weights):
        """
        Train a weak classifier on a feature
        Finds optimal threshold and parity
        
        Args:
            feature: HaarFeature to use
            integral_images: List of training integral images
            labels: Training labels
            weights: Sample weights
            
        Returns:
            Best WeakClassifier for this feature
        """
        n_samples = len(integral_images)
        
        # Compute feature responses for all samples
        feature_values = np.array([
            feature.compute(ii) for ii in integral_images
        ])
        
        # Sort samples by feature value
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Initialize running sums
        total_pos_weight = np.sum(weights[labels == 1])
        total_neg_weight = np.sum(weights[labels == 0])
        
        cumsum_pos = 0.0
        cumsum_neg = 0.0
        
        min_error = float('inf')
        best_threshold = 0.0
        best_parity = 1
        
        # Try each unique feature value as a threshold
        for i in range(n_samples):
            if sorted_labels[i] == 1:
                cumsum_pos += sorted_weights[i]
            else:
                cumsum_neg += sorted_weights[i]
            
            # Error if we predict all below threshold as positive
            error_below = cumsum_neg + (total_pos_weight - cumsum_pos)
            
            # Error if we predict all below threshold as negative
            error_above = cumsum_pos + (total_neg_weight - cumsum_neg)
            
            # Choose best
            if error_below < min_error:
                min_error = error_below
                best_threshold = sorted_values[i]
                best_parity = 1
            
            if error_above < min_error:
                min_error = error_above
                best_threshold = sorted_values[i]
                best_parity = -1
        
        return WeakClassifier(feature, best_threshold, best_parity), min_error