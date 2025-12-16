"""
Cascade Classifier Implementation
Multi-stage classifier with early rejection for speed
"""

import numpy as np
import pickle
from adaboost import AdaBoostClassifier
from preprocessing import IntegralImage


class CascadeClassifier:
    """
    Cascade of AdaBoost classifiers for fast face detection
    """
    
    def __init__(self, features, num_stages=5, 
                 max_features_per_stage=None,
                 min_detection_rate=0.995,
                 max_false_positive_rate=0.5):
        """
        Initialize cascade classifier
        
        Args:
            features: Pool of Haar features
            num_stages: Number of cascade stages
            max_features_per_stage: List of max features per stage
            min_detection_rate: Minimum TPR per stage
            max_false_positive_rate: Maximum FPR per stage
        """
        self.features = features
        self.num_stages = num_stages
        self.stages = []
        self.stage_thresholds = []
        
        if max_features_per_stage is None:
            max_features_per_stage = [2, 5, 10, 20, 50][:num_stages]
        
        self.max_features_per_stage = max_features_per_stage
        self.min_detection_rate = min_detection_rate
        self.max_false_positive_rate = max_false_positive_rate
    
    def train(self, integral_images, labels, raw_images=None):
        """
        Train the cascade classifier
        
        Args:
            integral_images: Initial training integral images
            labels: Initial training labels
            raw_images: Raw images for hard negative mining
        """
        current_ii = integral_images
        current_labels = labels
        
        print(f"\n   Training {self.num_stages}-stage cascade:")
        
        for stage_idx in range(self.num_stages):
            print(f"\n   === Stage {stage_idx + 1}/{self.num_stages} ===")
            print(f"   Training set size: {len(current_labels)} samples")
            print(f"     Positives: {np.sum(current_labels == 1)}")
            print(f"     Negatives: {np.sum(current_labels == 0)}")
            
            # Train AdaBoost classifier for this stage
            n_features = self.max_features_per_stage[stage_idx]
            stage_classifier = AdaBoostClassifier(n_estimators=n_features)
            
            stage_classifier.train(
                self.features, 
                current_ii, 
                current_labels,
                verbose=True
            )
            
            # Adjust threshold to meet detection rate requirement
            threshold = self._find_threshold(
                stage_classifier, 
                current_ii, 
                current_labels,
                target_tpr=self.min_detection_rate
            )
            
            # Evaluate stage performance
            acc, tpr, fpr = stage_classifier.evaluate(current_ii, current_labels)
            print(f"   Stage {stage_idx + 1} performance:")
            print(f"     Accuracy: {acc:.4f}")
            print(f"     Detection Rate (TPR): {tpr:.4f}")
            print(f"     False Positive Rate: {fpr:.4f}")
            print(f"     Threshold: {threshold:.4f}")
            
            # Add stage to cascade
            self.stages.append(stage_classifier)
            self.stage_thresholds.append(threshold)
            
            # Hard negative mining for next stage
            if stage_idx < self.num_stages - 1 and raw_images is not None:
                print(f"\n   Mining hard negatives for stage {stage_idx + 2}...")
                hard_negatives = self._mine_hard_negatives(
                    raw_images, 
                    labels,
                    max_negatives=5000
                )
                
                # Update training set
                positives_mask = current_labels == 1
                positive_ii = [current_ii[i] for i in range(len(current_ii)) 
                              if positives_mask[i]]
                positive_labels = current_labels[positives_mask]
                
                # Combine positives with hard negatives
                current_ii = positive_ii + hard_negatives
                current_labels = np.concatenate([
                    positive_labels,
                    np.zeros(len(hard_negatives))
                ])
                
                print(f"   Added {len(hard_negatives)} hard negatives")
    
    def _find_threshold(self, classifier, integral_images, labels, 
                       target_tpr=0.995):
        """
        Find threshold that achieves target detection rate
        
        Args:
            classifier: AdaBoost classifier
            integral_images: Validation images
            labels: Validation labels
            target_tpr: Target true positive rate
            
        Returns:
            Threshold value
        """
        # Get scores for all positive samples
        positive_ii = [integral_images[i] for i in range(len(integral_images)) 
                       if labels[i] == 1]
        
        if len(positive_ii) == 0:
            return 0.5
        
        scores = [classifier.predict_score(ii) for ii in positive_ii]
        scores = np.array(scores)
        
        # Find threshold that keeps target_tpr of positives
        sorted_scores = np.sort(scores)
        threshold_idx = int(len(sorted_scores) * (1 - target_tpr))
        threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else 0.0
        
        return threshold
    
    def _mine_hard_negatives(self, raw_images, labels, max_negatives=5000):
        """
        Mine hard negative examples (false positives from current cascade)
        
        Args:
            raw_images: Full training images
            labels: Image labels
            max_negatives: Maximum number of negatives to mine
            
        Returns:
            List of hard negative integral images
        """
        hard_negatives = []
        
        # Only mine from negative images
        negative_images = [raw_images[i] for i in range(len(raw_images)) 
                          if labels[i] == 0]
        
        patch_size = 16
        
        for img_idx, img in enumerate(negative_images):
            if len(hard_negatives) >= max_negatives:
                break
            
            h, w = img.shape
            
            # Slide window across image
            for y in range(0, h - patch_size, 4):
                for x in range(0, w - patch_size, 4):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    ii = IntegralImage.compute(patch)
                    
                    # Check if current cascade classifies as positive (false positive)
                    if self.predict(ii) == 1:
                        hard_negatives.append(ii)
                    
                    if len(hard_negatives) >= max_negatives:
                        break
                if len(hard_negatives) >= max_negatives:
                    break
            
            if (img_idx + 1) % 10 == 0:
                print(f"     Processed {img_idx + 1} images, found {len(hard_negatives)} hard negatives")
        
        return hard_negatives
    
    def predict(self, integral_image):
        """
        Predict using cascade (early rejection)
        
        Args:
            integral_image: Integral image to classify
            
        Returns:
            1 (face) or 0 (non-face)
        """
        for stage, threshold in zip(self.stages, self.stage_thresholds):
            score = stage.predict_score(integral_image)
            if score < threshold:
                return 0  # Rejected by this stage
        
        return 1  # Passed all stages
    
    def predict_batch(self, integral_images):
        """Predict multiple samples"""
        return np.array([self.predict(ii) for ii in integral_images])
    
    def detect_multiscale(self, image, scale_factor=1.2, 
                         min_window=(16, 16), step_size=4):
        """
        Detect faces at multiple scales using sliding window
        
        Args:
            image: Input grayscale image
            scale_factor: Scale increment between pyramid levels
            min_window: Minimum detection window size
            step_size: Sliding window step
            
        Returns:
            List of detections (x, y, w, h, confidence)
        """
        detections = []
        h, w = image.shape
        
        # Create image pyramid
        scale = 1.0
        while True:
            scaled_h = int(h / scale)
            scaled_w = int(w / scale)
            
            if scaled_h < min_window[1] or scaled_w < min_window[0]:
                break
            
            # Resize image
            from PIL import Image as PILImage
            scaled_img = np.array(PILImage.fromarray(image).resize(
                (scaled_w, scaled_h), PILImage.Resampling.LANCZOS
            ))
            
            # Slide window
            window_h, window_w = min_window
            
            for y in range(0, scaled_h - window_h, step_size):
                for x in range(0, scaled_w - window_w, step_size):
                    patch = scaled_img[y:y+window_h, x:x+window_w]
                    ii = IntegralImage.compute(patch)
                    
                    if self.predict(ii) == 1:
                        # Convert back to original image coordinates
                        det_x = int(x * scale)
                        det_y = int(y * scale)
                        det_w = int(window_w * scale)
                        det_h = int(window_h * scale)
                        
                        detections.append((det_x, det_y, det_w, det_h, 1.0))
            
            scale *= scale_factor
        
        # Apply non-maximum suppression
        detections = self._non_maximum_suppression(detections)
        
        return detections
    
    def _non_maximum_suppression(self, detections, overlap_thresh=0.3):
        """
        Apply non-maximum suppression to remove overlapping detections
        """
        if len(detections) == 0:
            return []
        
        boxes = np.array([(x, y, x+w, y+h) for x, y, w, h, _ in detections])
        scores = np.array([conf for _, _, _, _, conf in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            overlap = (w * h) / areas[order[1:]]
            
            order = order[np.where(overlap <= overlap_thresh)[0] + 1]
        
        return [detections[i] for i in keep]
    
    def save(self, filepath):
        """Save cascade model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load cascade model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)