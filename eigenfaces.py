"""
Eigenfaces Implementation using PCA
Face identification using Principal Component Analysis
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA


class EigenfaceIdentifier:
    """
    Face identification using Eigenfaces (PCA-based approach)
    """
    
    def __init__(self, n_components=50):
        """
        Initialize Eigenface identifier
        
        Args:
            n_components: Number of principal components (eigenfaces) to keep
        """
        self.n_components = n_components
        self.pca = None
        self.mean_face = None
        self.gallery_features = None
        self.gallery_labels = None
    
    def train(self, gallery_images, gallery_labels):
        """
        Train the Eigenface model on gallery images
        
        Args:
            gallery_images: Array of gallery face images (N x H x W)
            gallery_labels: Array of subject labels for gallery
        """
        print(f"\n   Training Eigenfaces with {self.n_components} components...")
        
        # Flatten images into vectors
        n_samples = len(gallery_images)
        h, w = gallery_images[0].shape
        
        print(f"   Flattening {n_samples} images of size {h}x{w}...")
        X = np.array([img.flatten() for img in gallery_images])
        
        # Compute mean face
        self.mean_face = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - self.mean_face
        
        # Perform PCA
        print(f"   Computing PCA...")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_centered)
        
        # Project gallery images onto eigenfaces
        print(f"   Projecting gallery images...")
        self.gallery_features = self.pca.transform(X_centered)
        self.gallery_labels = np.array(gallery_labels)
        
        # Print variance explained
        var_explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"   Variance explained: {var_explained:.4f}")
        print(f"   Training complete!")
    
    def predict(self, probe_images):
        """
        Predict identities for probe images
        
        Args:
            probe_images: Array of probe face images
            
        Returns:
            Array of predicted labels
        """
        # Flatten and center probe images
        X_probe = np.array([img.flatten() for img in probe_images])
        X_probe_centered = X_probe - self.mean_face
        
        # Project onto eigenfaces
        probe_features = self.pca.transform(X_probe_centered)
        
        # Find nearest neighbor in gallery
        predictions = []
        for probe_feat in probe_features:
            # Calculate Euclidean distances to all gallery samples
            distances = np.linalg.norm(
                self.gallery_features - probe_feat, axis=1
            )
            
            # Find closest match
            min_idx = np.argmin(distances)
            predictions.append(self.gallery_labels[min_idx])
        
        return np.array(predictions)
    
    def predict_with_distances(self, probe_images):
        """
        Predict identities and return distances for analysis
        
        Args:
            probe_images: Array of probe images
            
        Returns:
            predictions: Predicted labels
            distances: Minimum distances for each prediction
        """
        X_probe = np.array([img.flatten() for img in probe_images])
        X_probe_centered = X_probe - self.mean_face
        probe_features = self.pca.transform(X_probe_centered)
        
        predictions = []
        min_distances = []
        
        for probe_feat in probe_features:
            distances = np.linalg.norm(
                self.gallery_features - probe_feat, axis=1
            )
            min_idx = np.argmin(distances)
            predictions.append(self.gallery_labels[min_idx])
            min_distances.append(distances[min_idx])
        
        return np.array(predictions), np.array(min_distances)
    
    def get_gallery_features(self):
        """Get gallery feature vectors for visualization"""
        return self.gallery_features, self.gallery_labels
    
    def get_eigenfaces(self, n_faces=10):
        """
        Get the top eigenfaces for visualization
        
        Args:
            n_faces: Number of eigenfaces to return
            
        Returns:
            Array of eigenface images
        """
        n_faces = min(n_faces, self.n_components)
        
        # Get first image shape to reconstruct
        h = int(np.sqrt(len(self.mean_face)))
        w = h
        
        eigenfaces = []
        for i in range(n_faces):
            eigenface = self.pca.components_[i].reshape(h, w)
            
            # Normalize for visualization
            eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
            eigenfaces.append(eigenface)
        
        return np.array(eigenfaces)
    
    def save(self, filepath):
        """Save the trained model"""
        model_data = {
            'n_components': self.n_components,
            'pca': self.pca,
            'mean_face': self.mean_face,
            'gallery_features': self.gallery_features,
            'gallery_labels': self.gallery_labels
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load(filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        identifier = EigenfaceIdentifier(model_data['n_components'])
        identifier.pca = model_data['pca']
        identifier.mean_face = model_data['mean_face']
        identifier.gallery_features = model_data['gallery_features']
        identifier.gallery_labels = model_data['gallery_labels']
        
        return identifier