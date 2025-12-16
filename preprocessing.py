"""
Preprocessing Module for Assignment 5
Handles dataset loading, patch extraction, and integral image computation
"""

import numpy as np
from pathlib import Path
from PIL import Image
import random


class IntegralImage:
    """
    Efficient computation of integral images (summed area tables)
    Allows O(1) lookup for rectangular region sums
    """
    
    @staticmethod
    def compute(image):
        """
        Compute integral image from grayscale input
        
        Args:
            image: 2D numpy array (H x W)
            
        Returns:
            Integral image where ii[y,x] = sum of all pixels above and left of (y,x)
        """
        if len(image.shape) == 3:
            # Convert to grayscale if color
            image = np.mean(image, axis=2)
        
        h, w = image.shape
        ii = np.zeros((h + 1, w + 1), dtype=np.float64)
        
        # Use cumulative sums for efficient computation
        # ii[y,x] = image[y,x] + ii[y-1,x] + ii[y,x-1] - ii[y-1,x-1]
        for y in range(h):
            for x in range(w):
                ii[y + 1, x + 1] = (image[y, x] + 
                                     ii[y, x + 1] + 
                                     ii[y + 1, x] - 
                                     ii[y, x])
        
        return ii
    
    @staticmethod
    def sum_region(ii, x, y, w, h):
        """
        Compute sum of pixels in rectangle using integral image
        
        Args:
            ii: Integral image
            x, y: Top-left corner
            w, h: Width and height
            
        Returns:
            Sum of pixels in the rectangle
        """
        # Handle boundary cases
        x, y = max(0, x), max(0, y)
        x1, y1 = x + w, y + h
        
        # Clip to image bounds
        y1 = min(y1, ii.shape[0] - 1)
        x1 = min(x1, ii.shape[1] - 1)
        
        # Four-point computation: D - B - C + A
        return ii[y1, x1] - ii[y, x1] - ii[y1, x] + ii[y, x]


class DatasetLoader:
    """
    Handles loading and preprocessing of Faces94 dataset
    """
    
    def __init__(self, dataset_path):
        """
        Initialize dataset loader
        
        Args:
            dataset_path: Path to faces94 directory
        """
        self.dataset_path = Path(dataset_path)
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        self.male_staff_path = self.dataset_path / 'malestaff'
        self.female_path = self.dataset_path / 'female'
        self.male_path = self.dataset_path / 'male'
    
    def load_training_data(self, positive_dirs=['malestaff', 'female'], 
                          patch_size=(16, 16), num_negatives=5):
        """
        Load training data for Viola-Jones detector
        
        Args:
            positive_dirs: List of directories to use for positive samples
            patch_size: Size of extracted patches (default 16x16)
            num_negatives: Number of negative patches per image
            
        Returns:
            X_train: List of image patches
            y_train: List of labels (1 for face, 0 for non-face)
        """
        X_train = []
        y_train = []
        
        for dir_name in positive_dirs:
            dir_path = self.dataset_path / dir_name
            
            if not dir_path.exists():
                print(f"Warning: Directory {dir_name} not found, skipping...")
                continue
            
            # Iterate through all subject directories
            for subject_dir in sorted(dir_path.iterdir()):
                if not subject_dir.is_dir():
                    continue
                
                # Load all images for this subject
                for img_path in sorted(subject_dir.glob('*.jpg')):
                    img = self._load_image(img_path)
                    
                    if img is None:
                        continue
                    
                    # Extract positive patch (center of image)
                    pos_patch = self._extract_center_patch(img, patch_size)
                    X_train.append(pos_patch)
                    y_train.append(1)
                    
                    # Extract negative patches (random background regions)
                    neg_patches = self._extract_negative_patches(
                        img, patch_size, num_negatives
                    )
                    X_train.extend(neg_patches)
                    y_train.extend([0] * len(neg_patches))
        
        return np.array(X_train), np.array(y_train)
    
    def load_test_data(self, test_dir='male'):
        """
        Load test images from probe directory
        
        Args:
            test_dir: Name of test directory
            
        Returns:
            images: List of full-size test images
            labels: List of subject labels
        """
        images = []
        labels = []
        
        test_path = self.dataset_path / test_dir
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test directory not found: {test_path}")
        
        for subject_dir in sorted(test_path.iterdir()):
            if not subject_dir.is_dir():
                continue
            
            subject_label = subject_dir.name
            
            for img_path in sorted(subject_dir.glob('*.jpg')):
                img = self._load_image(img_path, resize=None)
                if img is not None:
                    images.append(img)
                    labels.append(subject_label)
        
        return images, labels
    
    def load_identification_data(self, gallery_dirs=['malestaff', 'female'], 
                                train_ratio=0.75):
        """
        Load data for face identification (75-25 split per subject)
        
        Args:
            gallery_dirs: Directories to use for gallery/probe
            train_ratio: Ratio of images to use for gallery (training)
            
        Returns:
            gallery_imgs: Gallery images
            gallery_labels: Gallery subject labels
            probe_imgs: Probe images  
            probe_labels: Probe subject labels
        """
        gallery_imgs = []
        gallery_labels = []
        probe_imgs = []
        probe_labels = []
        
        for dir_name in gallery_dirs:
            dir_path = self.dataset_path / dir_name
            
            if not dir_path.exists():
                continue
            
            for subject_dir in sorted(dir_path.iterdir()):
                if not subject_dir.is_dir():
                    continue
                
                subject_label = subject_dir.name
                
                # Load all images for subject
                subject_images = []
                for img_path in sorted(subject_dir.glob('*.jpg')):
                    img = self._load_image(img_path, resize=(100, 100))
                    if img is not None:
                        subject_images.append(img)
                
                # Split into gallery and probe
                n_train = int(len(subject_images) * train_ratio)
                
                gallery_imgs.extend(subject_images[:n_train])
                gallery_labels.extend([subject_label] * n_train)
                
                probe_imgs.extend(subject_images[n_train:])
                probe_labels.extend([subject_label] * (len(subject_images) - n_train))
        
        return (np.array(gallery_imgs), np.array(gallery_labels), 
                np.array(probe_imgs), np.array(probe_labels))
    
    def _load_image(self, img_path, resize=(100, 100)):
        """Load and preprocess a single image"""
        try:
            img = Image.open(img_path)
            
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            
            # Resize if specified
            if resize is not None:
                img = img.resize(resize, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img, dtype=np.uint8)
            
            return img_array
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None
    
    def _extract_center_patch(self, image, patch_size):
        """Extract patch from center of image (assumed to contain face)"""
        h, w = image.shape
        ph, pw = patch_size
        
        # Calculate center coordinates
        center_y = h // 2
        center_x = w // 2
        
        # Extract patch
        y1 = max(0, center_y - ph // 2)
        x1 = max(0, center_x - pw // 2)
        y2 = min(h, y1 + ph)
        x2 = min(w, x1 + pw)
        
        patch = image[y1:y2, x1:x2]
        
        # Ensure correct size
        if patch.shape != patch_size:
            patch = np.array(Image.fromarray(patch).resize(
                (patch_size[1], patch_size[0]), Image.Resampling.LANCZOS
            ))
        
        return patch
    
    def _extract_negative_patches(self, image, patch_size, num_patches):
        """Extract random patches from non-face regions (background)"""
        h, w = image.shape
        ph, pw = patch_size
        
        # Define center region to avoid (where face likely is)
        center_y = h // 2
        center_x = w // 2
        avoid_radius = max(ph, pw)
        
        patches = []
        attempts = 0
        max_attempts = num_patches * 10
        
        while len(patches) < num_patches and attempts < max_attempts:
            # Random position
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
            
            # Check if this overlaps with center face region
            patch_center_y = y + ph // 2
            patch_center_x = x + pw // 2
            
            dist = np.sqrt((patch_center_y - center_y)**2 + 
                          (patch_center_x - center_x)**2)
            
            if dist > avoid_radius:
                patch = image[y:y+ph, x:x+pw]
                patches.append(patch)
            
            attempts += 1
        
        return patches