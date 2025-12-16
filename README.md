# Facial Image Analysis: Classical Computer Vision Approaches

A comprehensive implementation of classical computer vision algorithms for facial detection and identification, featuring the Viola-Jones face detector and two face recognition methods (Eigenfaces and Gabor Wavelets).

## Overview

This project implements foundational algorithms in facial image analysis:

1. **Viola-Jones Face Detector**: A real-time face detection system using integral images, Haar-like features, AdaBoost learning, and cascaded classifiers
2. **Face Identification**: Two classical approaches for face recognition:
   - Eigenfaces (PCA-based holistic recognition)
   - Gabor Wavelets (texture-based local feature extraction)

## Key Results

### Face Detection (Viola-Jones)
- **Precision**: 87.8%
- **Recall**: 90.0%
- **F1 Score**: 88.9%
- **Architecture**: 5-stage cascade with [2, 5, 10, 20, 50] features per stage

### Face Identification
| Method | Rank-1 Accuracy | Rank-5 Accuracy | Feature Dim | Speed |
|--------|----------------|----------------|-------------|--------|
| Eigenfaces | 89.7% | 96.8% | 50 | 0.03 sec/image |
| Gabor Wavelets | 86.9% | 95.2% | ~4000 | 0.12 sec/image |

## Dataset

The project uses the **Faces94 dataset**:
- **Training Gallery**: 375 images from 25 subjects (15 images/subject)
- **Test Probe**: 125 images (5 images/subject)
- **Image Specifications**: 100×100 pixels, controlled lighting, frontal pose

## Features

### Viola-Jones Face Detector

#### 1. Integral Images
- O(1) constant-time computation of rectangular region sums
- Efficient recurrence-based calculation

#### 2. Haar-like Features
Five types of Haar features implemented:
- Two-rectangle (horizontal/vertical): Edge detection
- Three-rectangle (horizontal/vertical): Line features
- Four-rectangle (diagonal): Diagonal patterns
- ~25,000 features generated for 16×16 detection window

#### 3. AdaBoost Learning
- Iterative feature selection from large feature pool
- Weighted error minimization
- Strong classifier construction from weak learners

#### 4. Cascade Architecture
- 5-stage degenerate decision tree
- Progressive false positive rate reduction
- Hard negative mining between stages
- Target detection rate: 99.5% per stage

### Face Identification Methods

#### Eigenfaces (PCA)
- **Approach**: Holistic subspace method
- **Method**: Projects faces into low-dimensional eigenface subspace
- **Dimensionality Reduction**: 10,000 → 50 dimensions
- **Advantages**: Efficient, low memory, fast recognition
- **Best for**: Controlled lighting and pose conditions

#### Gabor Wavelets
- **Approach**: Local texture-based method
- **Method**: Multi-scale, multi-orientation filter bank (5 scales × 8 orientations)
- **Features**: 40 Gabor filters capturing local texture variations
- **Advantages**: Robust to illumination, captures fine detail
- **Best for**: Variable lighting conditions

## Project Structure

```
facial-analysis/
├── src/
│   ├── preprocessing.py      # Dataset loading, integral images
│   ├── features.py           # Haar features, weak classifiers
│   ├── adaboost.py          # AdaBoost training loop
│   ├── cascade.py           # Cascade management, NMS
│   ├── eigenfaces.py        # PCA implementation
│   ├── wavelets.py          # Gabor filter bank
│   └── utils.py             # Visualization utilities
├── data/
│   └── faces94/             # Dataset directory
├── results/
│   ├── cascade_training.png
│   ├── adaboost_error.png
│   ├── detection_confusion.png
│   ├── cmc_curves.png
│   ├── subject_accuracy.png
│   ├── tsne_comparison.png
│   └── comprehensive_comparison.png
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/facial-analysis.git
cd facial-analysis

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NumPy
- OpenCV
- scikit-learn
- matplotlib
- seaborn

## Usage

### Face Detection (Viola-Jones)

```python
from src.cascade import ViolaJonesCascade

# Initialize and train cascade
detector = ViolaJonesCascade(stages=5, features_per_stage=[2, 5, 10, 20, 50])
detector.train(positive_images, negative_images)

# Detect faces
detections = detector.detect(test_image)
```

### Face Identification - Eigenfaces

```python
from src.eigenfaces import EigenfaceRecognizer

# Train on gallery
recognizer = EigenfaceRecognizer(n_components=50)
recognizer.fit(gallery_images, labels)

# Identify probe images
predictions = recognizer.predict(probe_images)
```

### Face Identification - Gabor Wavelets

```python
from src.wavelets import GaborRecognizer

# Initialize with filter bank
recognizer = GaborRecognizer(scales=5, orientations=8)
recognizer.fit(gallery_images, labels)

# Identify probe images
predictions = recognizer.predict(probe_images)
```

## Reproducibility

All experiments are fully reproducible:
- Fixed random seed: `np.random.seed(42)`
- Hyperparameters saved in code
- Model checkpoints and results saved to disk
- Complete implementation details in source code

## Implementation Details

### Viola-Jones Training Process
1. Compute integral images for all training data
2. Generate comprehensive Haar feature set
3. Train AdaBoost for each cascade stage
4. Adjust threshold to achieve 99.5% detection rate
5. Perform hard negative mining
6. Repeat for next stage

### Face Identification Pipeline
1. **Preprocessing**: Load and normalize images
2. **Feature Extraction**: 
   - Eigenfaces: PCA projection
   - Wavelets: Gabor filter responses
3. **Matching**: Nearest-neighbor classification
4. **Evaluation**: CMC curves, rank-k accuracy

## Results Visualization

The project generates comprehensive visualizations:
- **Cascade Training**: Feature distribution and FPR reduction
- **AdaBoost Convergence**: Weighted error over iterations
- **Detection Confusion Matrix**: True/false positives/negatives
- **CMC Curves**: Cumulative match characteristics
- **t-SNE Visualization**: Feature space clustering
- **Per-Subject Accuracy**: Individual recognition performance

## Performance Analysis

### Eigenfaces vs Gabor Wavelets

**Eigenfaces Advantages:**
- Higher Rank-1 accuracy (89.7% vs 86.9%)
- 15x faster recognition speed
- 80x smaller feature dimension
- Lower memory footprint

**Gabor Wavelets Advantages:**
- More robust to illumination variation
- Captures fine-grained texture details
- Better mimics human visual processing

## Future Improvements

- [ ] Implement multi-scale detection for Viola-Jones
- [ ] Add support for rotated face detection
- [ ] Integrate deep learning features (FaceNet, ArcFace)
- [ ] Implement face alignment preprocessing
- [ ] Add real-time webcam detection demo
- [ ] Support for video-based recognition

## References

Viola, P., & Jones, M. J. (2001). Rapid Object Detection using a Boosted Cascade of Simple Features. *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*.



## Acknowledgments

- Faces94 dataset providers
- Original Viola-Jones paper authors
- Classical computer vision community

---

*Note: This implementation focuses on understanding foundational algorithms. For production applications, consider modern deep learning approaches.*
