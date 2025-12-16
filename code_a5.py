

import argparse
import sys
from pathlib import Path

# Import custom modules
from preprocessing import DatasetLoader, IntegralImage
from features import HaarFeatureGenerator
from adaboost import AdaBoostClassifier
from cascade import CascadeClassifier
from eigenfaces import EigenfaceIdentifier
from wavelets import WaveletIdentifier
from utils import visualize_detections, plot_comparison


def train_detector(data_path, output_path):
    """Train the Viola-Jones face detector from scratch"""
    print("=" * 60)
    print("PART 1: Training Viola-Jones Face Detector")
    print("=" * 60)
    
    # Load training data
    loader = DatasetLoader(data_path)
    print("\n[1/6] Loading training dataset...")
    X_train, y_train = loader.load_training_data(
        positive_dirs=['maleStaff', 'female'],
        patch_size=(16, 16),
        num_negatives=5
    )
    print(f"   Loaded {len(X_train)} samples ({sum(y_train)} positive, {len(y_train)-sum(y_train)} negative)")
    
    # Generate Haar features
    print("\n[2/6] Generating Haar feature pool...")
    feature_gen = HaarFeatureGenerator(window_size=(16, 16))
    features = feature_gen.generate_all_features()
    print(f"   Generated {len(features)} Haar features")
    
    # Compute integral images for all training samples
    print("\n[3/6] Computing integral images...")
    integral_images = [IntegralImage.compute(img) for img in X_train]
    print(f"   Computed {len(integral_images)} integral images")
    
    # Initialize cascade
    print("\n[4/6] Training cascade classifier...")
    cascade = CascadeClassifier(
        features=features,
        num_stages=5,
        max_features_per_stage=[2, 5, 10, 20, 50],
        min_detection_rate=0.995,
        max_false_positive_rate=0.5
    )
    
    # Train cascade
    cascade.train(integral_images, y_train, X_train)
    
    # Save model
    print("\n[5/6] Saving trained model...")
    cascade.save(output_path / "cascade_model.pkl")
    print(f"   Model saved to {output_path / 'cascade_model.pkl'}")
    
    # Test on validation set
    print("\n[6/6] Testing on training set...")
    predictions = cascade.predict_batch(integral_images)
    accuracy = sum([p == y for p, y in zip(predictions, y_train)]) / len(y_train)
    print(f"   Training Accuracy: {accuracy:.4f}")
    
    print("\n✓ Detector training complete!\n")


def test_detector(data_path, model_path, output_path):
    """Test the detector on unseen probe images"""
    print("=" * 60)
    print("Testing Viola-Jones Face Detector")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading trained cascade model...")
    cascade = CascadeClassifier.load(model_path / "cascade_model.pkl")
    print("   Model loaded successfully")
    
    # Load test data
    print("\n[2/4] Loading test dataset (male directory)...")
    loader = DatasetLoader(data_path)
    test_images, test_labels = loader.load_test_data(test_dir='male')
    print(f"   Loaded {len(test_images)} test images")
    
    # Detect faces
    print("\n[3/4] Running face detection with sliding window...")
    all_detections = []
    for i, img in enumerate(test_images):
        detections = cascade.detect_multiscale(
            img, 
            scale_factor=1.2, 
            min_window=(16, 16),
            step_size=4
        )
        all_detections.append(detections)
        if (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(test_images)} images")
    
    # Visualize results
    print("\n[4/4] Generating visualizations...")
    visualize_detections(test_images[:20], all_detections[:20], output_path / "detections.png")
    print(f"   Visualizations saved to {output_path / 'detections.png'}")
    
    print("\n✓ Detection testing complete!\n")


def train_identification(data_path, output_path):
    """Train face identification systems (Eigenfaces and Wavelets)"""
    print("=" * 60)
    print("PART 2 (BONUS): Training Face Identification Systems")
    print("=" * 60)
    
    # Load gallery and probe sets
    loader = DatasetLoader(data_path)
    print("\n[1/5] Loading gallery and probe sets (75-25 split)...")
    gallery_imgs, gallery_labels, probe_imgs, probe_labels = loader.load_identification_data(
        gallery_dirs=['maleStaff', 'female'],
        train_ratio=0.75
    )
    print(f"   Gallery: {len(gallery_imgs)} images, {len(set(gallery_labels))} subjects")
    print(f"   Probe: {len(probe_imgs)} images")
    
    # Train Eigenfaces
    print("\n[2/5] Training Eigenfaces (PCA)...")
    eigenface = EigenfaceIdentifier(n_components=50)
    eigenface.train(gallery_imgs, gallery_labels)
    eigenface.save(output_path / "eigenface_model.pkl")
    print("   Eigenface model trained and saved")
    
    # Train Wavelets
    print("\n[3/5] Training Gabor Wavelet identifier...")
    wavelet = WaveletIdentifier(n_scales=5, n_orientations=8)
    wavelet.train(gallery_imgs, gallery_labels)
    wavelet.save(output_path / "wavelet_model.pkl")
    print("   Wavelet model trained and saved")
    
    # Evaluate both
    print("\n[4/5] Evaluating identification performance...")
    
    eigen_preds = eigenface.predict(probe_imgs)
    eigen_acc = sum([p == l for p, l in zip(eigen_preds, probe_labels)]) / len(probe_labels)
    print(f"   Eigenfaces Rank-1 Accuracy: {eigen_acc:.4f}")
    
    wavelet_preds = wavelet.predict(probe_imgs)
    wavelet_acc = sum([p == l for p, l in zip(wavelet_preds, probe_labels)]) / len(probe_labels)
    print(f"   Wavelets Rank-1 Accuracy: {wavelet_acc:.4f}")
    
    # Generate comparison plots
    print("\n[5/5] Generating t-SNE visualizations and comparison plots...")
    plot_comparison(
        eigenface, wavelet, 
        gallery_imgs, gallery_labels,
        output_path / "comparison.png"
    )
    print(f"   Comparison plots saved to {output_path / 'comparison.png'}")
    
    print("\n✓ Identification training complete!\n")


def main():
    parser = argparse.ArgumentParser(
        description="ELL715 Assignment 5: Facial Image Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the Viola-Jones detector
  python main.py --mode train_detector --data ./faces94 --output ./results
  
  # Test the detector
  python main.py --mode test_detector --data ./faces94 --model ./results --output ./results
  
  # Train identification systems
  python main.py --mode train_identification --data ./faces94 --output ./results
  
  # Run complete pipeline
  python main.py --mode full --data ./faces94 --output ./results
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_detector', 'test_detector', 'train_identification', 'full'],
                       help='Operating mode')
    parser.add_argument('--data', type=str, default='./faces94',
                       help='Path to faces94 dataset directory')
    parser.add_argument('--model', type=str, default='./results',
                       help='Path to saved models (for testing)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results and models')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    data_path = Path(args.data)
    model_path = Path(args.model)
    
    # Verify dataset exists
    if not data_path.exists():
        print(f"ERROR: Dataset path '{data_path}' does not exist!")
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == 'train_detector':
        train_detector(data_path, output_path)
    
    elif args.mode == 'test_detector':
        if not (model_path / "cascade_model.pkl").exists():
            print(f"ERROR: Model file not found at {model_path / 'cascade_model.pkl'}")
            print("Please train the detector first using --mode train_detector")
            sys.exit(1)
        test_detector(data_path, model_path, output_path)
    
    elif args.mode == 'train_identification':
        train_identification(data_path, output_path)
    
    elif args.mode == 'full':
        print("\n" + "="*60)
        print("Running Complete Pipeline")
        print("="*60 + "\n")
        train_detector(data_path, output_path)
        test_detector(data_path, output_path, output_path)
        train_identification(data_path, output_path)
        print("\n" + "="*60)
        print("✓ Complete pipeline finished successfully!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()