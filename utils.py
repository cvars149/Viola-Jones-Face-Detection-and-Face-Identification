"""
Utilities Module
Visualization and analysis functions for the assignment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.manifold import TSNE


def visualize_detections(images, detections_list, save_path, max_images=20):
    """
    Visualize face detections on images
    
    Args:
        images: List of test images
        detections_list: List of detection results (list of (x,y,w,h,conf) tuples)
        save_path: Path to save visualization
        max_images: Maximum number of images to visualize
    """
    n_images = min(len(images), max_images)
    
    # Create grid
    cols = 5
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for idx in range(n_images):
        ax = axes[idx]
        img = images[idx]
        detections = detections_list[idx]
        
        # Display image
        ax.imshow(img, cmap='gray')
        
        # Draw detection boxes
        for x, y, w, h, conf in detections:
            rect = Rectangle((x, y), w, h, linewidth=2, 
                           edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 5, f'{conf:.2f}', color='red', 
                   fontsize=8, weight='bold')
        
        ax.set_title(f'Image {idx+1}: {len(detections)} detection(s)')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Detections saved to {save_path}")


def plot_comparison(eigenface_model, wavelet_model, gallery_images, 
                   gallery_labels, save_path):
    """
    Create comparison plots for Eigenfaces vs Wavelets
    
    Args:
        eigenface_model: Trained Eigenface identifier
        wavelet_model: Trained Wavelet identifier
        gallery_images: Gallery images
        gallery_labels: Gallery labels
        save_path: Path to save plots
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 1. t-SNE visualization of Eigenface features
    print("   Computing t-SNE for Eigenfaces...")
    eigen_features, eigen_labels = eigenface_model.get_gallery_features()
    
    tsne_eigen = TSNE(n_components=2, random_state=42, perplexity=30)
    eigen_2d = tsne_eigen.fit_transform(eigen_features)
    
    # 2. t-SNE visualization of Wavelet features
    print("   Computing t-SNE for Wavelets...")
    wavelet_features, wavelet_labels = wavelet_model.get_gallery_features()
    
    tsne_wavelet = TSNE(n_components=2, random_state=42, perplexity=30)
    wavelet_2d = tsne_wavelet.fit_transform(wavelet_features)
    
    # Get unique subjects for coloring
    unique_subjects = np.unique(eigen_labels)
    n_subjects = len(unique_subjects)
    colors = plt.cm.tab20(np.linspace(0, 1, n_subjects))
    
    # Create color map
    subject_to_color = {subj: colors[i] for i, subj in enumerate(unique_subjects)}
    
    # Plot 1: Eigenface t-SNE
    ax1 = plt.subplot(2, 3, 1)
    for subject in unique_subjects:
        mask = eigen_labels == subject
        ax1.scatter(eigen_2d[mask, 0], eigen_2d[mask, 1], 
                   c=[subject_to_color[subject]], 
                   label=subject if len(unique_subjects) <= 10 else None,
                   alpha=0.7, s=50)
    
    ax1.set_title('Eigenfaces Feature Space (t-SNE)', fontsize=14, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    if len(unique_subjects) <= 10:
        ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Wavelet t-SNE
    ax2 = plt.subplot(2, 3, 2)
    for subject in unique_subjects:
        mask = wavelet_labels == subject
        ax2.scatter(wavelet_2d[mask, 0], wavelet_2d[mask, 1],
                   c=[subject_to_color[subject]],
                   label=subject if len(unique_subjects) <= 10 else None,
                   alpha=0.7, s=50)
    
    ax2.set_title('Gabor Wavelet Feature Space (t-SNE)', fontsize=14, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    if len(unique_subjects) <= 10:
        ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sample eigenfaces
    ax3 = plt.subplot(2, 3, 3)
    eigenfaces = eigenface_model.get_eigenfaces(n_faces=9)
    eigenface_grid = create_image_grid(eigenfaces, grid_size=(3, 3))
    ax3.imshow(eigenface_grid, cmap='gray')
    ax3.set_title('Top 9 Eigenfaces', fontsize=14, weight='bold')
    ax3.axis('off')
    
    # Plot 4: Sample Gabor filters
    ax4 = plt.subplot(2, 3, 4)
    filters = wavelet_model.visualize_filters(n_filters=16)
    filter_grid = create_image_grid(filters, grid_size=(4, 4))
    ax4.imshow(filter_grid, cmap='gray')
    ax4.set_title('Sample Gabor Filters (16 of 40)', fontsize=14, weight='bold')
    ax4.axis('off')
    
    # Plot 5: Sample gallery images
    ax5 = plt.subplot(2, 3, 5)
    sample_indices = np.random.choice(len(gallery_images), 9, replace=False)
    sample_images = [gallery_images[i] for i in sample_indices]
    image_grid = create_image_grid(sample_images, grid_size=(3, 3))
    ax5.imshow(image_grid, cmap='gray')
    ax5.set_title('Sample Gallery Images', fontsize=14, weight='bold')
    ax5.axis('off')
    
    # Plot 6: Comparison text summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    COMPARATIVE ANALYSIS
    
    Dataset Statistics:
    • Gallery Size: {len(gallery_images)} images
    • Subjects: {n_subjects}
    • Image Resolution: {gallery_images[0].shape}
    
    Eigenfaces (PCA):
    • Components: {eigenface_model.n_components}
    • Approach: Holistic (global structure)
    • Strength: Fast, effective for controlled conditions
    • Limitation: Sensitive to lighting/pose variation
    
    Gabor Wavelets:
    • Filters: {wavelet_model.n_scales} scales × {wavelet_model.n_orientations} orientations
    • Approach: Local texture analysis
    • Strength: Robust to lighting changes
    • Limitation: Higher computational cost
    
    Key Insights:
    • Eigenfaces capture overall face structure
    • Wavelets capture local texture patterns
    • t-SNE shows feature space separability
    • Tight clusters = better discrimination
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='center', family='monospace')
    
    plt.suptitle('Face Identification: Eigenfaces vs Gabor Wavelets', 
                fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Comparison plot saved to {save_path}")


def create_image_grid(images, grid_size=(3, 3), padding=2):
    """
    Create a grid visualization of images
    
    Args:
        images: List of 2D numpy arrays
        grid_size: (rows, cols) for grid
        padding: Pixels between images
        
    Returns:
        Combined grid image
    """
    rows, cols = grid_size
    n_images = min(len(images), rows * cols)
    
    # Get image size
    h, w = images[0].shape
    
    # Create output grid
    grid_h = rows * h + (rows + 1) * padding
    grid_w = cols * w + (cols + 1) * padding
    grid = np.ones((grid_h, grid_w)) * 0.5  # Gray background
    
    for idx in range(n_images):
        row = idx // cols
        col = idx % cols
        
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        
        # Normalize image
        img = images[idx]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-5)
        
        grid[y:y+h, x:x+w] = img_norm
    
    return grid


def plot_training_curve(error_history, save_path):
    """
    Plot AdaBoost training error curve
    
    Args:
        error_history: List of errors per round
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(error_history) + 1), error_history, 'b-', linewidth=2)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Weighted Error', fontsize=12)
    plt.title('AdaBoost Training Error Curve', fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_confusion_matrix(true_labels, pred_labels, save_path):
    """
    Generate and plot confusion matrix
    
    Args:
        true_labels: True labels
        pred_labels: Predicted labels
        save_path: Path to save plot
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    unique_labels = np.unique(true_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()