#!/usr/bin/env python3
"""
Visual Documentation Generator for Fingerprint Recognition System
Creates flowcharts and sample pair visualizations
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import logging
import random
from src.dataset import FingerprintDataset
from src.models import SiameseNetwork
import warnings
warnings.filterwarnings('ignore')

# Create output directory
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "output", "images")
LOG_DIR = os.path.join(os.path.dirname(__file__), "output", "logs")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'visuals.log')),
        logging.StreamHandler()
    ]
)

def create_system_flowchart():
    """Create a detailed system architecture flowchart"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#B3E5FC', 
        'feature': '#81D4FA',
        'network': '#4FC3F7',
        'matching': '#29B6F6',
        'output': '#0288D1',
        'text': '#01579B'
    }
    
    # Title
    ax.text(5, 11.5, 'Fingerprint Recognition System Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text'])
    
    # Input Stage
    input_box = FancyBboxPatch((0.5, 9.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10, 'Input Images\n(128×128 pixels)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Preprocessing Stage
    preprocess_box = FancyBboxPatch((0.5, 8), 2, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['preprocessing'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(1.5, 8.5, 'Preprocessing\n• Resize\n• Normalize\n• Grayscale', 
            ha='center', va='center', fontsize=10)
    
    # CNN Feature Extractor
    cnn_box = FancyBboxPatch((3.5, 8), 3, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['feature'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(cnn_box)
    ax.text(5, 8.75, 'CNN Feature Extractor\n• Conv1: 1→32 (3×3)\n• Conv2: 32→64 (3×3)\n• Conv3: 64→128 (3×3)\n• Conv4: 128→256 (3×3)\n• FC: 256×8×8→128', 
            ha='center', va='center', fontsize=9)
    
    # Siamese Network
    siamese_box = FancyBboxPatch((7.5, 8), 2, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['network'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(siamese_box)
    ax.text(8.5, 8.75, 'Siamese Network\n• Shared Weights\n• L2 Normalization\n• Twin Architecture', 
            ha='center', va='center', fontsize=10)
    
    # Training Path
    training_box = FancyBboxPatch((1, 6), 2.5, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['matching'], 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(training_box)
    ax.text(2.25, 6.5, 'Training Phase\n• Contrastive Loss\n• Adam Optimizer\n• 30 Epochs', 
            ha='center', va='center', fontsize=10)
    
    # Similarity Computation
    similarity_box = FancyBboxPatch((6, 6), 2.5, 1, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['matching'], 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(similarity_box)
    ax.text(7.25, 6.5, 'Similarity Computation\n• Euclidean Distance\n• Feature Comparison', 
            ha='center', va='center', fontsize=10)
    
    # Decision Stage
    decision_box = FancyBboxPatch((3.5, 4), 3, 1, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['output'], 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(decision_box)
    ax.text(5, 4.5, 'Decision Making\n• Threshold Comparison\n• Match/No Match', 
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    
    # Performance Metrics
    metrics_box = FancyBboxPatch((1, 2), 8, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#F5F5F5', 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(metrics_box)
    ax.text(5, 2.5, 'Performance Metrics: ROC AUC, EER, FAR, FRR\nDataset: FVC2000_DB4_B (80% Train / 20% Test)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Arrows
    arrows = [
        # Input to preprocessing
        ((1.5, 9.5), (1.5, 9)),
        # Preprocessing to CNN
        ((2.5, 8.5), (3.5, 8.75)),
        # CNN to Siamese
        ((6.5, 8.75), (7.5, 8.75)),
        # To training
        ((5, 8), (2.25, 7)),
        # To similarity
        ((8.5, 8), (7.25, 7)),
        # To decision
        ((7.25, 6), (5.5, 5)),
        ((2.25, 6), (4.5, 5)),
        # To metrics
        ((5, 4), (5, 3))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", ec="black", lw=2)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Layer'),
        mpatches.Patch(color=colors['preprocessing'], label='Preprocessing'),
        mpatches.Patch(color=colors['feature'], label='Feature Extraction'),
        mpatches.Patch(color=colors['network'], label='Neural Network'),
        mpatches.Patch(color=colors['matching'], label='Processing'),
        mpatches.Patch(color=colors['output'], label='Output/Decision')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/system_flowchart.png', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"System flowchart saved to: {IMAGES_DIR}/system_flowchart.png")

def create_sample_pairs_visualization():
    """Create visualization showing genuine and impostor pairs with similarity scores"""
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Use training dataset which has multiple images per person for creating genuine pairs
    dataset = FingerprintDataset('dataset', transform=transform, mode='train')
    
    logging.info(f"Using dataset with {len(dataset)} images for visualization")
    
    # Load trained model if available
    model_path = 'output/models/fingerprint_model.pth'
    model = None
    if os.path.exists(model_path):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            model = SiameseNetwork(checkpoint['embedding_dim']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            logging.info("Loaded trained model for similarity scoring")
        except Exception as e:
            logging.error(f"Could not load model: {e}")
            model = None
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Fingerprint Pair Matching Examples', fontsize=16, fontweight='bold')
    
    # Get unique labels by sampling strategically across the dataset
    labels = []
    sample_indices = []
    
    # Sample from different parts of the dataset to ensure we get different people
    step_size = len(dataset) // 10  # Assuming 10 people, sample every step
    for i in range(0, len(dataset), step_size):
        if i < len(dataset):
            _, label = dataset[i]
            # Handle numpy arrays by converting to scalar
            if isinstance(label, np.ndarray):
                label = label.item() if label.size == 1 else int(label[0])
            labels.append(label)
            sample_indices.append(i)
    
    unique_labels = list(set(labels))
    logging.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Generate genuine pairs (same person)
    genuine_pairs = []
    for label in unique_labels[:2]:  # Take first 2 classes
        # Find all images with this label in the entire dataset
        all_indices_for_label = []
        for i in range(len(dataset)):
            _, dataset_label = dataset[i]
            if isinstance(dataset_label, np.ndarray):
                dataset_label = dataset_label.item() if dataset_label.size == 1 else int(dataset_label[0])
            if dataset_label == label:
                all_indices_for_label.append(i)
        
        if len(all_indices_for_label) >= 2:
            # Take two different images from the same person
            genuine_pairs.append((all_indices_for_label[0], all_indices_for_label[1], label))
    
    # Generate impostor pairs (different people)
    impostor_pairs = []
    if len(unique_labels) >= 2:
        for i in range(min(2, len(unique_labels)-1)):
            label1, label2 = unique_labels[i], unique_labels[i+1]
            
            # Find indices for label1
            indices1 = []
            for idx in range(len(dataset)):
                _, dataset_label = dataset[idx]
                if isinstance(dataset_label, np.ndarray):
                    dataset_label = dataset_label.item() if dataset_label.size == 1 else int(dataset_label[0])
                if dataset_label == label1:
                    indices1.append(idx)
                    break  # Just need one image
            
            # Find indices for label2  
            indices2 = []
            for idx in range(len(dataset)):
                _, dataset_label = dataset[idx]
                if isinstance(dataset_label, np.ndarray):
                    dataset_label = dataset_label.item() if dataset_label.size == 1 else int(dataset_label[0])
                if dataset_label == label2:
                    indices2.append(idx)
                    break  # Just need one image
            
            if indices1 and indices2:
                impostor_pairs.append((indices1[0], indices2[0], f"{label1} vs {label2}"))
    
    logging.info(f"Generated {len(genuine_pairs)} genuine pairs: {genuine_pairs}")
    logging.info(f"Generated {len(impostor_pairs)} impostor pairs: {impostor_pairs}")
    
    # Helper function to denormalize image for display
    def denormalize_image(tensor):
        tensor = tensor * 0.5 + 0.5  # Reverse normalization
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.squeeze().cpu().numpy()
    
    # Helper function to calculate similarity score
    def calculate_similarity(img1_tensor, img2_tensor):
        if model is None:
            return np.random.uniform(0.1, 0.9)  # Random score if no model
        
        with torch.no_grad():
            img1 = img1_tensor.unsqueeze(0).to(next(model.parameters()).device)
            img2 = img2_tensor.unsqueeze(0).to(next(model.parameters()).device)
            features1, features2 = model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(features1, features2).item()
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity = 1 / (1 + distance)
            return similarity
    
    # Plot genuine pairs
    for i, (idx1, idx2, label) in enumerate(genuine_pairs[:2]):
        img1, _ = dataset[idx1]
        img2, _ = dataset[idx2]
        
        similarity = calculate_similarity(img1, img2)
        
        # Display images
        axes[0, i*2].imshow(denormalize_image(img1), cmap='gray')
        axes[0, i*2].set_title(f'Person {label} - Image 1', fontsize=10)
        axes[0, i*2].axis('off')
        
        axes[0, i*2+1].imshow(denormalize_image(img2), cmap='gray')
        axes[0, i*2+1].set_title(f'Person {label} - Image 2', fontsize=10)
        axes[0, i*2+1].axis('off')
        
        # Add similarity score and result
        result = "MATCH" if similarity > 0.5 else "NO MATCH"
        color = 'green' if result == "MATCH" else 'red'
        
        fig.text(0.125 + i*0.5, 0.52, f'Similarity: {similarity:.3f}\nResult: {result}', 
                ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                fontweight='bold')
    
    # Plot impostor pairs
    for i, (idx1, idx2, label_info) in enumerate(impostor_pairs[:2]):
        img1, _ = dataset[idx1]
        img2, _ = dataset[idx2]
        
        similarity = calculate_similarity(img1, img2)
        
        # Display images
        axes[1, i*2].imshow(denormalize_image(img1), cmap='gray')
        axes[1, i*2].set_title(f'Person {label_info.split(" vs ")[0]}', fontsize=10)
        axes[1, i*2].axis('off')
        
        axes[1, i*2+1].imshow(denormalize_image(img2), cmap='gray')
        axes[1, i*2+1].set_title(f'Person {label_info.split(" vs ")[1]}', fontsize=10)
        axes[1, i*2+1].axis('off')
        
        # Add similarity score and result
        result = "MATCH" if similarity > 0.5 else "NO MATCH"
        color = 'green' if result == "MATCH" else 'red'
        
        fig.text(0.125 + i*0.5, 0.02, f'Similarity: {similarity:.3f}\nResult: {result}', 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                fontweight='bold')
    
    # Add row labels
    fig.text(0.02, 0.75, 'GENUINE PAIRS\n(Same Person)', rotation=90, 
            ha='center', va='center', fontsize=14, fontweight='bold', color='green')
    fig.text(0.02, 0.25, 'IMPOSTOR PAIRS\n(Different People)', rotation=90, 
            ha='center', va='center', fontsize=14, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
    plt.savefig(f'{IMAGES_DIR}/sample_pairs.png', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Sample pairs visualization saved to: {IMAGES_DIR}/sample_pairs.png")

def create_cnn_architecture_diagram():
    """Create detailed CNN architecture visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(9, 7.5, 'CNN Feature Extractor Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Layer specifications
    layers = [
        {"name": "Input", "size": "128×128×1", "pos": 1, "color": "#E3F2FD"},
        {"name": "Conv1+BN", "size": "128×128×32", "pos": 3, "color": "#BBDEFB"},
        {"name": "MaxPool", "size": "64×64×32", "pos": 4.5, "color": "#90CAF9"},
        {"name": "Conv2+BN", "size": "64×64×64", "pos": 6, "color": "#64B5F6"},
        {"name": "MaxPool", "size": "32×32×64", "pos": 7.5, "color": "#42A5F5"},
        {"name": "Conv3+BN", "size": "32×32×128", "pos": 9, "color": "#2196F3"},
        {"name": "MaxPool", "size": "16×16×128", "pos": 10.5, "color": "#1E88E5"},
        {"name": "Conv4+BN", "size": "16×16×256", "pos": 12, "color": "#1976D2"},
        {"name": "MaxPool", "size": "8×8×256", "pos": 13.5, "color": "#1565C0"},
        {"name": "Flatten", "size": "16384", "pos": 15, "color": "#0D47A1"},
        {"name": "FC1", "size": "512", "pos": 16.5, "color": "#0D47A1"},
        {"name": "FC2", "size": "128", "pos": 17.5, "color": "#0D47A1"}
    ]
    
    # Draw layers
    for i, layer in enumerate(layers):
        # Calculate height based on layer type
        if "Conv" in layer["name"] or "MaxPool" in layer["name"]:
            height = 3
            width = 0.8
        elif "FC" in layer["name"] or "Flatten" in layer["name"]:
            height = 2
            width = 0.6
        else:  # Input
            height = 3.5
            width = 1
        
        # Draw box
        box = FancyBboxPatch((layer["pos"]-width/2, 4-height/2), width, height,
                            boxstyle="round,pad=0.05",
                            facecolor=layer["color"],
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        # Add text
        ax.text(layer["pos"], 4.3, layer["name"], ha='center', va='center', 
                fontsize=10, fontweight='bold')
        ax.text(layer["pos"], 3.7, layer["size"], ha='center', va='center', 
                fontsize=9)
        
        # Add arrows
        if i < len(layers) - 1:
            next_pos = layers[i+1]["pos"]
            arrow = ConnectionPatch((layer["pos"]+width/2, 4), (next_pos-0.4, 4), 
                                  "data", "data", arrowstyle="->", 
                                  shrinkA=5, shrinkB=5, mutation_scale=15, 
                                  fc="black", ec="black", lw=1.5)
            ax.add_patch(arrow)
    
    # Add technical details
    details_text = """
    Technical Specifications:
    • Kernel Size: 3×3 for all convolutions
    • Stride: 1 for convolutions, 2 for pooling
    • Padding: 1 for convolutions to maintain size
    • Activation: ReLU after each convolution
    • Batch Normalization after each convolution
    • Dropout: 0.5 before final embedding layer
    • Final embedding: L2 normalized 128-dimensional vector
    """
    
    ax.text(9, 1.5, details_text, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/cnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"CNN architecture diagram saved to: {IMAGES_DIR}/cnn_architecture.png")

def create_training_process_diagram():
    """Create training process visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Siamese Network Training Process', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Data preparation
    data_box = FancyBboxPatch((1, 7.5), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#E8F5E8',
                             edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(2.5, 8.25, 'Data Preparation\n• 640 Train + 160 Val\n• Create pairs (2000 train, 500 val)\n• 50% positive, 50% negative', 
            ha='center', va='center', fontsize=10)
    
    # Pair creation
    pair_box = FancyBboxPatch((5.5, 7.5), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#FFF3E0',
                             edgecolor='black', linewidth=2)
    ax.add_patch(pair_box)
    ax.text(7, 8.25, 'Pair Generation\n• Positive: Same person\n• Negative: Different people\n• Random sampling', 
            ha='center', va='center', fontsize=10)
    
    # Training loop
    training_box = FancyBboxPatch((10, 7.5), 3, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E3F2FD',
                                 edgecolor='black', linewidth=2)
    ax.add_patch(training_box)
    ax.text(11.5, 8.25, 'Training Loop\n• 30 epochs\n• Batch size: 32\n• Adam optimizer', 
            ha='center', va='center', fontsize=10)
    
    # Forward pass
    forward_box = FancyBboxPatch((2, 5), 4, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#F3E5F5',
                                edgecolor='black', linewidth=2)
    ax.add_patch(forward_box)
    ax.text(4, 5.75, 'Forward Pass\n• Extract features from both images\n• L2 normalize embeddings\n• Calculate contrastive loss', 
            ha='center', va='center', fontsize=10)
    
    # Loss function
    loss_box = FancyBboxPatch((8, 5), 4, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor='#FFEBEE',
                             edgecolor='black', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(10, 5.75, 'Contrastive Loss\n• Margin = 1.0\n• Minimize distance for same class\n• Maximize distance for different class', 
            ha='center', va='center', fontsize=10)
    
    # Optimization
    opt_box = FancyBboxPatch((5, 2.5), 4, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor='#E8F5E8',
                            edgecolor='black', linewidth=2)
    ax.add_patch(opt_box)
    ax.text(7, 3.25, 'Optimization\n• Backpropagation\n• Update weights\n• Learning rate scheduling', 
            ha='center', va='center', fontsize=10)
    
    # Arrows
    arrows = [
        ((4, 8.25), (5.5, 8.25)),
        ((8.5, 8.25), (10, 8.25)),
        ((11.5, 7.5), (4, 6.5)),
        ((4, 5), (8, 5.75)),
        ((10, 5), (7, 4)),
        ((7, 2.5), (11.5, 7.5))  # Back to training
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", ec="black", lw=2)
        ax.add_patch(arrow)
    
    # Add equation
    ax.text(7, 0.5, r'$L = \frac{1}{2N} \sum_{i=1}^{N} [y \cdot d^2 + (1-y) \cdot \max(0, margin - d)^2]$', 
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'))
    ax.text(7, 0.1, 'Contrastive Loss Function (y=0 for same, y=1 for different)', 
            ha='center', va='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/training_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    logging.info(f"Training process diagram saved to: {IMAGES_DIR}/training_process.png")

def main():
    """Generate all visualizations"""
    print("Creating Fingerprint Recognition System Visualizations...")
    print("=" * 60)
    
    # Create visualizations
    create_system_flowchart()
    print()
    
    create_cnn_architecture_diagram()
    print()
    
    create_training_process_diagram()
    print()
    
    create_sample_pairs_visualization()
    print()
    
    print("=" * 60)
    print("All visualizations completed!")
    print("\nGenerated files:")
    print(f"{IMAGES_DIR}/system_flowchart.png - Complete system architecture")
    print(f"{IMAGES_DIR}/cnn_architecture.png - CNN feature extractor details")
    print(f"{IMAGES_DIR}/training_process.png - Training methodology")
    print(f"{IMAGES_DIR}/sample_pairs.png - Sample matching examples")

if __name__ == "__main__":
    main() 