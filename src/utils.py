# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")



# %%
def create_siamese_pairs(dataset, num_pairs=1000):
    """Create pairs for Siamese network training"""
    pairs = []
    labels = []
    
    # Get all data
    all_images = []
    all_labels = []
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        all_images.append(image)
        all_labels.append(label)
    
    # Convert labels to numpy for easier processing
    all_labels = np.array(all_labels)
    unique_labels = np.unique(all_labels)
    
    # Calculate pairs per type
    positive_pairs = num_pairs // 2
    negative_pairs = num_pairs - positive_pairs
    
    # Create positive pairs (same class)
    for _ in range(positive_pairs):
        # Choose a random class
        class_label = np.random.choice(unique_labels)
        class_indices = np.where(all_labels == class_label)[0]
        
        if len(class_indices) >= 2:
            # Choose two different samples from the same class
            idx1, idx2 = np.random.choice(class_indices, 2, replace=False)
            pairs.append((all_images[idx1], all_images[idx2]))
            labels.append(0)  # 0 for same class
    
    # Create negative pairs (different classes)
    for _ in range(negative_pairs):
        # Choose two different classes
        if len(unique_labels) >= 2:
            class1, class2 = np.random.choice(unique_labels, 2, replace=False)
            
            indices1 = np.where(all_labels == class1)[0]
            indices2 = np.where(all_labels == class2)[0]
            
            idx1 = np.random.choice(indices1)
            idx2 = np.random.choice(indices2)
            
            pairs.append((all_images[idx1], all_images[idx2]))
            labels.append(1)  # 1 for different class
    
    print(f"Created {len(pairs)} pairs: {labels.count(0)} positive, {labels.count(1)} negative")
    return pairs, labels


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{LOGS_DIR}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(fpr, tpr, roc_auc, genuine_scores, impostor_scores):
    """Plot ROC curve and score distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Score distributions
    ax2.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine Scores', color='green')
    ax2.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor Scores', color='red')
    ax2.set_xlabel('Distance Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distributions')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_model(model, embedding_dim, margin, filepath):
    """Save the trained model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'margin': margin
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device):
    """Load a trained model"""
    from .models import SiameseNetwork
    
    checkpoint = torch.load(filepath, map_location=device)
    embedding_dim = checkpoint['embedding_dim']
    margin = checkpoint['margin']
    model = SiameseNetwork(embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model, embedding_dim, margin 