#!/usr/bin/env python3
"""
Improved Baseline Fingerprint Recognition Model using PyTorch
FVC2000_DB4_B Dataset Implementation

This improved version addresses NaN issues and provides better debugging.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FingerprintDataset(Dataset):
    """Custom Dataset for fingerprint images"""
    
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        # Load numpy data if available
        if os.path.exists(os.path.join(data_path, 'np_data')):
            self.load_numpy_data()
        else:
            self.load_image_data()
    
    def load_numpy_data(self):
        """Load preprocessed numpy data"""
        np_path = os.path.join(self.data_path, 'np_data')
        
        if self.mode == 'train':
            self.images = np.load(os.path.join(np_path, 'img_train.npy'))
            self.labels = np.load(os.path.join(np_path, 'label_train.npy'))
        else:
            self.images = np.load(os.path.join(np_path, 'img_real.npy'))
            self.labels = np.load(os.path.join(np_path, 'label_real.npy'))
        
        print(f"Loaded {len(self.images)} {self.mode} images")
        print(f"Image shape: {self.images.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels.flatten())}")
    
    def load_image_data(self):
        """Load raw image data"""
        if self.mode == 'train':
            img_dir = os.path.join(self.data_path, 'train_data')
        else:
            img_dir = os.path.join(self.data_path, 'real_data')
        
        self.image_paths = []
        self.labels = []
        
        for filename in sorted(os.listdir(img_dir)):
            if filename.endswith('.bmp'):
                filepath = os.path.join(img_dir, filename)
                self.image_paths.append(filepath)
                # Extract label from filename (assuming format: 00000_00.bmp)
                label = int(filename.split('_')[0])
                self.labels.append(label)
        
        self.images = None  # Will load on demand
        print(f"Found {len(self.image_paths)} {self.mode} images")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.images is not None:
            # Use preloaded numpy data
            image = self.images[idx]
            # Squeeze singleton dimensions
            image = np.squeeze(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image.astype(np.uint8))
        else:
            # Load image from file
            image = Image.open(self.image_paths[idx]).convert('L')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FingerprintCNN(nn.Module):
    """CNN architecture for fingerprint feature extraction"""
    
    def __init__(self, embedding_dim=128):
        super(FingerprintCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        # 8 * 8 * 256 = 16384
        self.fc1 = nn.Linear(8 * 8 * 256, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, 100)  # Assuming 100 classes
    
    def forward(self, x, return_features=False):
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        features = self.fc2(x)
        
        if return_features:
            return features
        
        # Classification
        x = self.classifier(features)
        return x

class SiameseNetwork(nn.Module):
    """Siamese network for similarity learning"""
    
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FingerprintCNN(embedding_dim)
        self.embedding_dim = embedding_dim
    
    def forward(self, x1, x2):
        # Extract features for both inputs
        features1 = self.feature_extractor(x1, return_features=True)
        features2 = self.feature_extractor(x2, return_features=True)
        
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        return features1, features2
    
    def get_embedding(self, x):
        """Get embedding for a single image"""
        return self.feature_extractor(x, return_features=True)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, features1, features2, labels):
        # Calculate Euclidean distance
        dist = F.pairwise_distance(features1, features2)
        
        # Contrastive loss
        loss_same = (1 - labels) * torch.pow(dist, 2)
        loss_diff = labels * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        
        loss = torch.mean(loss_same + loss_diff)
        return loss

class FingerprintRecognitionSystem:
    """Complete fingerprint recognition system"""
    
    def __init__(self, embedding_dim=128, margin=1.0):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.model = None
    
    def create_siamese_pairs(self, dataset, num_pairs=1000):
        """Create positive and negative pairs for training"""
        # Support for Subset objects
        if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
            indices = np.array(dataset.indices)
            labels = np.array(dataset.dataset.labels)[indices]
        else:
            labels = np.array(dataset.labels)
            indices = np.arange(len(labels))

        labels = labels.flatten()  # Ensure labels is 1D

        pairs = []
        pair_labels = []
        unique_labels = np.unique(labels)
        
        print(f"Creating pairs from {len(unique_labels)} unique classes")

        # Create positive pairs (same class)
        for _ in range(num_pairs // 2):
            label = np.random.choice(unique_labels)
            same_class_mask = (labels == label)
            same_class_indices = indices[same_class_mask]
            if len(same_class_indices) >= 2:
                idx1, idx2 = np.random.choice(same_class_indices, 2, replace=False)
                # Map back to subset indices
                pairs.append((np.where(indices == idx1)[0][0], np.where(indices == idx2)[0][0]))
                pair_labels.append(0)  # 0 for same class

        # Create negative pairs (different class)
        for _ in range(num_pairs // 2):
            label1, label2 = np.random.choice(unique_labels, 2, replace=False)
            idx1 = np.random.choice(indices[labels == label1])
            idx2 = np.random.choice(indices[labels == label2])
            pairs.append((np.where(indices == idx1)[0][0], np.where(indices == idx2)[0][0]))
            pair_labels.append(1)  # 1 for different class

        return pairs, pair_labels
    
    def train_siamese_network(self, train_dataset, val_dataset, epochs=50, batch_size=32, lr=0.001):
        """Train the Siamese network"""
        print("Training Siamese Network...")
        
        # Initialize model
        self.model = SiameseNetwork(self.embedding_dim).to(device)
        criterion = ContrastiveLoss(self.margin).to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Create data loaders
        train_pairs, train_labels = self.create_siamese_pairs(train_dataset, num_pairs=2000)
        val_pairs, val_labels = self.create_siamese_pairs(val_dataset, num_pairs=500)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            # Shuffle pairs
            indices = np.random.permutation(len(train_pairs))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_pairs = [train_pairs[j] for j in batch_indices]
                batch_labels = torch.tensor([train_labels[j] for j in batch_indices], dtype=torch.float32).to(device)
                
                # Get images
                img1_batch = []
                img2_batch = []
                
                for idx1, idx2 in batch_pairs:
                    img1, _ = train_dataset[idx1]
                    img2, _ = train_dataset[idx2]
                    img1_batch.append(img1)
                    img2_batch.append(img2)
                
                img1_batch = torch.stack(img1_batch).to(device)
                img2_batch = torch.stack(img2_batch).to(device)
                
                # Forward pass
                features1, features2 = self.model(img1_batch, img2_batch)
                loss = criterion(features1, features2, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for i in range(0, len(val_pairs), batch_size):
                    batch_pairs = val_pairs[i:i+batch_size]
                    batch_labels = torch.tensor([val_labels[j] for j in range(i, min(i+batch_size, len(val_pairs)))], 
                                              dtype=torch.float32).to(device)
                    
                    img1_batch = []
                    img2_batch = []
                    
                    for idx1, idx2 in batch_pairs:
                        img1, _ = val_dataset[idx1]
                        img2, _ = val_dataset[idx2]
                        img1_batch.append(img1)
                        img2_batch.append(img2)
                    
                    img1_batch = torch.stack(img1_batch).to(device)
                    img2_batch = torch.stack(img2_batch).to(device)
                    
                    features1, features2 = self.model(img1_batch, img2_batch)
                    loss = criterion(features1, features2, batch_labels)
                    val_loss += loss.item()
            
            train_loss /= (len(train_pairs) // batch_size)
            val_loss /= (len(val_pairs) // batch_size)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses)
        
        return train_losses, val_losses
    
    def evaluate_performance(self, test_dataset, num_trials=1000):
        """Evaluate system performance with improved debugging"""
        print("Evaluating Performance...")
        
        self.model.eval()
        
        # Generate genuine and impostor scores
        genuine_scores = []
        impostor_scores = []
        
        unique_labels = np.unique(test_dataset.labels.flatten())
        print(f"Evaluating with {len(unique_labels)} unique classes")
        
        with torch.no_grad():
            # Genuine scores (same person)
            for _ in range(num_trials // 2):
                label = np.random.choice(unique_labels)
                same_class_indices = np.where(test_dataset.labels.flatten() == label)[0]
                
                if len(same_class_indices) >= 2:
                    idx1, idx2 = np.random.choice(same_class_indices, 2, replace=False)
                    img1, _ = test_dataset[idx1]
                    img2, _ = test_dataset[idx2]
                    
                    img1 = img1.unsqueeze(0).to(device)
                    img2 = img2.unsqueeze(0).to(device)
                    
                    features1, features2 = self.model(img1, img2)
                    distance = F.pairwise_distance(features1, features2).item()
                    genuine_scores.append(distance)
            
            # Impostor scores (different person)
            for _ in range(num_trials // 2):
                label1, label2 = np.random.choice(unique_labels, 2, replace=False)
                idx1 = np.random.choice(np.where(test_dataset.labels.flatten() == label1)[0])
                idx2 = np.random.choice(np.where(test_dataset.labels.flatten() == label2)[0])
                
                img1, _ = test_dataset[idx1]
                img2, _ = test_dataset[idx2]
                
                img1 = img1.unsqueeze(0).to(device)
                img2 = img2.unsqueeze(0).to(device)
                
                features1, features2 = self.model(img1, img2)
                distance = F.pairwise_distance(features1, features2).item()
                impostor_scores.append(distance)
        
        # Calculate metrics
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        print(f"Genuine scores: {len(genuine_scores)}, mean: {np.mean(genuine_scores):.4f}, std: {np.std(genuine_scores):.4f}")
        print(f"Impostor scores: {len(impostor_scores)}, mean: {np.mean(impostor_scores):.4f}, std: {np.std(impostor_scores):.4f}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(genuine_scores)) or np.any(np.isnan(impostor_scores)):
            print("Warning: NaN values detected in scores!")
            genuine_scores = genuine_scores[~np.isnan(genuine_scores)]
            impostor_scores = impostor_scores[~np.isnan(impostor_scores)]
        
        if np.any(np.isinf(genuine_scores)) or np.any(np.isinf(impostor_scores)):
            print("Warning: Infinite values detected in scores!")
            genuine_scores = genuine_scores[~np.isinf(genuine_scores)]
            impostor_scores = impostor_scores[~np.isinf(impostor_scores)]
        
        # Check if we have enough scores
        if len(genuine_scores) < 10 or len(impostor_scores) < 10:
            print("Error: Not enough scores for evaluation!")
            return None
        
        # ROC curve
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        all_labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
        
        # Check if scores are all identical
        if np.std(all_scores) == 0:
            print("Error: All scores are identical! Cannot compute ROC.")
            return None
        
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # EER calculation
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer_threshold = thresholds[eer_idx]
        eer = fpr[eer_idx]
        
        # FAR and FRR at different thresholds
        threshold_0_1 = np.percentile(impostor_scores, 10)  # 10% FAR
        threshold_0_01 = np.percentile(impostor_scores, 1)  # 1% FAR
        
        far_0_1 = np.mean(impostor_scores <= threshold_0_1)
        frr_0_1 = np.mean(genuine_scores > threshold_0_1)
        
        far_0_01 = np.mean(impostor_scores <= threshold_0_01)
        frr_0_01 = np.mean(genuine_scores > threshold_0_01)
        
        # Print results
        print(f"\n=== Performance Results ===")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Equal Error Rate (EER): {eer:.4f}")
        print(f"EER Threshold: {eer_threshold:.4f}")
        print(f"\nAt 10% FAR:")
        print(f"  FAR: {far_0_1:.4f}")
        print(f"  FRR: {frr_0_1:.4f}")
        print(f"\nAt 1% FAR:")
        print(f"  FAR: {far_0_01:.4f}")
        print(f"  FRR: {frr_0_01:.4f}")
        
        # Plot ROC curve
        self.plot_roc_curve(fpr, tpr, roc_auc, genuine_scores, impostor_scores)
        
        return {
            'roc_auc': roc_auc,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'far_0_1': far_0_1,
            'frr_0_1': frr_0_1,
            'far_0_01': far_0_01,
            'frr_0_01': frr_0_01,
            'genuine_scores': genuine_scores,
            'impostor_scores': impostor_scores
        }
    
    def plot_training_curves(self, train_losses, val_losses):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, genuine_scores, impostor_scores):
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
        plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'margin': self.margin
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.embedding_dim = checkpoint['embedding_dim']
        self.margin = checkpoint['margin']
        self.model = SiameseNetwork(self.embedding_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filepath}")

def main():
    """Main function to run the complete pipeline"""
    print("=== Improved Fingerprint Recognition Baseline Model ===")
    print("Dataset: FVC2000_DB4_B")
    print("Framework: PyTorch")
    print("=" * 50)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = FingerprintDataset('dataset', transform=transform, mode='train')
    test_dataset = FingerprintDataset('dataset', transform=transform, mode='real')
    
    # Split train dataset into train and validation
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=42, 
        stratify=train_dataset.labels.flatten()
    )
    
    # Create train and validation datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize system
    system = FingerprintRecognitionSystem(embedding_dim=128, margin=1.0)
    
    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    train_losses, val_losses = system.train_siamese_network(
        train_subset, val_subset, epochs=30, batch_size=32, lr=0.001
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate performance
    print("\nEvaluating performance...")
    results = system.evaluate_performance(test_dataset, num_trials=1000)
    
    # Save model
    system.save_model('fingerprint_improved_model.pth')
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Training time: {training_time:.2f} seconds")
    if results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"EER: {results['eer']:.4f}")
    else:
        print("Evaluation failed - check the output above for issues")
    print(f"Model saved as: fingerprint_improved_model.pth")
    print("Plots saved as: training_curves.png, performance_analysis.png")

if __name__ == "__main__":
    main() 