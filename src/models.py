"""
Neural network models for fingerprint recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # 8 * 8 * 256 = 16384
        self.fc1 = nn.Linear(8 * 8 * 256, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        self.classifier = nn.Linear(embedding_dim, 100)
    
    def forward(self, x, return_features=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
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