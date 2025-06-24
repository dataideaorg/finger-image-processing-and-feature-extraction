# %%
import torch
from .models import SiameseNetwork, ContrastiveLoss
from .training import train_siamese_network
from .evaluation import evaluate_performance
from .utils import save_model, load_model

# %%
class FingerprintRecognitionSystem:
    """Main system for fingerprint recognition training and evaluation"""
    
    def __init__(self, embedding_dim=128, margin=1.0, device=None):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseNetwork(embedding_dim).to(self.device)
        self.criterion = ContrastiveLoss(margin)
    
    def train(self, train_dataset, val_dataset, epochs=50, batch_size=32, lr=0.001):
        """Train the Siamese network"""
        return train_siamese_network(
            self.model, self.criterion, train_dataset, val_dataset, 
            self.device, epochs, batch_size, lr
        )
    
    def evaluate(self, train_dataset, test_dataset, num_trials=1000):
        """Evaluate the performance of the trained model"""
        return evaluate_performance(
            self.model, train_dataset, test_dataset, self.device, num_trials
        )
    
    def save(self, filepath):
        """Save the trained model"""
        save_model(self.model, self.embedding_dim, self.margin, filepath)
    
    def load(self, filepath):
        """Load a trained model"""
        self.model, self.embedding_dim, self.margin = load_model(filepath, self.device)
    
    def predict_similarity(self, image1, image2):
        """Calculate similarity between two images"""
        self.model.eval()
        with torch.no_grad():
            if len(image1.shape) == 3:
                image1 = image1.unsqueeze(0)
            if len(image2.shape) == 3:
                image2 = image2.unsqueeze(0)
            
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            
            features1 = torch.nn.functional.normalize(self.model.get_embedding(image1), p=2, dim=1)
            features2 = torch.nn.functional.normalize(self.model.get_embedding(image2), p=2, dim=1)
            
            distance = torch.nn.functional.pairwise_distance(features1, features2).item()
            
            return distance 