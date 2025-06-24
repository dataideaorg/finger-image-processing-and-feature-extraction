# %%
import os
import time
import warnings
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
# %%
from src.dataset import FingerprintDataset
from src.recognition_system import FingerprintRecognitionSystem

warnings.filterwarnings('ignore')

ROOT_FOLDER = os.path.dirname(__file__)
OUTPUT_FOLDER = os.path.join(ROOT_FOLDER, 'output')
LOGS_FOLDER = os.path.join(OUTPUT_FOLDER, 'logs')
MODELS_FOLDER = os.path.join(OUTPUT_FOLDER, 'models')

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    """Main function to run the complete pipeline"""
    print("=== Final Fingerprint Recognition Baseline Model ===")
    print("Dataset: FVC2000_DB4_B")
    print("Framework: PyTorch")
    print("=" * 50)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("Loading datasets...")
    train_dataset = FingerprintDataset('dataset', transform=transform, mode='train')
    test_dataset = FingerprintDataset('dataset', transform=transform, mode='real')
    
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=42, 
        stratify=train_dataset.labels.flatten()
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    system = FingerprintRecognitionSystem(embedding_dim=128, margin=1.0, device=device)
    
    print("\nStarting training...")
    start_time = time.time()
    train_losses, val_losses = system.train(
        train_subset, val_subset, epochs=30, batch_size=32, lr=0.001
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    print("\nEvaluating performance...")
    results = system.evaluate(train_dataset, test_dataset, num_trials=1000)
    
    system.save('fingerprint_final_model.pth')
    
    print("\n=== Summary ===")
    print(f"Training time: {training_time:.2f} seconds")
    if results:
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"EER: {results['eer']:.4f}")
        print(f"Model performance: {'Good' if results['roc_auc'] > 0.8 else 'Needs improvement'}")
    else:
        print("Evaluation failed - check the output above for issues")
    print(f"Model saved as: fingerprint_final_model.pth")
    print("Plots saved as: training_curves.png, performance_analysis.png")


if __name__ == "__main__":
    main() 