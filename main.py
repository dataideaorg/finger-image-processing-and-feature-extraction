# %%
import os
import time
import warnings
import logging
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
# %%
from src.dataset import FingerprintDataset
from src.recognition_system import FingerprintRecognitionSystem

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'main.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


def main():
    """Main function to run the complete pipeline"""
    logger.info("=== Final Fingerprint Recognition Baseline Model ===")
    logger.info("Dataset: FVC2000_DB4_B")
    logger.info("Framework: PyTorch")
    logger.info("=" * 50)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    logger.info("Loading datasets...")
    train_dataset = FingerprintDataset('dataset', transform=transform, mode='train')
    test_dataset = FingerprintDataset('dataset', transform=transform, mode='real')
    
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=42, 
        stratify=train_dataset.labels.flatten()
    )
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    logger.info(f"Train samples: {len(train_subset)}")
    logger.info(f"Validation samples: {len(val_subset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    system = FingerprintRecognitionSystem(embedding_dim=128, margin=1.0, device=device)
    
    logger.info("\nStarting training...")
    start_time = time.time()
    train_losses, val_losses = system.train(
        train_subset, val_subset, epochs=30, batch_size=32, lr=0.001
    )
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    logger.info("\nEvaluating performance...")
    results = system.evaluate(train_dataset, test_dataset, num_trials=1000)
    
    system.save(os.path.join(MODELS_DIR, 'fingerprint_model.pth'))
    
    logger.info("\n=== Summary ===")
    logger.info(f"Training time: {training_time:.2f} seconds")
    if results:
        logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
        logger.info(f"EER: {results['eer']:.4f}")
    else:
        logger.info("Evaluation failed - check the output above for issues")
    logger.info(f"Model saved as: {os.path.join(MODELS_DIR, 'fingerprint_model.pth')}")
    logger.info(f"Plots saved as: {os.path.join(IMAGES_DIR, 'training_curves.png')}, {os.path.join(IMAGES_DIR, 'performance_analysis.png')}")


if __name__ == "__main__":
    main() 