# %%
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict, Tuple, Optional

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# %%
def evaluate_performance(model: torch.nn.Module, train_dataset, test_dataset, 
                        device: torch.device, num_trials: int = 1000) -> Optional[Dict]:
    """
    Evaluate system performance using both train and test datasets
    
    Args:
        model: Trained Siamese network model
        train_dataset: Training dataset for evaluation
        test_dataset: Test dataset
        device: Computing device (CPU/GPU)
        num_trials: Number of trials for evaluation
    
    Returns:
        Dictionary containing performance metrics or None if evaluation fails
    """
    print("Evaluating Performance...")
    
    model.eval()
    
    genuine_scores = []
    impostor_scores = []
    
    train_labels = train_dataset.labels.flatten()
    unique_labels = np.unique(train_labels)
    print(f"Evaluating with {len(unique_labels)} unique classes from training data")
    
    with torch.no_grad():
        for _ in range(num_trials // 2):
            label = np.random.choice(unique_labels)
            same_class_indices = np.where(train_labels == label)[0]
            
            if len(same_class_indices) >= 2:
                idx1, idx2 = np.random.choice(same_class_indices, 2, replace=False)
                img1, _ = train_dataset[idx1]
                img2, _ = train_dataset[idx2]
                
                img1 = img1.unsqueeze(0).to(device)
                img2 = img2.unsqueeze(0).to(device)
                
                features1, features2 = model(img1, img2)
                distance = F.pairwise_distance(features1, features2).item()
                genuine_scores.append(distance)
        
        for _ in range(num_trials // 2):
            label1, label2 = np.random.choice(unique_labels, 2, replace=False)
            idx1 = np.random.choice(np.where(train_labels == label1)[0])
            idx2 = np.random.choice(np.where(train_labels == label2)[0])
            
            img1, _ = train_dataset[idx1]
            img2, _ = train_dataset[idx2]
            
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            
            features1, features2 = model(img1, img2)
            distance = F.pairwise_distance(features1, features2).item()
            impostor_scores.append(distance)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    print(f"Genuine scores: {len(genuine_scores)}, mean: {np.mean(genuine_scores):.4f}, std: {np.std(genuine_scores):.4f}")
    print(f"Impostor scores: {len(impostor_scores)}, mean: {np.mean(impostor_scores):.4f}, std: {np.std(impostor_scores):.4f}")
    
    if np.any(np.isnan(genuine_scores)) or np.any(np.isnan(impostor_scores)):
        print("Warning: NaN values detected in scores!")
        genuine_scores = genuine_scores[~np.isnan(genuine_scores)]
        impostor_scores = impostor_scores[~np.isnan(impostor_scores)]
    
    if np.any(np.isinf(genuine_scores)) or np.any(np.isinf(impostor_scores)):
        print("Warning: Infinite values detected in scores!")
        genuine_scores = genuine_scores[~np.isinf(genuine_scores)]
        impostor_scores = impostor_scores[~np.isinf(impostor_scores)]
    

    if len(genuine_scores) < 10 or len(impostor_scores) < 10:
        print("Error: Not enough scores for evaluation!")
        return None

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
    
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
    
    # Plot ROC curve and score distributions
    plot_performance_analysis(fpr, tpr, roc_auc, genuine_scores, impostor_scores)
    
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

# %%
def plot_performance_analysis(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, 
                            genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> None:
    """
    Plot ROC curve and score distributions
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under ROC curve
        genuine_scores: Genuine comparison scores
        impostor_scores: Impostor comparison scores
    """
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
    plt.savefig(os.path.join(IMAGES_DIR, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# %%
def calculate_metrics(genuine_scores: np.ndarray, impostor_scores: np.ndarray, 
                     threshold: float) -> Tuple[float, float]:
    """
    Calculate FAR and FRR at a given threshold
    
    Args:
        genuine_scores: Genuine comparison scores
        impostor_scores: Impostor comparison scores
        threshold: Decision threshold
    
    Returns:
        Tuple of (FAR, FRR)
    """
    far = np.mean(impostor_scores <= threshold)
    frr = np.mean(genuine_scores > threshold)
    return far, frr

# %%
def find_eer_threshold(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float]:
    """
    Find the Equal Error Rate (EER) threshold
    
    Args:
        genuine_scores: Genuine comparison scores
        impostor_scores: Impostor comparison scores
    
    Returns:
        Tuple of (EER, threshold)
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([np.zeros(len(genuine_scores)), np.ones(len(impostor_scores))])
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    
    return eer, eer_threshold 