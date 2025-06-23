# Fingerprint Recognition Baseline Model

A PyTorch-based fingerprint recognition system using Siamese networks for biometric authentication.

## Overview

This project implements a baseline fingerprint recognition model using the FVC2000_DB4_B dataset. The system uses deep learning techniques to extract features from fingerprint images and perform similarity matching.

## Dataset

- **Training Data**: 800 fingerprint images (80 images per class, 10 classes)
- **Test Data**: 10 fingerprint images (1 image per class, 10 classes)
- **Image Format**: 160x160 grayscale images in numpy format

## Methodology

### 1. Data Loading and Preprocessing
- Load fingerprint images from numpy arrays
- Resize images to 128x128 pixels
- Apply normalization (mean=0.5, std=0.5)
- Handle singleton dimensions in image arrays

### 2. Model Architecture
- **Siamese Network**: Twin CNN architectures sharing weights
- **Feature Extractor**: 4-layer CNN with batch normalization
  - Conv layers: 32 → 64 → 128 → 256 channels
  - Max pooling after each layer
  - Final embedding dimension: 128
- **Loss Function**: Contrastive loss with margin=1.0

### 3. Training Process
- **Pair Generation**: Create positive (same person) and negative (different person) pairs
- **Training Split**: 80% training, 20% validation
- **Optimization**: Adam optimizer with learning rate=0.001
- **Epochs**: 30 with learning rate scheduling
- **Batch Size**: 32

### 4. Evaluation Strategy
- Generate genuine scores (same person comparisons)
- Generate impostor scores (different person comparisons)
- Calculate performance metrics:
  - ROC AUC (Receiver Operating Characteristic Area Under Curve)
  - EER (Equal Error Rate)
  - FAR/FRR at different thresholds

### 5. Performance Metrics
- **ROC AUC**: Measures overall classification performance
- **EER**: Point where False Acceptance Rate equals False Rejection Rate
- **FAR**: False Acceptance Rate (impostor accepted as genuine)
- **FRR**: False Rejection Rate (genuine rejected as impostor)

## Usage

### Prerequisites
```bash
# Install dependencies
poetry install
poetry add opencv-python

# Activate environment
source .venv/bin/activate
```

### Running the Model
```bash
# Run the baseline model
python baseline_fingerprint_model.py

# Run the improved model (recommended)
python final_baseline_model.py
```

### Expected Output
```
Training time: ~170 seconds
ROC AUC: 0.85-0.95 (good performance)
EER: 0.05-0.15 (lower is better)
```

## File Structure
```
balton/
├── dataset/
│   ├── np_data/
│   │   ├── img_train.npy    # Training images
│   │   ├── label_train.npy  # Training labels
│   │   ├── img_real.npy     # Test images
│   │   └── label_real.npy   # Test labels
│   ├── train_data/          # Raw training images
│   └── real_data/           # Raw test images
├── baseline_fingerprint_model.py    # Initial baseline
├── final_baseline_model.py          # Improved version
├── training_curves.png              # Training loss plots
├── performance_analysis.png         # ROC curve and score distributions
└── fingerprint_final_model.pth      # Saved model weights
```

## Key Features

1. **Robust Data Handling**: Supports both numpy arrays and raw images
2. **Flexible Architecture**: Modular CNN design for easy modification
3. **Comprehensive Evaluation**: Multiple performance metrics
4. **Visualization**: Training curves and performance plots
5. **Model Persistence**: Save/load trained models

## Technical Details

### CNN Architecture
```
Input (128x128x1) → Conv(32) → Pool → Conv(64) → Pool → 
Conv(128) → Pool → Conv(256) → Pool → FC(512) → FC(128)
```

### Siamese Network Training
1. Extract features from image pairs
2. Normalize feature vectors (L2 normalization)
3. Calculate Euclidean distance
4. Apply contrastive loss
5. Backpropagate and update weights

### Evaluation Process
1. Generate 1000 comparison trials
2. 500 genuine pairs (same person)
3. 500 impostor pairs (different people)
4. Calculate distance scores
5. Compute ROC curve and metrics

## Results Interpretation

- **ROC AUC > 0.8**: Good performance
- **EER < 0.2**: Acceptable accuracy
- **Clear score separation**: Well-trained model

## Troubleshooting

### Common Issues
1. **NaN metrics**: Usually due to insufficient test data
2. **Memory errors**: Reduce batch size
3. **Import errors**: Check dependencies installation

### Solutions
- Use training data for evaluation if test set is small
- Implement proper error handling for edge cases
- Ensure all dependencies are installed in the poetry environment

## Future Improvements

1. **Data Augmentation**: Increase training data diversity
2. **Advanced Architectures**: Try attention mechanisms or transformers
3. **Multi-scale Features**: Combine features from different scales
4. **Real-time Optimization**: Optimize for edge devices
5. **Anti-spoofing**: Add liveness detection capabilities

## References

- FVC2000: Fingerprint Verification Competition 2000
- Siamese Networks for One-shot Learning
- Contrastive Loss for Deep Metric Learning 