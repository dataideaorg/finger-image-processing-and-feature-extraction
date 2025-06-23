"""
Dataset module for fingerprint image loading and preprocessing.
"""

import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


class FingerprintDataset(Dataset):
    """Custom Dataset for fingerprint images"""
    
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
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
                label = int(filename.split('_')[0])
                self.labels.append(label)
        
        self.images = None
        print(f"Found {len(self.image_paths)} {self.mode} images")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.images is not None:
            image = self.images[idx]
            image = np.squeeze(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image.astype(np.uint8))
        else:
            image = Image.open(self.image_paths[idx]).convert('L')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label 