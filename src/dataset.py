# %%
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

# %%
class FingerprintDataset(Dataset):
    """Custom Dataset for fingerprint images"""
    
    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        # Check if numpy data exists first
        if os.path.exists(os.path.join(data_path, 'np_data')):
            print("Using numpy data format")
            self.load_numpy_data()
        else:
            print("Using image data format")
            self.load_image_data()
    
    def load_numpy_data(self):
        """Load data from numpy arrays"""
        np_path = os.path.join(self.data_path, 'np_data')
        
        if self.mode == 'train':
            self.images = np.load(os.path.join(np_path, 'img_train.npy'))
            self.labels = np.load(os.path.join(np_path, 'label_train.npy'))
        else:
            self.images = np.load(os.path.join(np_path, 'img_real.npy'))
            self.labels = np.load(os.path.join(np_path, 'label_real.npy'))
        
        print(f"Loaded {len(self.images)} {self.mode} images from numpy arrays")
        print(f"Image shape: {self.images.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Unique labels: {np.unique(self.labels.flatten())}")
    
    def load_image_data(self):
        """Load data from image files"""
        # Priority: Use new split directories if available, fallback to original structure
        if self.mode == 'train':
            # Check for new split structure first
            if os.path.exists(os.path.join(self.data_path, 'train_split')):
                img_dir = os.path.join(self.data_path, 'train_split')
                print(f"Using new train split directory: {img_dir}")
            else:
                img_dir = os.path.join(self.data_path, 'train_data')
                print(f"Using original train directory: {img_dir}")
        else:  # test mode (includes 'test' and 'real' modes)
            # Check for new split structure first
            if os.path.exists(os.path.join(self.data_path, 'test_split')):
                img_dir = os.path.join(self.data_path, 'test_split')
                print(f"Using new test split directory: {img_dir}")
            else:
                img_dir = os.path.join(self.data_path, 'real_data')
                print(f"Using original test directory: {img_dir}")
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        
        self.image_paths = []
        self.labels = []
        
        # Get all image files and sort them
        image_files = [f for f in os.listdir(img_dir) if f.endswith('.bmp')]
        image_files.sort()
        
        for filename in image_files:
            filepath = os.path.join(img_dir, filename)
            self.image_paths.append(filepath)
            
            # Extract label from filename
            # Handle both formats: "00001_02.bmp" and "00001.bmp"
            if '_' in filename:
                label = int(filename.split('_')[0])
            else:
                label = int(filename.split('.')[0])
            
            self.labels.append(label)
        
        self.images = None
        print(f"Found {len(self.image_paths)} {self.mode} images")
        
        # Print distribution by person
        from collections import Counter
        labels_for_count = self.labels.flatten().tolist() if isinstance(self.labels, np.ndarray) else self.labels
        label_counts = Counter(labels_for_count)
        print(f"Distribution by person: {dict(sorted(label_counts.items()))}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.images is not None:
            # Using numpy arrays
            image = self.images[idx]
            image = np.squeeze(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(image.astype(np.uint8))
        else:
            # Using image files
            image = Image.open(self.image_paths[idx]).convert('L')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def get_stats(self):
        """Get dataset statistics"""
        from collections import Counter
        
        # Handle both numpy arrays and lists
        if isinstance(self.labels, np.ndarray):
            labels_flat = self.labels.flatten().tolist()
        else:
            labels_flat = self.labels
            
        label_counts = Counter(labels_flat)
        
        stats = {
            'total_images': len(self.labels),
            'num_classes': len(label_counts),
            'images_per_class': dict(sorted(label_counts.items())),
            'min_images_per_class': min(label_counts.values()),
            'max_images_per_class': max(label_counts.values()),
            'avg_images_per_class': sum(label_counts.values()) / len(label_counts)
        }
        
        return stats

    def print_stats(self):
        """Print dataset statistics"""
        stats = self.get_stats()
        print(f"\n=== {self.mode.upper()} Dataset Statistics ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"Images per class: {stats['images_per_class']}")
        print(f"Min/Max/Avg images per class: {stats['min_images_per_class']}/{stats['max_images_per_class']}/{stats['avg_images_per_class']:.1f}")
 
# %%