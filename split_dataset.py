#!/usr/bin/env python3
"""
Dataset Splitting Script
Splits the fingerprint dataset into 80% training and 20% testing
while maintaining proper stratification across persons.
"""

import os
import shutil
import random
from collections import defaultdict
import numpy as np

def create_directories():
    """Create new directory structure"""
    directories = [
        'dataset/train_split',
        'dataset/test_split'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    
    print("Created new directory structure:")
    for directory in directories:
        print(f"  - {directory}")

def analyze_current_dataset():
    """Analyze current dataset structure"""
    print("=== Current Dataset Analysis ===")
    
    # Analyze training data
    train_files = []
    if os.path.exists('dataset/train_data'):
        train_files = [f for f in os.listdir('dataset/train_data') if f.endswith('.bmp')]
    
    # Analyze test data
    test_files = []
    if os.path.exists('dataset/real_data'):
        test_files = [f for f in os.listdir('dataset/real_data') if f.endswith('.bmp')]
    
    print(f"Current training files: {len(train_files)}")
    print(f"Current test files: {len(test_files)}")
    
    # Group by person
    person_files = defaultdict(list)
    
    # Add training files
    for filename in train_files:
        person_id = filename.split('_')[0]
        person_files[person_id].append(os.path.join('dataset/train_data', filename))
    
    # Add test files (they're named differently)
    for filename in test_files:
        person_id = filename.split('.')[0]  # e.g., "00001.bmp" -> "00001"
        person_files[person_id].append(os.path.join('dataset/real_data', filename))
    
    print(f"\nPerson distribution:")
    total_images = 0
    for person_id in sorted(person_files.keys()):
        count = len(person_files[person_id])
        print(f"  Person {person_id}: {count} images")
        total_images += count
    
    print(f"\nTotal images across all sources: {total_images}")
    return person_files

def split_dataset(person_files, train_ratio=0.8):
    """Split dataset maintaining person distribution"""
    print(f"\n=== Splitting Dataset ({int(train_ratio*100)}% train, {int((1-train_ratio)*100)}% test) ===")
    
    train_files = []
    test_files = []
    
    for person_id in sorted(person_files.keys()):
        files = person_files[person_id]
        n_files = len(files)
        
        if n_files < 2:
            print(f"Warning: Person {person_id} has only {n_files} image(s). Adding to training set.")
            train_files.extend(files)
            continue
        
        # Shuffle files for random selection
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        
        # Calculate split
        n_train = max(1, int(n_files * train_ratio))  # At least 1 for training
        n_test = n_files - n_train
        
        # Split
        person_train = files_shuffled[:n_train]
        person_test = files_shuffled[n_train:]
        
        train_files.extend(person_train)
        test_files.extend(person_test)
        
        print(f"  Person {person_id}: {n_train} train, {n_test} test (total: {n_files})")
    
    print(f"\nFinal split:")
    print(f"  Training: {len(train_files)} images")
    print(f"  Testing: {len(test_files)} images")
    print(f"  Total: {len(train_files) + len(test_files)} images")
    
    return train_files, test_files

def copy_files(train_files, test_files):
    """Copy files to new directory structure"""
    print(f"\n=== Copying Files ===")
    
    # Copy training files
    print("Copying training files...")
    for i, src_path in enumerate(train_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join('dataset/train_split', filename)
        shutil.copy2(src_path, dst_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(train_files)} training files")
    
    print(f"  Completed copying {len(train_files)} training files")
    
    # Copy test files
    print("Copying test files...")
    for i, src_path in enumerate(test_files):
        filename = os.path.basename(src_path)
        dst_path = os.path.join('dataset/test_split', filename)
        shutil.copy2(src_path, dst_path)
        
        if (i + 1) % 20 == 0:
            print(f"  Copied {i + 1}/{len(test_files)} test files")
    
    print(f"  Completed copying {len(test_files)} test files")

def verify_split():
    """Verify the new split"""
    print(f"\n=== Verifying New Split ===")
    
    # Count files in new directories
    train_split_files = [f for f in os.listdir('dataset/train_split') if f.endswith('.bmp')]
    test_split_files = [f for f in os.listdir('dataset/test_split') if f.endswith('.bmp')]
    
    print(f"New training files: {len(train_split_files)}")
    print(f"New test files: {len(test_split_files)}")
    
    # Check person distribution in new split
    train_persons = defaultdict(int)
    test_persons = defaultdict(int)
    
    for filename in train_split_files:
        if '_' in filename:
            person_id = filename.split('_')[0]
        else:
            person_id = filename.split('.')[0]
        train_persons[person_id] += 1
    
    for filename in test_split_files:
        if '_' in filename:
            person_id = filename.split('_')[0]
        else:
            person_id = filename.split('.')[0]
        test_persons[person_id] += 1
    
    print(f"\nPerson distribution in new split:")
    all_persons = set(train_persons.keys()) | set(test_persons.keys())
    for person_id in sorted(all_persons):
        train_count = train_persons.get(person_id, 0)
        test_count = test_persons.get(person_id, 0)
        total = train_count + test_count
        train_pct = (train_count / total * 100) if total > 0 else 0
        print(f"  Person {person_id}: {train_count} train ({train_pct:.1f}%), {test_count} test, total: {total}")

def main():
    """Main function"""
    print("=== Fingerprint Dataset Splitting Tool ===")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create new directory structure
    create_directories()
    
    # Analyze current dataset
    person_files = analyze_current_dataset()
    
    if not person_files:
        print("Error: No dataset files found!")
        return
    
    # Split dataset
    train_files, test_files = split_dataset(person_files, train_ratio=0.8)
    
    # Copy files
    copy_files(train_files, test_files)
    
    # Verify split
    verify_split()
    
    print(f"\n=== Dataset Split Complete! ===")
    print(f"New dataset structure:")
    print(f"  dataset/train_split/ - {len(train_files)} training images")
    print(f"  dataset/test_split/  - {len(test_files)} test images")
    print(f"\nNext steps:")
    print(f"1. Update your dataset class to use the new directories")
    print(f"2. Update your training script to use train_split and test_split")

if __name__ == "__main__":
    main() 