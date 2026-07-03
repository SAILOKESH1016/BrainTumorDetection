import os
import hashlib
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def compute_file_hash(filepath):
    """Compute MD5 hash of a file to detect exact duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def scan_and_validate_dataset(data_dir):
    """
    Scans the dataset folder, detects corrupt images, removes exact duplicates,
    and returns a clean DataFrame with columns: filepath, label.
    """
    data_list = []
    seen_hashes = {}
    duplicates = []
    corrupted = []
    
    # Supported extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    print(f"Scanning dataset in: {data_dir}")
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
                
            filepath = os.path.join(root, file)
            label = os.path.basename(root).lower()
            
            # Validate Image Integrity
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify image file is not corrupted
                
                # Check for duplicate
                file_hash = compute_file_hash(filepath)
                if file_hash in seen_hashes:
                    duplicates.append((filepath, seen_hashes[file_hash]))
                    continue
                else:
                    seen_hashes[file_hash] = filepath
                    
                data_list.append({
                    'filepath': filepath,
                    'label': label
                })
            except Exception as e:
                corrupted.append((filepath, str(e)))
                
    df = pd.DataFrame(data_list)
    
    print("\n--- Image Scan Summary ---")
    print(f"Total Valid Images Found: {len(df)}")
    print(f"Duplicates Detected & Removed: {len(duplicates)}")
    print(f"Corrupted Images Detected & Removed: {len(corrupted)}")
    
    if len(duplicates) > 0:
        print(f"Sample Duplicate: {duplicates[0][0]} is duplicate of {duplicates[0][1]}")
    if len(corrupted) > 0:
        print(f"Sample Corrupted Error: {corrupted[0][0]} -> {corrupted[0][1]}")
        
    return df, duplicates, corrupted

def prepare_splits(archive_dir, val_size=0.15, random_state=42):
    """
    Ingests training and testing folders, performs validation,
    splits Training into Train & Validation subsets, and returns clean DataFrames.
    """
    train_dir = os.path.join(archive_dir, 'Training')
    test_dir = os.path.join(archive_dir, 'Testing')
    
    # Ingest and validate
    train_full_df, train_dup, train_corr = scan_and_validate_dataset(train_dir)
    test_df, test_dup, test_corr = scan_and_validate_dataset(test_dir)
    
    # Stratified split for training and validation
    train_df, val_df = train_test_split(
        train_full_df,
        test_size=val_size,
        stratify=train_full_df['label'],
        random_state=random_state
    )
    
    print(f"\nData Splits:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Testing set: {len(test_df)} samples")
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    val_df['label_encoded'] = label_encoder.transform(val_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])
    
    # Save label encoder to models directory
    os.makedirs('models', exist_ok=True)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Saved label_encoder.pkl to models/")
    
    return train_df, val_df, test_df, label_encoder

def get_image_generators(train_df, val_df, test_df, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE):
    """
    Returns data generators for training (with augmentation) and validation/testing (only normalization).
    """
    # Augmentation options for Training MRI scans (rotation, zooms, shifts, horizontal flip)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation and testing generators only rescale (normalization)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators using flow_from_dataframe
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save preprocessing configuration
    preprocessor_config = {
        'image_size': img_size,
        'batch_size': batch_size,
        'rescale': 1./255
    }
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor_config, f)
    print("Saved preprocessor.pkl to models/")
    
    return train_generator, val_generator, test_generator

if __name__ == '__main__':
    # Dry run script to verify ingestion pipeline
    archive_dir = 'archive'
    if os.path.exists(archive_dir):
        train_df, val_df, test_df, le = prepare_splits(archive_dir)
        train_gen, val_gen, test_gen = get_image_generators(train_df, val_df, test_df)
        print("Data ingestion and generator configuration successful!")
    else:
        print(f"Error: Archive directory '{archive_dir}' not found.")
