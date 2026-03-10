"""
Dataset preparation and PyTorch Dataset class.
- Gathers DR and HR images from all sources
- Applies scikit-image CLAHE preprocessing
- Handles data augmentation with torchvision transforms
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage import exposure, color, io, transform as sk_transform

import config

# Suppress harmless color space conversion warnings from scikit-image
warnings.filterwarnings("ignore", message=".*negative Z values.*")


def gather_image_paths_and_labels():
    """
    Collect all image file paths and assign labels:
      - Label 0: Diabetic Retinopathy (DR)
      - Label 1: Hypertensive Retinopathy (HR)

    HR images come from two separate datasets:
      1. Hypertensive Retinopathy/ folder (all are HR)
      2. Hypertensive Classification images labeled as Hypertensive=1
    """
    paths = []
    labels = []

    # --- Diabetic Retinopathy images (label=0) ---
    dr_files = glob.glob(os.path.join(config.DR_IMAGES_DIR, "*.*"))
    for f in dr_files:
        paths.append(f)
        labels.append(0)
    print(f"  DR images found: {len(dr_files)}")

    # --- Hypertensive Retinopathy images from dedicated folder (label=1) ---
    hr_files = glob.glob(os.path.join(config.HR_IMAGES_DIR, "*.*"))
    for f in hr_files:
        paths.append(f)
        labels.append(1)
    print(f"  HR images (dedicated folder): {len(hr_files)}")

    # --- Hypertensive Classification images labeled as Hypertensive (label=1) ---
    hr_csv = pd.read_csv(config.HR_CLASSIFICATION_LABELS_CSV)
    hr_positive = hr_csv[hr_csv["Hypertensive"] == 1]
    hc_count = 0
    for _, row in hr_positive.iterrows():
        img_name = row["Image"]
        img_path = os.path.join(config.HR_CLASSIFICATION_IMAGES_DIR, img_name)
        if os.path.exists(img_path):
            paths.append(img_path)
            labels.append(1)
            hc_count += 1
    print(f"  HR images (classification dataset): {hc_count}")

    total_dr = labels.count(0)
    total_hr = labels.count(1)
    print(f"  Total: {total_dr} DR + {total_hr} HR = {total_dr + total_hr} images")

    return np.array(paths), np.array(labels)


def apply_clahe(image_np):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    using scikit-image to enhance retinal image contrast.
    This is a standard preprocessing step in retinal image analysis.
    """
    # Convert to LAB color space (if RGB)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        lab = color.rgb2lab(image_np)
        # Apply CLAHE only to the L (lightness) channel
        l_channel = lab[:, :, 0]
        l_channel = l_channel / 100.0  # Normalize to [0, 1] for CLAHE
        l_channel = exposure.equalize_adapthist(
            l_channel, clip_limit=config.CLAHE_CLIP_LIMIT
        )
        lab[:, :, 0] = l_channel * 100.0  # Scale back
        enhanced = color.lab2rgb(lab)
    else:
        # Grayscale image
        enhanced = exposure.equalize_adapthist(
            image_np, clip_limit=config.CLAHE_CLIP_LIMIT
        )
    return enhanced


def get_train_transforms():
    """Augmentation transforms for training (applied after CLAHE)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """Validation transforms (no augmentation, only normalize)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RetinopathyDataset(Dataset):
    """
    Custom PyTorch Dataset for retinal images.
    Applies scikit-image CLAHE preprocessing and torchvision transforms.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image with scikit-image (returns float64 in [0,1] for most formats)
        image = io.imread(img_path)

        # Handle RGBA or grayscale images
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = color.rgba2rgb(image)

        # Ensure image is float in [0, 1] for scikit-image processing
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255.0

        # Resize using scikit-image
        image = sk_transform.resize(
            image, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
            anti_aliasing=True
        )

        # Apply CLAHE enhancement with scikit-image
        image = apply_clahe(image)

        # Clip to valid range and convert to uint8 for PIL
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)

        # Convert to PIL for torchvision transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
