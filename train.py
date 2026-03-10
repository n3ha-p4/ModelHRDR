"""
Training pipeline with Stratified K-Fold Cross Validation.
Handles class imbalance with weighted loss and weighted sampling.
Includes early stopping and learning rate scheduling.
"""
import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from dataset import RetinopathyDataset, get_train_transforms, get_val_transforms
from model import build_model


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for handling imbalance."""
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights)


def create_weighted_sampler(labels):
    """Create a WeightedRandomSampler so each batch is class-balanced."""
    counts = np.bincount(labels)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Evaluate on validation set and return loss, accuracy, predictions, and true labels."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def train_fold(fold, train_paths, train_labels, val_paths, val_labels, device):
    """
    Train a single fold: builds model, trains with early stopping,
    returns the best model and validation results.
    """
    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1}/{config.NUM_FOLDS}")
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)}")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = RetinopathyDataset(train_paths, train_labels, get_train_transforms())
    val_dataset = RetinopathyDataset(val_paths, val_labels, get_val_transforms())

    # Weighted sampler for balanced batches during training
    sampler = create_weighted_sampler(train_labels)

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=use_pin_memory,
    )

    # Build model with pretrained weights
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=True).to(device)

    # Weighted loss for class imbalance
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer: different learning rates for frozen vs unfrozen layers
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    # Training loop with early stopping
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    best_results = None

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels_out, val_probs = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch+1:2d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_results = {
                "val_preds": val_preds,
                "val_labels": val_labels_out,
                "val_probs": val_probs,
                "val_acc": val_acc,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (patience={config.PATIENCE})")
            break

    print(f"  Best Val Accuracy for Fold {fold+1}: {best_val_acc:.4f}")

    # Save best model for this fold
    model_path = os.path.join(config.MODEL_DIR, f"model_fold{fold+1}.pth")
    torch.save(best_model_state, model_path)

    # Save fold results to disk for resume capability
    results_path = os.path.join(config.MODEL_DIR, f"results_fold{fold+1}.json")
    with open(results_path, "w") as f:
        json.dump({
            "val_acc": float(best_results["val_acc"]),
            "val_preds": best_results["val_preds"].tolist(),
            "val_labels": best_results["val_labels"].tolist(),
            "val_probs": best_results["val_probs"].tolist(),
        }, f)

    # Load best model state for returning
    model.load_state_dict(best_model_state)

    return model, best_results, val_paths, val_labels


def resume_fold(fold, val_paths, val_labels, device):
    """
    Resume a completed fold by loading the saved model and results.
    Skips training entirely.
    """
    model_path = os.path.join(config.MODEL_DIR, f"model_fold{fold+1}.pth")
    results_path = os.path.join(config.MODEL_DIR, f"results_fold{fold+1}.json")

    print(f"\n{'='*60}")
    print(f"  FOLD {fold + 1}/{config.NUM_FOLDS} — RESUMING (already completed)")
    print(f"{'='*60}")

    # Load saved model
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Load saved results
    with open(results_path, "r") as f:
        saved = json.load(f)

    results = {
        "val_preds": np.array(saved["val_preds"]),
        "val_labels": np.array(saved["val_labels"]),
        "val_probs": np.array(saved["val_probs"]),
        "val_acc": saved["val_acc"],
    }

    print(f"  Loaded saved model and results. Val Accuracy: {results['val_acc']:.4f}")
    return model, results


def run_cross_validation(all_paths, all_labels, device):
    """
    Run Stratified K-Fold cross-validation across all data.
    Returns aggregated predictions for evaluation.
    """
    print("\n" + "=" * 60)
    print("  STARTING STRATIFIED K-FOLD CROSS VALIDATION")
    print(f"  Folds: {config.NUM_FOLDS} | Epochs: {config.NUM_EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE} | LR: {config.LEARNING_RATE}")
    print("=" * 60)

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    all_fold_preds = []
    all_fold_labels = []
    all_fold_probs = []
    fold_accuracies = []
    best_model = None
    best_acc = 0.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        train_paths = all_paths[train_idx]
        train_labels = all_labels[train_idx]
        val_paths = all_paths[val_idx]
        val_labels = all_labels[val_idx]

        # Check if this fold was already completed (resume support)
        fold_model_path = os.path.join(config.MODEL_DIR, f"model_fold{fold+1}.pth")
        fold_results_path = os.path.join(config.MODEL_DIR, f"results_fold{fold+1}.json")
        if os.path.exists(fold_model_path) and os.path.exists(fold_results_path):
            model, results = resume_fold(fold, val_paths, val_labels, device)
        else:
            model, results, _, _ = train_fold(
                fold, train_paths, train_labels, val_paths, val_labels, device
            )

        all_fold_preds.extend(results["val_preds"])
        all_fold_labels.extend(results["val_labels"])
        all_fold_probs.extend(results["val_probs"])
        fold_accuracies.append(results["val_acc"])

        if results["val_acc"] > best_acc:
            best_acc = results["val_acc"]
            best_model = model

    # Save the overall best model
    best_model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    torch.save(best_model.state_dict(), best_model_path)
    print(f"\n  Best model saved to {best_model_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1}: {acc:.4f}")
    print(f"  Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print("=" * 60)

    return (
        np.array(all_fold_preds),
        np.array(all_fold_labels),
        np.array(all_fold_probs),
        fold_accuracies,
        best_model,
    )
