"""
Recompute fold-wise predictions from saved fold checkpoints and run evaluation.
Outputs plots and metrics_summary.json into output/plots.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import config
from dataset import gather_image_paths_and_labels, RetinopathyDataset, get_val_transforms
from evaluate import run_evaluation
from model import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_paths, all_labels = gather_image_paths_and_labels()

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )

    all_preds = []
    all_labels_out = []
    all_probs = []
    fold_accuracies = []

    for fold, (_, val_idx) in enumerate(skf.split(all_paths, all_labels), start=1):
        model_path = os.path.join(config.MODEL_DIR, f"model_fold{fold}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")

        val_paths = all_paths[val_idx]
        val_labels = all_labels[val_idx]

        val_dataset = RetinopathyDataset(val_paths, val_labels, get_val_transforms())
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        fold_preds = []
        fold_labels = []
        fold_probs = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())

        fold_acc = correct / total if total else 0.0
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold} Accuracy: {fold_acc:.4f}")

        all_preds.extend(fold_preds)
        all_labels_out.extend(fold_labels)
        all_probs.extend(fold_probs)

    print("\nRunning full evaluation and saving summary...")
    metrics = run_evaluation(
        np.array(all_labels_out),
        np.array(all_preds),
        np.array(all_probs),
        fold_accuracies,
    )

    print("\nDone.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Fold Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Fold Accuracy: {np.std(fold_accuracies):.4f}")
    print(f"Saved JSON: {os.path.join(config.PLOTS_DIR, 'metrics_summary.json')}")


if __name__ == "__main__":
    main()
