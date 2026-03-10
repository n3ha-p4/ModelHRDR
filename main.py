"""
=============================================================================
  DR vs HR Retinopathy Classification — Main Entry Point
=============================================================================
  This script orchestrates the full pipeline:
    1. Data collection from all image sources
    2. Stratified K-Fold cross-validation training with EfficientNet-B0
    3. Model evaluation (accuracy, precision, recall, F1, confusion matrix)
    4. GradCAM heatmap generation

  Usage:
    python main.py                 # Run full pipeline
    python main.py --gradcam-only  # Only generate GradCAM (requires trained model)
=============================================================================
"""
import sys
import random
import numpy as np
import torch

import config
from dataset import gather_image_paths_and_labels
from train import run_cross_validation
from evaluate import run_evaluation
from gradcam import generate_gradcam_grid
from model import build_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    print("=" * 60)
    print("  DR vs HR RETINOPATHY CLASSIFICATION")
    print("  Using EfficientNet-B0 + scikit-image CLAHE")
    print("  with Stratified K-Fold Cross Validation")
    print("=" * 60)

    # Set seeds for reproducibility
    set_seed(config.RANDOM_SEED)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check if we only want GradCAM
    gradcam_only = "--gradcam-only" in sys.argv

    if gradcam_only:
        print("\n  Running GradCAM only (loading saved model)...")
        import os
        model_path = os.path.join(config.MODEL_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            print(f"  ERROR: No trained model found at {model_path}")
            print("  Please run the full pipeline first: python main.py")
            return
        model = build_model(num_classes=config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        generate_gradcam_grid(model, device, num_samples=5)
        return

    # -------------------------------------------------------
    # STEP 1: Gather all image paths and labels
    # -------------------------------------------------------
    print("\n  Step 1: Collecting images...")
    all_paths, all_labels = gather_image_paths_and_labels()

    # -------------------------------------------------------
    # STEP 2: Train with K-Fold Cross Validation
    # -------------------------------------------------------
    print("\n  Step 2: Training with cross-validation...")
    all_preds, all_labels_out, all_probs, fold_accs, best_model = run_cross_validation(
        all_paths, all_labels, device
    )

    # -------------------------------------------------------
    # STEP 3: Evaluate
    # -------------------------------------------------------
    print("\n  Step 3: Evaluating model...")
    metrics = run_evaluation(all_labels_out, all_preds, all_probs, fold_accs)

    # -------------------------------------------------------
    # STEP 4: GradCAM Heatmaps
    # -------------------------------------------------------
    print("\n  Step 4: Generating GradCAM heatmaps...")
    generate_gradcam_grid(best_model, device, num_samples=5)

    # -------------------------------------------------------
    # DONE
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"  Models saved in:   {config.MODEL_DIR}")
    print(f"  Plots saved in:    {config.PLOTS_DIR}")
    print(f"  GradCAM saved in:  {config.GRADCAM_DIR}")
    print(f"\n  Final Cross-Validation Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
