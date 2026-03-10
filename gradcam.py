"""
GradCAM Visualization Module
Generates heatmaps showing which regions of the retinal images
the model focuses on when making predictions.
Uses the pytorch-grad-cam library for reliable GradCAM computation.
"""
import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from skimage import io as sk_io, color, transform as sk_transform, exposure
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from model import build_model, get_target_layer
from dataset import apply_clahe, get_val_transforms, gather_image_paths_and_labels


def load_and_preprocess_image(img_path):
    """Load an image and apply scikit-image preprocessing (returns both raw and preprocessed)."""
    image = sk_io.imread(img_path)

    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    elif image.shape[2] == 4:
        image = color.rgba2rgb(image)

    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0

    image = sk_transform.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), anti_aliasing=True)
    enhanced = apply_clahe(image)
    enhanced = np.clip(enhanced, 0, 1).astype(np.float32)

    return enhanced


def generate_gradcam_for_image(model, img_path, true_label, device, save_path):
    """Generate and save GradCAM heatmap for a single image."""
    # Preprocess image
    rgb_img = load_and_preprocess_image(img_path)

    # Create tensor for model
    transform = get_val_transforms()
    pil_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # Set up GradCAM
    target_layer = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Generate heatmap for the predicted class
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Overlay heatmap on original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original (CLAHE Enhanced)", fontsize=12)
    axes[0].axis("off")

    # GradCAM heatmap only
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("GradCAM Heatmap", fontsize=12)
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(visualization)
    true_name = config.CLASS_NAMES[true_label]
    pred_name = config.CLASS_NAMES[pred_class]
    correct = "✓" if true_label == pred_class else "✗"
    axes[2].set_title(
        f"Overlay — Pred: {pred_name} ({confidence:.1%}) {correct}\n"
        f"True: {true_name}",
        fontsize=11,
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return pred_class, confidence


def generate_gradcam_grid(model, device, num_samples=5):
    """
    Generate GradCAM visualizations for a grid of sample images.
    Selects samples from both classes: correctly and incorrectly classified.
    """
    print("\n" + "#" * 60)
    print("  GENERATING GRADCAM HEATMAPS")
    print("#" * 60)

    # Gather all images
    all_paths, all_labels = gather_image_paths_and_labels()

    # Select random samples from each class
    dr_indices = np.where(all_labels == 0)[0]
    hr_indices = np.where(all_labels == 1)[0]

    random.seed(config.RANDOM_SEED)
    dr_samples = random.sample(list(dr_indices), min(num_samples, len(dr_indices)))
    hr_samples = random.sample(list(hr_indices), min(num_samples, len(hr_indices)))

    selected = dr_samples + hr_samples

    print(f"  Generating GradCAM for {len(selected)} images...")

    for i, idx in enumerate(selected):
        img_path = all_paths[idx]
        true_label = all_labels[idx]
        class_name = config.CLASS_NAMES[true_label].replace(" ", "_")
        save_path = os.path.join(config.GRADCAM_DIR, f"gradcam_{class_name}_{i+1}.png")

        pred_class, confidence = generate_gradcam_for_image(
            model, img_path, true_label, device, save_path
        )

        status = "CORRECT" if pred_class == true_label else "WRONG"
        print(f"  [{status}] {os.path.basename(img_path)} -> "
              f"{config.CLASS_NAMES[pred_class]} ({confidence:.1%})")

    # Create a combined summary grid
    _create_summary_grid(model, all_paths, all_labels, device)

    print(f"\n  GradCAM heatmaps saved to: {config.GRADCAM_DIR}")


def _create_summary_grid(model, all_paths, all_labels, device):
    """Create a 2x4 summary grid showing GradCAM for both classes."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    random.seed(config.RANDOM_SEED + 1)

    for row, class_idx in enumerate([0, 1]):
        indices = np.where(all_labels == class_idx)[0]
        samples = random.sample(list(indices), min(4, len(indices)))

        for col, idx in enumerate(samples):
            img_path = all_paths[idx]
            rgb_img = load_and_preprocess_image(img_path)

            # Generate GradCAM
            transform = get_val_transforms()
            pil_img = Image.fromarray((rgb_img * 255).astype(np.uint8))
            input_tensor = transform(pil_img).unsqueeze(0)
            input_tensor = input_tensor.to(next(model.parameters()).device)

            target_layer = get_target_layer(model)
            cam = GradCAM(model=model, target_layers=[target_layer])
            targets = [ClassifierOutputTarget(class_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            axes[row, col].imshow(visualization)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(config.CLASS_NAMES[class_idx], fontsize=14, fontweight="bold")

    fig.suptitle(
        "GradCAM Heatmaps — Model Decision Regions\n"
        "(Top: Diabetic Retinopathy, Bottom: Hypertensive Retinopathy)",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(config.GRADCAM_DIR, "gradcam_summary_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary grid saved to {path}")
