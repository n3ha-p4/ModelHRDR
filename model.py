"""
Model definition using transfer learning with EfficientNet-B0.
EfficientNet provides excellent accuracy with efficient computation,
making it ideal for medical image classification with limited data.
"""
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True):
    """
    Build an EfficientNet-B0 model with a custom classification head.

    Strategy:
    - Use pretrained ImageNet weights (transfer learning)
    - Freeze early layers to preserve learned feature extraction
    - Replace the final classifier for our 2-class problem
    - Unfreeze last few blocks for fine-tuning
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last 3 blocks of features for fine-tuning
    # EfficientNet-B0 has features[0..8], we unfreeze from block 6 onwards
    for i in range(6, 9):
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Replace the classifier head
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def get_target_layer(model):
    """Return the target layer for GradCAM visualization."""
    # Last convolutional block in EfficientNet features
    return model.features[-1]
