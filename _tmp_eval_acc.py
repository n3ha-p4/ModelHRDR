import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

import config
from dataset import gather_image_paths_and_labels, RetinopathyDataset, get_val_transforms
from model import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_paths, all_labels = gather_image_paths_and_labels()
skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

fold_accs = []
for fold, (_, val_idx) in enumerate(skf.split(all_paths, all_labels), start=1):
    val_paths = all_paths[val_idx]
    val_labels = all_labels[val_idx]

    val_ds = RetinopathyDataset(val_paths, val_labels, get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = build_model(num_classes=config.NUM_CLASSES, pretrained=False).to(device)
    model_path = os.path.join(config.MODEL_DIR, f'model_fold{fold}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0
    fold_accs.append(acc)
    print(f'Fold {fold}: {acc:.6f}')

print('Fold accuracies:', [round(a, 6) for a in fold_accs])
print(f'Mean: {np.mean(fold_accs):.6f}')
print(f'Std: {np.std(fold_accs):.6f}')
