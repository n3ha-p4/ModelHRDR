"""
Configuration file for DR vs HR Classification Model
All paths, hyperparameters, and settings are defined here.
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diabetic Retinopathy data
DR_IMAGES_DIR = os.path.join(BASE_DIR, "Diabetic Retinopathy Images", "DR")

# Hypertensive Retinopathy data (from two separate datasets)
HR_IMAGES_DIR = os.path.join(BASE_DIR, "Hypertensive Retinopathy Images", "Hypertensive Retinopathy")
HR_CLASSIFICATION_IMAGES_DIR = os.path.join(
    BASE_DIR, "Hypertensive Retinopathy Images", "Hypertensive Classification",
    "1-Images", "1-Training Set"
)
HR_CLASSIFICATION_LABELS_CSV = os.path.join(
    BASE_DIR, "Hypertensive Retinopathy Images", "Hypertensive Classification",
    "2-Groundtruths", "HRDC Hypertensive Classification Training Labels.csv"
)

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

for d in [OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CLASS LABELS
# ============================================================
CLASS_NAMES = ["Diabetic Retinopathy", "Hypertensive Retinopathy"]
NUM_CLASSES = 2

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
IMAGE_SIZE = 224           # Input size for EfficientNet
CLAHE_CLIP_LIMIT = 0.03   # CLAHE clip limit for scikit-image

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
NUM_FOLDS = 5              # K-Fold cross-validation
BATCH_SIZE = 16            # Smaller batch for CPU training
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4       # Lower LR for fine-tuning pretrained model
WEIGHT_DECAY = 1e-4
PATIENCE = 5               # Early stopping patience
NUM_WORKERS = 0            # Set to 0 for Windows compatibility
RANDOM_SEED = 42
