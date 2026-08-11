from pathlib import Path

# ==========================
# Project Directories
# ==========================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "PlantVillage"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# ==========================
# Model Files
# ==========================

MODEL_NAME = "plant_disease_mobilenetv2.keras"
MODEL_PATH = MODEL_DIR / MODEL_NAME

BEST_MODEL_NAME = "best_plant_disease_model.keras"
BEST_MODEL_PATH = MODEL_DIR / BEST_MODEL_NAME

# ==========================
# Evaluation Files
# ==========================

CONFUSION_MATRIX_PATH = LOG_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH = LOG_DIR / "classification_report.txt"
ACCURACY_PLOT_PATH = LOG_DIR / "accuracy.png"
LOSS_PLOT_PATH = LOG_DIR / "loss.png"

# ==========================
# Dataset Configuration
# ==========================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ==========================
# Training Configuration
# ==========================

EPOCHS = 5
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
PATIENCE = 2