import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Grid / canvas
GRID_SIZE = 16
CELL_SIZE = 24

# Per-stage latent dimensions
STAGE1_Z = 4
STAGE2_Z = 3
STAGE3_Z = 3
STAGE4_Z = 3

# Stage data files
STAGE1_DATA = os.path.join(DATA_DIR, "stage1_heads.csv")

STAGE2_BASE_DATA = os.path.join(DATA_DIR, "stage2_eyes_base.csv")
STAGE2_TARGET_DATA = os.path.join(DATA_DIR, "stage2_eyes_target.csv")

STAGE3_BASE_DATA = os.path.join(DATA_DIR, "stage3_smile_base.csv")
STAGE3_TARGET_DATA = os.path.join(DATA_DIR, "stage3_smile_target.csv")

STAGE4_BASE_DATA = os.path.join(DATA_DIR, "stage4_detail_base.csv")
STAGE4_TARGET_DATA = os.path.join(DATA_DIR, "stage4_detail_target.csv")

# Model weight files
MODEL_STAGE1 = os.path.join(DATA_DIR, "model_stage1.pth")
MODEL_STAGE2 = os.path.join(DATA_DIR, "model_stage2.pth")
MODEL_STAGE3 = os.path.join(DATA_DIR, "model_stage3.pth")
MODEL_STAGE4 = os.path.join(DATA_DIR, "model_stage4.pth")

# Minimum samples required before training is allowed
STAGE_MIN_SAMPLES = {
    1: 30,
    2: 60,
    3: 40,
    4: 40,
}

# Training hyperparameters
KL_WARMUP_START = 150
KL_WARMUP_END = 600
KL_FINAL_BETA = 0.3
TRAINING_EPOCHS = 800
TRAINING_LR = 5e-4
NOISE_FACTOR = 0.03
SHARPNESS_WEIGHT = 0.15

# Rendering: threshold for binarizing sigmoid outputs
# With sharpening loss, the model produces crisp 0-or-1 outputs,
# so 0.5 works well. Only raise this if outputs are still soft.
RENDER_THRESHOLD = 0.5

# Stage metadata (for UI labels)
STAGE_NAMES = {
    1: "Head Shapes",
    2: "Eyes",
    3: "Smile",
    4: "Details",
}

STAGE_ICONS = {
    1: "🟠",
    2: "👁",
    3: "😊",
    4: "✨",
}

# Maps stage number -> (base_csv, target_csv, model_path)
# Stage 1 has no base (unconditional), so base is None
STAGE_FILES = {
    1: (None, STAGE1_DATA, MODEL_STAGE1),
    2: (STAGE2_BASE_DATA, STAGE2_TARGET_DATA, MODEL_STAGE2),
    3: (STAGE3_BASE_DATA, STAGE3_TARGET_DATA, MODEL_STAGE3),
    4: (STAGE4_BASE_DATA, STAGE4_TARGET_DATA, MODEL_STAGE4),
}
