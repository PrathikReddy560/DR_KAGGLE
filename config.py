"""Central configuration for the two-stage diabetic-retinopathy pipeline.

The defaults are deliberately conservative for a Kaggle P100.  Override the
data mount with ``APTOS_DIR`` rather than editing source code in a notebook.
"""

import os


ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def kaggle_path(env_name, default):
    """Allow a notebook to override a mounted Kaggle dataset path."""
    return os.environ.get(env_name, default)


if ON_KAGGLE:
    # Supports the APTOS copy used during the original Stage-1 experiment.
    APTOS_DIR = kaggle_path(
        "APTOS_DIR", "/kaggle/input/datasets/mariaherrerot/aptos2019"
    )
    OUTPUT_DIR = kaggle_path("OUTPUT_DIR", "/kaggle/working")
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    APTOS_DIR = kaggle_path("APTOS_DIR", os.path.join(ROOT, "sample_data", "aptos2019"))
    OUTPUT_DIR = kaggle_path("OUTPUT_DIR", os.path.join(ROOT, "outputs"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative DR")


class GANomalyConfig:
    """Stage 1: healthy-retina anomaly gatekeeper."""

    IMG_SIZE = 128
    LATENT_DIM = 100
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 2e-4
    BETA1, BETA2 = 0.5, 0.999
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    W_ADV = 1.0
    W_CON = 50.0
    W_ENC = 1.0
    GP_WEIGHT = 10.0
    N_CRITIC = 5
    # Cosine decay avoids the late high-learning-rate divergence seen in the
    # original run; early stopping keeps the best validation checkpoint.
    MIN_LR = 1e-5
    EARLY_STOPPING_PATIENCE = 50
    CKPT_PATH = os.path.join(OUTPUT_DIR, "ganomaly_best.pth")


class EffNetConfig:
    """Stage 2 settings transcribed from the approved methodology report."""

    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # Section 2.1 of the report specifies five head-only epochs followed by
    # 50 epochs with the last 20 backbone layers unfrozen. The final full-model
    # phase completes the requested progressive fine-tuning protocol.
    HEAD_EPOCHS = 5
    LAST_20_EPOCHS = 50
    FULL_FINETUNE_EPOCHS = 15
    LAST_BACKBONE_LAYERS = 20
    HEAD_LR = 3e-4
    LAST_LAYERS_LR = 3e-5
    FULL_BACKBONE_LR = 1e-5
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    DROPOUT = 0.35  # Method A classifier-head dropout / MC Dropout probability
    GRADIENT_CLIP_NORM = 1.0
    COSINE_MIN_LR = 1e-6
    PLATEAU_FACTOR = 0.3
    PLATEAU_PATIENCE = 3
    EARLY_STOPPING_PATIENCE = 10

    # Report section 3.4.1: LAB L-channel CLAHE, clipLimit=2.0, 8x8 tiles.
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    AUGMENT_ROTATION_DEGREES = 15
    AUGMENT_BRIGHTNESS = 0.20
    AUGMENT_CONTRAST = 0.20

    # Report section 3.5.3: 20-30 MC passes, 10-20 VBLL samples, Gate 2=0.70.
    MC_DROPOUT_SAMPLES = 25
    VBLL_SAMPLES = 15
    CONFIDENCE_THRESHOLD = 0.70
    GANOMALY_GATE_THRESHOLD = 0.1982
    VBLL_PRIOR_STD = 1.0
    VBLL_KL_WEIGHT = 1.0
    ECE_BINS = 15
    USE_BALANCED_SAMPLER = True

    # APTOS is primary. Attach IDRiD/Messidor-2 and provide one manifest per
    # dataset through STAGE2_EXTRA_MANIFESTS (path,diagnosis[,id_code]).
    EXTRA_MANIFESTS = os.environ.get("STAGE2_EXTRA_MANIFESTS", "")
    SYNTHETIC_TRAIN_MANIFEST = os.environ.get("STAGE2_SYNTHETIC_MANIFEST", "")
    USE_PRETRAINED = True
    MC_CHECKPOINT = os.path.join(OUTPUT_DIR, "effnet_mc_dropout_best.pth")
    VBLL_CHECKPOINT = os.path.join(OUTPUT_DIR, "effnet_vbll_best.pth")
    SPLITS_PATH = os.path.join(OUTPUT_DIR, "stage2_splits.csv")
