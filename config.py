import os

# ---------------------------------------------------------------------------
# Environment detection — same script runs unmodified locally (quick shape/
# syntax checks) or on Kaggle (real training on the P100).
# ---------------------------------------------------------------------------
ON_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None

if ON_KAGGLE:
    # NOTE: these folder names must match the exact slugs Kaggle mounts your
    # datasets under — check the sidebar of your Kaggle notebook (or
    # `!ls /kaggle/input`) and correct these if they don't match.
    APTOS_DIR = "/kaggle/input/datasets/mariaherrerot/aptos2019"
    IDRID_DIR = "/kaggle/input/datasets/lakshmiprathik/idrid-dr-dataset"
    ODIR_DIR = "/kaggle/input/datasets/lakshmiprathik/odir-5k"
    # confirmed via actual notebook output — NOT the flat /kaggle/input/<slug>
    # pattern the other three assume. Verify whether APTOS/IDRiD/ODIR follow
    # this same "datasets/<owner>/<slug>" nesting before trusting those three.
    MESSIDOR_DIR = "/kaggle/input/datasets/lakshmiprathik/messidor2-dr-dataset"
    OUTPUT_DIR = "/kaggle/working"
else:
    # Local dev machine — NOT used for real training (4GB GTX 1650 OOMs).
    # Only here so the scripts don't crash if you run a quick syntax check
    # on your laptop before pushing to GitHub.
    ROOT = os.path.dirname(os.path.abspath(__file__))
    APTOS_DIR = os.path.join(ROOT, "sample_data", "aptos2019")
    IDRID_DIR = os.path.join(ROOT, "sample_data", "idrid")
    ODIR_DIR = os.path.join(ROOT, "sample_data", "odir5k")
    MESSIDOR_DIR = os.path.join(ROOT, "sample_data", "messidor2")
    OUTPUT_DIR = os.path.join(ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


class GANomalyConfig:
    """Stage 1 — anomaly detection gatekeeper."""
    IMG_SIZE = 128
    LATENT_DIM = 100
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 2e-4
    BETA1, BETA2 = 0.5, 0.999
    VAL_SPLIT = 0.15          # held-out healthy images for checkpointing
    W_ADV = 1.0               # adversarial (feature-matching) weight
    W_CON = 50.0              # contextual (SSIM) weight — GANomaly paper default
    W_ENC = 1.0               # latent/encoder weight
    GP_WEIGHT = 10.0          # WGAN-GP penalty weight
    N_CRITIC = 5              # discriminator steps per generator step
    CKPT_PATH = os.path.join(OUTPUT_DIR, "ganomaly_best.pth")


class EffNetConfig:
    """Stage 2 — severity grader. Not implemented yet; placeholders only
    so config.py doesn't need to change again once Stage 2 starts."""
    IMG_SIZE = 224            # EfficientNet-B0's native input resolution — confirm this fits your plan
    BATCH_SIZE = 32
    NUM_CLASSES = 5
    PHASE1_EPOCHS = 10        # frozen backbone
    PHASE2_EPOCHS = 30        # unfrozen fine-tune
    PHASE1_LR = 1e-3
    PHASE2_LR = 1e-5
    CKPT_PATH = os.path.join(OUTPUT_DIR, "effnet_best.pth")