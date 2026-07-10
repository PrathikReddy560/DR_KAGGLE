import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from config import APTOS_DIR, GANomalyConfig as C, ON_KAGGLE, SEED
from dataset_ganomaly import AnomalyRetinaDataset, HealthyRetinaDataset
from losses import adversarial_loss, contextual_loss, encoder_loss, gradient_penalty
from model_ganomaly import Discriminator, Generator


torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    if required_arch not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"This PyTorch build has no kernel for {required_arch}. "
            "Kaggle P100 requires a build that includes sm_60; install the "
            "pinned PyTorch 2.6 CUDA 12.6 packages from requirements.txt, "
            "then restart the notebook session."
        )
    # Inputs are always 128x128, so cuDNN can reuse its fastest kernel choice.
    torch.backends.cudnn.benchmark = True
print(f"Running on Kaggle: {ON_KAGGLE} | Device: {device}")


# ---------------------------------------------------------------------------
# Data — validation selects the model and threshold; test stays untouched.
# ---------------------------------------------------------------------------
full_dataset = HealthyRetinaDataset(APTOS_DIR, img_size=C.IMG_SIZE)
n_val = max(1, int(len(full_dataset) * C.VAL_SPLIT))
n_test = max(1, int(len(full_dataset) * C.TEST_SPLIT))
n_train = len(full_dataset) - n_val - n_test
if n_train < 1:
    raise ValueError("Dataset is too small for train/validation/test splitting.")

train_ds, val_healthy_ds, test_healthy_ds = random_split(
    full_dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Train: {n_train} | Val healthy: {n_val} | Test healthy: {n_test}")

num_workers = min(4, os.cpu_count() or 1)
loader_kwargs = {
    "num_workers": num_workers,
    "pin_memory": device.type == "cuda",
    "persistent_workers": num_workers > 0,
}
train_loader = DataLoader(
    train_ds,
    batch_size=C.BATCH_SIZE,
    shuffle=True,
    drop_last=len(train_ds) >= C.BATCH_SIZE,
    **loader_kwargs,
)

anomaly_ds = AnomalyRetinaDataset(APTOS_DIR, img_size=C.IMG_SIZE)
if len(anomaly_ds) < 2:
    raise ValueError("Need at least two DR-positive APTOS images for evaluation.")
anomaly_val_ds, anomaly_test_ds = random_split(
    anomaly_ds,
    [len(anomaly_ds) // 2, len(anomaly_ds) - len(anomaly_ds) // 2],
    generator=torch.Generator().manual_seed(SEED),
)


def make_balanced_eval_loader(healthy_ds, abnormal_ds):
    """Equal class counts keep threshold-based accuracy easy to interpret."""
    n = min(len(healthy_ds), len(abnormal_ds))
    if n == 0:
        raise ValueError("Both healthy and abnormal evaluation splits must be non-empty.")
    dataset = ConcatDataset([
        Subset(healthy_ds, range(n)),
        Subset(abnormal_ds, range(n)),
    ])
    return (
        DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=False, **loader_kwargs),
        [0] * n + [1] * n,
    )


val_loader, val_labels = make_balanced_eval_loader(val_healthy_ds, anomaly_val_ds)
test_loader, test_labels = make_balanced_eval_loader(test_healthy_ds, anomaly_test_ds)
print(f"Validation: {len(val_labels)} | Final test: {len(test_labels)}")


def score_dataset(generator, loader):
    generator.eval()
    scores = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            _, z, z_hat = generator(images)
            scores.extend(torch.mean((z - z_hat) ** 2, dim=(1, 2, 3)).cpu().numpy())
    return np.asarray(scores)


def threshold_at_youden_j(labels, scores):
    """Choose the validation threshold with the best sensitivity/specificity trade-off."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
G = Generator(C.LATENT_DIM).to(device)
D = Discriminator().to(device)
opt_G = Adam(G.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))
opt_D = Adam(D.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))

best_auc = -float("inf")

for epoch in range(C.EPOCHS):
    G.train()
    D.train()
    running_g, running_d = 0.0, 0.0

    for real in train_loader:
        real = real.to(device, non_blocking=True)

        # WGAN-GP trains the critic more often than the generator.
        for _ in range(C.N_CRITIC):
            with torch.no_grad():
                fake, _, _ = G(real)
            score_real, _ = D(real)
            score_fake, _ = D(fake)
            gp = gradient_penalty(D, real, fake, device)
            d_loss = score_fake.mean() - score_real.mean() + C.GP_WEIGHT * gp

            opt_D.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_D.step()

        # D provides feature-matching targets but is not optimized here.
        for parameter in D.parameters():
            parameter.requires_grad_(False)
        fake, z, z_hat = G(real)
        with torch.no_grad():
            _, feat_real = D(real)
        _, feat_fake = D(fake)

        l_adv = adversarial_loss(feat_real, feat_fake)
        l_con = contextual_loss(real, fake)
        l_enc = encoder_loss(z, z_hat)
        g_loss = C.W_ADV * l_adv + C.W_CON * l_con + C.W_ENC * l_enc

        opt_G.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_G.step()
        for parameter in D.parameters():
            parameter.requires_grad_(True)

        running_g += g_loss.item()
        running_d += d_loss.item()

    # Validation is the only split used for selecting a checkpoint.
    val_scores = score_dataset(G, val_loader)
    val_auc = roc_auc_score(val_labels, val_scores)
    print(
        f"Epoch {epoch + 1}/{C.EPOCHS} | "
        f"G: {running_g / len(train_loader):.4f} | "
        f"D: {running_d / len(train_loader):.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        threshold = threshold_at_youden_j(val_labels, val_scores)
        torch.save(
            {
                "generator_state_dict": G.state_dict(),
                "epoch": epoch + 1,
                "val_auc": best_auc,
                "threshold": threshold,
                "img_size": C.IMG_SIZE,
                "latent_dim": C.LATENT_DIM,
                "normalization": "(x / 255 - 0.5) / 0.5",
            },
            C.CKPT_PATH,
        )
        print(f"  -> New best validation AUC ({best_auc:.4f}), checkpoint saved.")

checkpoint = torch.load(C.CKPT_PATH, map_location=device)
G.load_state_dict(checkpoint["generator_state_dict"])
test_scores = score_dataset(G, test_loader)
test_auc = roc_auc_score(test_labels, test_scores)
test_predictions = (test_scores >= checkpoint["threshold"]).astype(int)
test_accuracy = (test_predictions == np.asarray(test_labels)).mean()

print(f"Training complete. Best validation AUC: {best_auc:.4f}")
print(f"Final test AUC: {test_auc:.4f} | Balanced-set accuracy: {test_accuracy:.4f}")
print(f"Best checkpoint: {C.CKPT_PATH}")
