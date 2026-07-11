"""Train and evaluate Stage 1 (GANomaly) on the APTOS training split."""

import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from config import APTOS_DIR, GANomalyConfig as C, ON_KAGGLE, OUTPUT_DIR, SEED
from dataset_ganomaly import AnomalyRetinaDataset, HealthyRetinaDataset
from evaluation import save_classification_report
from losses import adversarial_loss, contextual_loss, encoder_loss, gradient_penalty
from model_ganomaly import Discriminator, Generator


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, *, shuffle=False, drop_last=False, device=None, generator=None):
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def score_dataset(generator, loader, device):
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


def make_balanced_eval_loader(healthy_ds, abnormal_ds, device):
    """Use equal class counts only for interpretable binary gatekeeper metrics."""
    count = min(len(healthy_ds), len(abnormal_ds))
    if count == 0:
        raise ValueError("Both healthy and abnormal evaluation splits must be non-empty.")
    dataset = ConcatDataset([Subset(healthy_ds, range(count)), Subset(abnormal_ds, range(count))])
    return make_loader(dataset, device=device), [0] * count + [1] * count


def load_checkpoint(path, device):
    """Load checkpoints on both current and older Kaggle PyTorch builds."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Running on Kaggle: {ON_KAGGLE} | Device: {device}")

    # Two dataset objects preserve exactly the same image ordering: only the
    # training object receives the conservative horizontal-flip augmentation.
    healthy_eval = HealthyRetinaDataset(APTOS_DIR, img_size=C.IMG_SIZE, augment=False)
    healthy_train = HealthyRetinaDataset(APTOS_DIR, img_size=C.IMG_SIZE, augment=True)
    count = len(healthy_eval)
    val_count = max(1, int(count * C.VAL_SPLIT))
    test_count = max(1, int(count * C.TEST_SPLIT))
    train_count = count - val_count - test_count
    if train_count < 2:
        raise ValueError("Dataset is too small for train/validation/test splitting.")
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = permutation[:train_count]
    val_indices = permutation[train_count:train_count + val_count]
    test_indices = permutation[train_count + val_count:]
    train_ds = Subset(healthy_train, train_indices)
    val_healthy_ds = Subset(healthy_eval, val_indices)
    test_healthy_ds = Subset(healthy_eval, test_indices)
    print(f"Train: {len(train_ds)} | Val healthy: {len(val_healthy_ds)} | Test healthy: {len(test_healthy_ds)}")

    train_loader = make_loader(
        train_ds,
        shuffle=True,
        drop_last=len(train_ds) > C.BATCH_SIZE,
        device=device,
        generator=torch.Generator().manual_seed(SEED),
    )
    anomaly_ds = AnomalyRetinaDataset(APTOS_DIR, img_size=C.IMG_SIZE)
    if len(anomaly_ds) < 2:
        raise ValueError("Need at least two DR-positive APTOS images for evaluation.")
    anomaly_val_ds, anomaly_test_ds = random_split(
        anomaly_ds,
        [len(anomaly_ds) // 2, len(anomaly_ds) - len(anomaly_ds) // 2],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader, val_labels = make_balanced_eval_loader(val_healthy_ds, anomaly_val_ds, device)
    test_loader, test_labels = make_balanced_eval_loader(test_healthy_ds, anomaly_test_ds, device)
    print(f"Validation: {len(val_labels)} | Final test: {len(test_labels)}")

    generator = Generator(C.LATENT_DIM).to(device)
    discriminator = Discriminator().to(device)
    generator_optimizer = Adam(generator.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=C.LR, betas=(C.BETA1, C.BETA2))
    generator_scheduler = CosineAnnealingLR(generator_optimizer, T_max=C.EPOCHS, eta_min=C.MIN_LR)
    discriminator_scheduler = CosineAnnealingLR(discriminator_optimizer, T_max=C.EPOCHS, eta_min=C.MIN_LR)

    best_auc, stale_epochs = -float("inf"), 0
    for epoch in range(C.EPOCHS):
        generator.train()
        discriminator.train()
        running_g, running_d, batches = 0.0, 0.0, 0
        for real in train_loader:
            real = real.to(device, non_blocking=True)
            # The generated batch is identical for all critic updates.  The old
            # implementation recomputed it N_CRITIC times, wasting GPU work.
            with torch.no_grad():
                fake_for_critic = generator(real)[0]
            discriminator_loss = None
            for _ in range(C.N_CRITIC):
                score_real, _ = discriminator(real)
                score_fake, _ = discriminator(fake_for_critic)
                penalty = gradient_penalty(discriminator, real, fake_for_critic, device)
                discriminator_loss = score_fake.mean() - score_real.mean() + C.GP_WEIGHT * penalty
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_loss.backward()
                discriminator_optimizer.step()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            fake, z, z_hat = generator(real)
            with torch.no_grad():
                _, features_real = discriminator(real)
            _, features_fake = discriminator(fake)
            generator_loss = (
                C.W_ADV * adversarial_loss(features_real, features_fake)
                + C.W_CON * contextual_loss(real, fake)
                + C.W_ENC * encoder_loss(z, z_hat)
            )
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            generator_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            running_g += generator_loss.item()
            running_d += discriminator_loss.item()
            batches += 1

        val_scores = score_dataset(generator, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_scores)
        generator_scheduler.step()
        discriminator_scheduler.step()
        print(
            f"Epoch {epoch + 1}/{C.EPOCHS} | G: {running_g / batches:.4f} | "
            f"D: {running_d / batches:.4f} | Val AUC: {val_auc:.4f} | "
            f"LR: {generator_optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_auc > best_auc:
            best_auc, stale_epochs = val_auc, 0
            threshold = threshold_at_youden_j(val_labels, val_scores)
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
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
        else:
            stale_epochs += 1
            if stale_epochs >= C.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {stale_epochs} epochs without AUC improvement.")
                break

    checkpoint = load_checkpoint(C.CKPT_PATH, device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    test_scores = score_dataset(generator, test_loader, device)
    test_labels_array = np.asarray(test_labels)
    test_predictions = (test_scores >= checkpoint["threshold"]).astype(int)
    test_auc = roc_auc_score(test_labels_array, test_scores)
    report = save_classification_report(
        test_labels_array,
        test_predictions,
        ("Healthy", "DR present"),
        OUTPUT_DIR,
        "ganomaly_test",
    )
    print(f"Training complete. Best validation AUC: {best_auc:.4f}")
    print(
        f"Final test AUC: {test_auc:.4f} | Accuracy: {report['accuracy']:.4f} | "
        f"Macro F1: {report['macro_f1']:.4f}"
    )
    print(f"Best checkpoint: {C.CKPT_PATH}")


if __name__ == "__main__":
    main()
