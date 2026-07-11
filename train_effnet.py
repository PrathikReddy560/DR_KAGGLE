"""Train both report-specified Stage-2 EfficientNet-B0 uncertainty methods."""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import APTOS_DIR, CLASS_NAMES, EffNetConfig as C, ON_KAGGLE, SEED
from dataset_effnet import (
    DRSeverityDataset,
    create_or_load_splits,
    load_synthetic_training_records,
)
from model_effnet import (
    build_stage2_model,
    configure_finetuning,
    freeze_nontrainable_batchnorm,
)


def seed_everything(seed):
    """Set every relevant RNG for repeatable Kaggle experiments."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, device, *, sampler=None, shuffle=False, seed=SEED):
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def build_loaders(device):
    """Build real-image splits and keep optional cGAN images train-only."""
    frame = create_or_load_splits(
        APTOS_DIR,
        C.EXTRA_MANIFESTS,
        C.SPLITS_PATH,
        C.TRAIN_SPLIT,
        C.VAL_SPLIT,
        C.TEST_SPLIT,
        SEED,
    )
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    synthetic = load_synthetic_training_records(C.SYNTHETIC_TRAIN_MANIFEST)
    if synthetic is not None:
        train_frame = pd.concat([train_frame, synthetic], ignore_index=True)
        print(f"Added {len(synthetic)} attached cGAN samples to the training set only.")
    else:
        print("No cGAN synthetic manifest attached; training uses real images only.")
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)

    train_dataset = DRSeverityDataset(
        train_frame,
        C.IMG_SIZE,
        train=True,
        clip_limit=C.CLAHE_CLIP_LIMIT,
        tile_grid_size=C.CLAHE_TILE_GRID_SIZE,
    )
    val_dataset = DRSeverityDataset(
        val_frame, C.IMG_SIZE, train=False, clip_limit=C.CLAHE_CLIP_LIMIT,
        tile_grid_size=C.CLAHE_TILE_GRID_SIZE,
    )
    counts = train_frame["diagnosis"].value_counts().reindex(range(C.NUM_CLASSES), fill_value=0)
    if (counts == 0).any():
        raise ValueError(f"A training class is missing after preparation: {counts.to_dict()}")
    inverse_frequency = len(train_frame) / (C.NUM_CLASSES * counts.to_numpy(dtype=float))
    class_weights = torch.tensor(inverse_frequency, dtype=torch.float32, device=device)
    if C.USE_BALANCED_SAMPLER:
        sample_weights = train_frame["diagnosis"].map(lambda grade: inverse_frequency[int(grade)]).to_numpy()
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        train_loader = make_loader(train_dataset, device, sampler=sampler)
    else:
        train_loader = make_loader(train_dataset, device, shuffle=True)
    val_loader = make_loader(val_dataset, device, seed=SEED + 1)
    return train_loader, val_loader, class_weights


def make_optimizer(model, phase):
    head = list(model.head_parameters())
    backbone = [parameter for parameter in model.backbone.features.parameters() if parameter.requires_grad]
    if phase == "head":
        return AdamW(head, lr=C.HEAD_LR, weight_decay=C.WEIGHT_DECAY)
    return AdamW(
        [
            {"params": head, "lr": C.LAST_LAYERS_LR},
            {
                "params": backbone,
                "lr": C.LAST_LAYERS_LR if phase == "last_20" else C.FULL_BACKBONE_LR,
            },
        ],
        weight_decay=C.WEIGHT_DECAY,
    )


def make_grad_scaler(device):
    """Use the current AMP API while retaining compatibility with older Kaggle images."""
    try:
        return torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=device.type == "cuda")


def train_epoch(model, loader, criterion, optimizer, scaler, device, method, train_size):
    model.train()
    freeze_nontrainable_batchnorm(model)
    total_loss, labels, predictions = 0.0, [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if method == "mc_dropout":
                logits = model(images)
                loss = criterion(logits, targets)
                nll, kl = loss.detach(), torch.zeros((), device=device)
            else:
                features = model.extract_features(images)
                # VBLL numerics remain float32 while the convolutional backbone
                # benefits from AMP. This is the ELBO: weighted NLL + KL/N.
                with torch.autocast(device_type=device.type, enabled=False):
                    logits = model.vbll.sample_logits(features.float(), samples=1).squeeze(0)
                    nll = criterion(logits, targets)
                    kl = model.vbll.kl_divergence() * C.VBLL_KL_WEIGHT / train_size
                    loss = nll + kl
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRADIENT_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * targets.size(0)
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(labels, predictions), f1_score(
        labels, predictions, average="macro", zero_division=0
    )


@torch.no_grad()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)  # dropout off / VBLL posterior mean for stable model selection
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), accuracy_score(labels, predictions), f1_score(
        labels, predictions, average="macro", zero_division=0
    )


def checkpoint_path(method):
    return C.MC_CHECKPOINT if method == "mc_dropout" else C.VBLL_CHECKPOINT


def train_method(method, train_loader, val_loader, class_weights, device):
    model = build_stage2_model(
        method, C.NUM_CLASSES, C.DROPOUT, C.VBLL_PRIOR_STD, use_pretrained=C.USE_PRETRAINED
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=C.LABEL_SMOOTHING)
    scaler = make_grad_scaler(device)
    phases = (
        ("head", C.HEAD_EPOCHS, "cosine"),
        ("last_20", C.LAST_20_EPOCHS, "cosine"),
        ("full", C.FULL_FINETUNE_EPOCHS, "plateau"),
    )
    best_f1, global_epoch = -1.0, 0
    checkpoint = checkpoint_path(method)
    for phase, epochs, scheduler_name in phases:
        configure_finetuning(model, phase, C.LAST_BACKBONE_LAYERS)
        optimizer = make_optimizer(model, phase)
        scheduler = (
            CosineAnnealingLR(optimizer, T_max=epochs, eta_min=C.COSINE_MIN_LR)
            if scheduler_name == "cosine" else
            ReduceLROnPlateau(optimizer, mode="max", factor=C.PLATEAU_FACTOR, patience=C.PLATEAU_PATIENCE)
        )
        stale_epochs = 0
        print(f"\n{method}: starting {phase} phase for up to {epochs} epochs.")
        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            train_loss, train_accuracy, train_f1 = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device, method, len(train_loader.dataset)
            )
            val_loss, val_accuracy, val_f1 = validate_epoch(model, val_loader, criterion, device)
            if scheduler_name == "cosine":
                scheduler.step()
            else:
                scheduler.step(val_f1)
            lrs = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
            print(
                f"{method} epoch {global_epoch} [{phase} {phase_epoch}/{epochs}] | "
                f"train loss {train_loss:.4f}, acc {train_accuracy:.4f}, F1 {train_f1:.4f} | "
                f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}, F1 {val_f1:.4f} | LR {lrs}"
            )
            if val_f1 > best_f1:
                best_f1, stale_epochs = val_f1, 0
                torch.save(
                    {
                        "method": method,
                        "model_state_dict": model.state_dict(),
                        "epoch": global_epoch,
                        "phase": phase,
                        "val_macro_f1": best_f1,
                        "class_names": CLASS_NAMES,
                        "img_size": C.IMG_SIZE,
                        "clahe": {"clip_limit": C.CLAHE_CLIP_LIMIT, "tile_grid_size": C.CLAHE_TILE_GRID_SIZE},
                        "confidence_threshold": C.CONFIDENCE_THRESHOLD,
                        "vbll_samples": C.VBLL_SAMPLES,
                        "mc_dropout_samples": C.MC_DROPOUT_SAMPLES,
                    },
                    checkpoint,
                )
                print(f"  -> new best validation macro F1 {best_f1:.4f}: {checkpoint}")
            else:
                stale_epochs += 1
                if stale_epochs >= C.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping {phase} after {stale_epochs} stagnant validation epochs.")
                    break
    print(f"{method} training complete. Best validation macro F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mc_dropout", "vbll", "all"), default="all")
    args = parser.parse_args()
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Kaggle: {ON_KAGGLE} | Device: {device}")
    methods = ("mc_dropout", "vbll") if args.method == "all" else (args.method,)
    for method in methods:
        seed_everything(SEED)
        # Recreate the sampler generator so both methods see the same seeded
        # sampling protocol while retaining independently trained weights.
        train_loader, val_loader, class_weights = build_loaders(device)
        train_method(method, train_loader, val_loader, class_weights, device)


if __name__ == "__main__":
    main()
