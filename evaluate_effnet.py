"""Compare report-specified MC Dropout and VBLL Stage-2 models on one held-out split."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

from config import APTOS_DIR, CLASS_NAMES, EffNetConfig as C, GANomalyConfig as G, OUTPUT_DIR, SEED
from dataset_effnet import DRSeverityDataset, create_or_load_splits
from dataset_ganomaly import ganomaly_transform
from evaluation import save_uncertainty_evaluation
from model_effnet import build_stage2_model, predictive_distribution
from model_ganomaly import Generator


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_for_method(method):
    return C.MC_CHECKPOINT if method == "mc_dropout" else C.VBLL_CHECKPOINT


def make_loader(frame, checkpoint, device):
    clahe = checkpoint.get("clahe", {})
    dataset = DRSeverityDataset(
        frame,
        checkpoint.get("img_size", C.IMG_SIZE),
        train=False,
        clip_limit=clahe.get("clip_limit", C.CLAHE_CLIP_LIMIT),
        tile_grid_size=tuple(clahe.get("tile_grid_size", C.CLAHE_TILE_GRID_SIZE)),
    )
    workers = min(4, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=C.BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def score_ganomaly_paths(paths, checkpoint_path, device):
    """Run the locked report Gate-1 score on the held-out Stage-2 paths."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = Generator(checkpoint.get("latent_dim", G.LATENT_DIM)).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    transform = ganomaly_transform(checkpoint.get("img_size", G.IMG_SIZE), augment=False)
    scores = []
    for start in range(0, len(paths), C.BATCH_SIZE):
        images = []
        for path in paths[start:start + C.BATCH_SIZE]:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
        _, z, z_hat = model(torch.stack(images).to(device, non_blocking=True))
        scores.extend(torch.mean((z - z_hat) ** 2, dim=(1, 2, 3)).cpu().tolist())
    return np.asarray(scores, dtype=float)


@torch.no_grad()
def predict(model, loader, method, samples, device):
    labels, predictions, means, variances = [], [], [], []
    for images, targets in loader:
        mean, variance = predictive_distribution(
            model, images.to(device, non_blocking=True), method, samples
        )
        labels.extend(targets.tolist())
        predictions.extend(mean.argmax(dim=1).cpu().tolist())
        means.append(mean.cpu())
        variances.append(variance.cpu())
    return (
        np.asarray(labels),
        np.asarray(predictions),
        torch.cat(means).numpy(),
        torch.cat(variances).numpy(),
    )


def evaluate_method(method, checkpoint_path, frame, args, device, gate1_scores=None):
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("method") != method:
        raise ValueError(f"{checkpoint_path} is not a {method} checkpoint.")
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    model = build_stage2_model(
        method, len(class_names), C.DROPOUT, C.VBLL_PRIOR_STD, use_pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(frame, checkpoint, device)
    samples = C.MC_DROPOUT_SAMPLES if method == "mc_dropout" else C.VBLL_SAMPLES
    labels, predictions, probabilities, variance = predict(model, loader, method, samples, device)
    threshold = float(checkpoint.get("confidence_threshold", C.CONFIDENCE_THRESHOLD))
    confidence = probabilities.max(axis=1)
    confidence_pass = confidence >= threshold
    # Method A follows its report definition (confidence-only acceptance).
    # Method B follows the exact dual gate: GANomaly >= 0.1982 AND confidence >= 0.70.
    if method == "vbll":
        if gate1_scores is None:
            raise FileNotFoundError("VBLL evaluation requires the completed GANomaly checkpoint for Gate 1.")
        gate1_pass = gate1_scores >= C.GANOMALY_GATE_THRESHOLD
        accepted = gate1_pass & confidence_pass
    else:
        gate1_pass = None
        accepted = confidence_pass

    prefix = f"{method}_{args.split}"
    report = save_uncertainty_evaluation(
        labels, predictions, probabilities, variance, accepted, class_names,
        args.output_dir, prefix, threshold, gate1_pass=gate1_pass,
    )
    table = frame.loc[:, ["sample_id", "source", "id_code", "diagnosis", "path"]].copy()
    table["prediction"] = predictions
    table["confidence"] = confidence
    table["uncertainty"] = variance.mean(axis=1)
    table["accepted"] = accepted
    table["referred"] = ~accepted
    if gate1_scores is not None:
        table["ganomaly_score"] = gate1_scores
        table["ganomaly_gate_pass"] = gate1_scores >= C.GANOMALY_GATE_THRESHOLD
    for grade in range(len(class_names)):
        table[f"predictive_mean_{grade}"] = probabilities[:, grade]
        table[f"predictive_variance_{grade}"] = variance[:, grade]
    table.to_csv(os.path.join(args.output_dir, f"{prefix}_predictions.csv"), index=False)
    accepted_metrics = report["accepted_predictions"]
    accuracy = accepted_metrics["accuracy"]
    print(
        f"{method}: accepted accuracy={accuracy if accuracy is not None else 'n/a'} | "
        f"coverage={report['coverage']:.4f} | rejection={report['rejection_rate']:.4f} | "
        f"ECE={report['ece_all']:.4f} | binary AUC={report['binary_dr']['binary_dr_auc']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mc_dropout", "vbll", "both"), default="both")
    parser.add_argument("--checkpoint", help="Override a checkpoint when evaluating exactly one method.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--ganomaly-checkpoint", default=G.CKPT_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.checkpoint and args.method == "both":
        parser.error("--checkpoint can only be used with one --method.")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = create_or_load_splits(
        APTOS_DIR, C.EXTRA_MANIFESTS, C.SPLITS_PATH,
        C.TRAIN_SPLIT, C.VAL_SPLIT, C.TEST_SPLIT, SEED,
    )
    frame = frame.loc[frame["split"] == args.split].reset_index(drop=True)
    methods = ("mc_dropout", "vbll") if args.method == "both" else (args.method,)
    gate1_scores = None
    if "vbll" in methods:
        if not Path(args.ganomaly_checkpoint).is_file():
            raise FileNotFoundError(
                "The report's primary VBLL evaluation requires ganomaly_best.pth. "
                f"Not found: {args.ganomaly_checkpoint}"
            )
        gate1_scores = score_ganomaly_paths(frame["path"].tolist(), args.ganomaly_checkpoint, device)
    for method in methods:
        evaluate_method(
            method,
            args.checkpoint or checkpoint_for_method(method),
            frame,
            args,
            device,
            gate1_scores=gate1_scores if method == "vbll" else None,
        )


if __name__ == "__main__":
    main()
