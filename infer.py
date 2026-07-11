"""Report-aligned Stage-2 inference: uncertainty gates and accepted-only Grad-CAM."""

import argparse
import glob
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

from config import CLASS_NAMES, EffNetConfig as C, GANomalyConfig as G, OUTPUT_DIR
from dataset_effnet import stage2_transform
from dataset_ganomaly import ganomaly_transform
from gradcam import EfficientNetGradCAM, save_gradcam_overlay
from model_effnet import build_stage2_model, predictive_distribution
from model_ganomaly import Generator


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_image_paths(items):
    paths = []
    for item in items:
        candidate = Path(item)
        if candidate.is_dir():
            matches = [path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        else:
            matches = [Path(path) for path in glob.glob(item) if Path(path).suffix.lower() in IMAGE_SUFFIXES]
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                matches = [candidate]
        paths.extend(matches)
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise FileNotFoundError("No supported image files were supplied to --images.")
    return paths


@torch.no_grad()
def score_stage1(paths, checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = Generator(checkpoint.get("latent_dim", G.LATENT_DIM)).to(device)
    model.load_state_dict(checkpoint["generator_state_dict"])
    model.eval()
    transform = ganomaly_transform(checkpoint.get("img_size", G.IMG_SIZE), augment=False)
    scores = []
    for start in range(0, len(paths), C.BATCH_SIZE):
        batch = []
        for path in paths[start:start + C.BATCH_SIZE]:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        _, z, z_hat = model(torch.stack(batch).to(device, non_blocking=True))
        scores.extend(torch.mean((z - z_hat) ** 2, dim=(1, 2, 3)).cpu().tolist())
    return np.asarray(scores, dtype=float)


@torch.no_grad()
def stage2_predict(paths, model, method, transform, samples, device):
    results = []
    for start in range(0, len(paths), C.BATCH_SIZE):
        batch_paths = paths[start:start + C.BATCH_SIZE]
        tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        mean, variance = predictive_distribution(
            model, torch.stack(tensors).to(device, non_blocking=True), method, samples
        )
        for path, probability, posterior_variance in zip(batch_paths, mean.cpu(), variance.cpu()):
            results.append({
                "path": str(path),
                "model_prediction": int(probability.argmax().item()),
                "confidence": float(probability.max().item()),
                "uncertainty": float(posterior_variance.mean().item()),
                "predictive_mean": probability.tolist(),
                "predictive_variance": posterior_variance.tolist(),
            })
    return results


def add_gradcam(results, model, transform, output_dir):
    """Generate overlays solely for Stage-2 accepted predictions, per the report."""
    camera = EfficientNetGradCAM(model)
    try:
        for index, result in enumerate(results):
            if not result["accepted"]:
                result["gradcam_path"] = ""
                continue
            with Image.open(result["path"]) as image:
                original = image.convert("RGB")
                tensor = transform(original).unsqueeze(0).to(next(model.parameters()).device)
            heatmap = camera.generate(tensor, result["predicted_grade"])
            destination = os.path.join(output_dir, f"gradcam_{index:03d}_{Path(result['path']).stem}.png")
            result["gradcam_path"] = save_gradcam_overlay(original, heatmap, destination)
    finally:
        camera.remove()


def save_summary_grid(results, output_path):
    columns = min(4, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.8 * rows), squeeze=False)
    for axis, result in zip(axes.flat, results):
        image_path = result["gradcam_path"] or result["path"]
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        title = (
            f"{result['label']} (grade {result['predicted_grade']})\n"
            f"confidence {result['confidence']:.1%}; uncertainty {result['uncertainty']:.4g}\n"
            f"{result['decision']}"
        )
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(results):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def default_checkpoint(method):
    return C.MC_CHECKPOINT if method == "mc_dropout" else C.VBLL_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="Image files, quoted globs, or directories.")
    parser.add_argument("--method", choices=("mc_dropout", "vbll"), default="vbll")
    parser.add_argument("--checkpoint", help="Stage-2 checkpoint; defaults to the chosen method's best checkpoint.")
    parser.add_argument("--ganomaly-checkpoint", default=G.CKPT_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = resolve_image_paths(args.images)
    stage2_checkpoint = load_checkpoint(args.checkpoint or default_checkpoint(args.method), device)
    if stage2_checkpoint.get("method") != args.method:
        raise ValueError("The selected checkpoint does not match --method.")
    if not os.path.isfile(args.ganomaly_checkpoint):
        raise FileNotFoundError(
            "The report's Stage-2 inference requires the completed GANomaly Gate-1 checkpoint: "
            f"{args.ganomaly_checkpoint}"
        )
    class_names = tuple(stage2_checkpoint.get("class_names", CLASS_NAMES))
    model = build_stage2_model(
        args.method, len(class_names), C.DROPOUT, C.VBLL_PRIOR_STD, use_pretrained=False
    ).to(device)
    model.load_state_dict(stage2_checkpoint["model_state_dict"])
    clahe = stage2_checkpoint.get("clahe", {})
    transform = stage2_transform(
        stage2_checkpoint.get("img_size", C.IMG_SIZE),
        train=False,
        clip_limit=clahe.get("clip_limit", C.CLAHE_CLIP_LIMIT),
        tile_grid_size=tuple(clahe.get("tile_grid_size", C.CLAHE_TILE_GRID_SIZE)),
    )
    samples = C.MC_DROPOUT_SAMPLES if args.method == "mc_dropout" else C.VBLL_SAMPLES
    results = stage2_predict(paths, model, args.method, transform, samples, device)
    stage1_scores = score_stage1(paths, args.ganomaly_checkpoint, device)
    threshold = float(stage2_checkpoint.get("confidence_threshold", C.CONFIDENCE_THRESHOLD))
    for result, score in zip(results, stage1_scores):
        stage1_pass = score >= C.GANOMALY_GATE_THRESHOLD
        confidence_pass = result["confidence"] >= threshold
        result["ganomaly_score"] = float(score)
        result["ganomaly_gate_pass"] = bool(stage1_pass)
        result["confidence_gate_pass"] = bool(confidence_pass)
        # Gate 1 below 0.1982 is the report's terminal "No DR Detected" path.
        # A Stage-2 grade and Grad-CAM are issued only after Gate 1 passes and
        # the 0.70 Stage-2 confidence gate also passes.
        result["accepted"] = bool(stage1_pass and confidence_pass)
        result["referred"] = bool(stage1_pass and not confidence_pass)
        if not stage1_pass:
            result["predicted_grade"] = 0
            result["label"] = class_names[0]
            result["decision"] = "No DR Detected"
        elif confidence_pass:
            result["predicted_grade"] = result["model_prediction"]
            result["label"] = class_names[result["predicted_grade"]]
            result["decision"] = "Accepted"
        else:
            result["predicted_grade"] = result["model_prediction"]
            result["label"] = class_names[result["predicted_grade"]]
            result["decision"] = "Refer to Ophthalmologist"
    add_gradcam(results, model, transform, args.output_dir)

    rows = []
    for result in results:
        row = {key: value for key, value in result.items() if key not in {"predictive_mean", "predictive_variance"}}
        row.update({f"predictive_mean_{grade}": value for grade, value in enumerate(result["predictive_mean"])})
        row.update({f"predictive_variance_{grade}": value for grade, value in enumerate(result["predictive_variance"])})
        rows.append(row)
    csv_path = os.path.join(args.output_dir, "inference_predictions.csv")
    grid_path = os.path.join(args.output_dir, "inference_summary.png")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    save_summary_grid(results, grid_path)
    print(f"Predicted {len(results)} image(s) with {args.method} on {device}.")
    print(f"Predictions: {csv_path}")
    print(f"Grad-CAM summary: {grid_path}")


if __name__ == "__main__":
    main()
