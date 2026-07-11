"""Stage-2 data handling exactly matching the report's preprocessing protocol."""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
import torchvision.transforms as T

from dataset_ganomaly import aptos_dataframe


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CLAHELAB:
    """Apply the report's CLAHE operation to the L-channel in LAB space."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def stage2_transform(img_size, train, *, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE plus the report's flips, +/-15 degree rotation, and colour jitter."""
    transforms = [CLAHELAB(clip_limit, tile_grid_size), T.Resize((img_size, img_size))]
    if train:
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15, fill=(0, 0, 0)),
            T.ColorJitter(brightness=0.20, contrast=0.20),
        ])
    transforms.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(transforms)


class DRSeverityDataset(Dataset):
    """Five-class fundus dataset; CLAHE is applied identically in every split."""

    def __init__(self, frame, img_size, train, clip_limit=2.0, tile_grid_size=(8, 8)):
        required = {"sample_id", "diagnosis", "path"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Expected Stage-2 columns: {sorted(required)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = stage2_transform(
            img_size, train, clip_limit=clip_limit, tile_grid_size=tile_grid_size
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["diagnosis"])


def _validate_grades(frame, source_name):
    if frame.empty:
        raise ValueError(f"{source_name} has no usable images.")
    invalid = sorted(set(frame["diagnosis"].astype(int)) - set(range(5)))
    if invalid:
        raise ValueError(f"{source_name} has diagnoses outside 0..4: {invalid}")


def _normalise_manifest(manifest_path, source_name):
    """Load an attached IDRiD/Messidor-2/synthetic manifest without guessing layout.

    Each manifest must contain ``path`` and ``diagnosis``; paths can be relative
    to the CSV. ``id_code`` is optional and defaults to the image filename.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-2 manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    if not {"path", "diagnosis"}.issubset(frame.columns):
        raise ValueError(f"{manifest_path} must contain path and diagnosis columns.")
    frame = frame.copy()
    frame["path"] = frame["path"].map(
        lambda value: str((manifest_path.parent / value).resolve())
        if not os.path.isabs(str(value)) else str(Path(value).resolve())
    )
    missing = [path for path in frame["path"] if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{manifest_path} references missing images; first: {missing[0]}")
    frame["diagnosis"] = frame["diagnosis"].astype(int)
    frame["id_code"] = frame.get("id_code", frame["path"].map(lambda value: Path(value).stem)).astype(str)
    frame["source"] = source_name
    frame["sample_id"] = frame["source"] + ":" + frame["id_code"]
    _validate_grades(frame, source_name)
    return frame.loc[:, ["sample_id", "id_code", "diagnosis", "path", "source"]]


def collect_stage2_records(aptos_dir, extra_manifests=""):
    """Combine APTOS primary records with explicitly attached report datasets."""
    aptos = aptos_dataframe(aptos_dir).copy()
    aptos["source"] = "aptos2019"
    aptos["sample_id"] = aptos["source"] + ":" + aptos["id_code"].astype(str)
    records = [aptos.loc[:, ["sample_id", "id_code", "diagnosis", "path", "source"]]]
    # A semicolon-separated list works in both Kaggle/Linux and Windows, while
    # avoiding the colon inside a Windows drive letter being treated as a split.
    for manifest in filter(None, (part.strip() for part in extra_manifests.split(";"))):
        records.append(_normalise_manifest(manifest, Path(manifest).stem.lower()))
    frame = pd.concat(records, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected across Stage-2 sources.")
    _validate_grades(frame, "combined Stage-2 data")
    return frame


def load_synthetic_training_records(manifest_path):
    """Optionally ingest the report's cGAN minority samples into *training only*.

    Synthetic generation is intentionally not recreated here: the report does not
    specify a cGAN architecture and no synthetic artifact is included in this
    repository. This loader prevents synthetic records ever reaching validation
    or test splits when a prepared cGAN manifest is attached.
    """
    return _normalise_manifest(manifest_path, "synthetic_cgan") if manifest_path else None


def create_or_load_splits(aptos_dir, extra_manifests, split_path, train_split, val_split, test_split, seed):
    """Persist a deterministic 70/15/15 stratified split for every real image."""
    if abs(train_split + val_split + test_split - 1.0) > 1e-8:
        raise ValueError("Stage-2 split fractions must add to 1.")
    source = collect_stage2_records(aptos_dir, extra_manifests)
    if sorted(source["diagnosis"].unique()) != [0, 1, 2, 3, 4]:
        raise ValueError("Stage 2 requires all five DR grades 0..4 in the training pool.")

    if os.path.isfile(split_path):
        manifest = pd.read_csv(split_path)
        required = {"sample_id", "diagnosis", "split"}
        if not required.issubset(manifest.columns):
            raise ValueError(f"Existing split manifest is invalid: {split_path}")
        if set(manifest["sample_id"]) != set(source["sample_id"]):
            raise ValueError(
                "The saved split manifest does not match the current real-image pool. "
                f"Delete {split_path} only to start a new experiment."
            )
    else:
        first = StratifiedShuffleSplit(n_splits=1, test_size=val_split + test_split, random_state=seed)
        train_indices, heldout_indices = next(first.split(source, source["diagnosis"]))
        heldout = source.iloc[heldout_indices]
        second = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_split / (val_split + test_split),
            random_state=seed + 1,
        )
        val_indices, test_indices = next(second.split(heldout, heldout["diagnosis"]))
        manifest = pd.concat([
            source.iloc[train_indices].assign(split="train"),
            heldout.iloc[val_indices].assign(split="val"),
            heldout.iloc[test_indices].assign(split="test"),
        ], ignore_index=True).loc[:, ["sample_id", "diagnosis", "split"]]
        os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
        manifest.to_csv(split_path, index=False)

    frame = source.merge(manifest, on=["sample_id", "diagnosis"], how="inner", validate="one_to_one")
    if len(frame) != len(source):
        raise RuntimeError("A saved Stage-2 split could not be matched to an image.")
    counts = frame.groupby(["split", "diagnosis"]).size().unstack(fill_value=0)
    print("Stage-2 real-image split counts:\n" + counts.to_string())
    return frame
