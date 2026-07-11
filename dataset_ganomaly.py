"""APTOS dataset utilities shared by both stages.

Only files that are listed in the APTOS CSV are used.  This prevents accidental
mixing of unrelated ODIR images into the healthy Stage-1 training set.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def _find_aptos_training_layout(aptos_dir):
    """Support official APTOS names and the project's Kaggle dataset copy."""
    csv_names = ("train.csv", "train_1.csv")
    image_dirs = (os.path.join("train_images", "train_images"), "train_images")
    csv_path = next(
        (os.path.join(aptos_dir, name) for name in csv_names
         if os.path.isfile(os.path.join(aptos_dir, name))),
        None,
    )
    img_dir = next(
        (os.path.join(aptos_dir, name) for name in image_dirs
         if os.path.isdir(os.path.join(aptos_dir, name))),
        None,
    )
    if csv_path is None or img_dir is None:
        raise FileNotFoundError(
            "Could not find APTOS data. Expected train.csv/train_1.csv and "
            "train_images/ (or train_images/train_images/) under: " + aptos_dir
        )
    return csv_path, img_dir


def aptos_dataframe(aptos_dir):
    """Return valid image paths and integer diagnoses from an APTOS mount.

    A few public repackagings have CSV rows without a corresponding image; they
    are intentionally dropped with a warning instead of failing mid-training.
    """
    csv_path, img_dir = _find_aptos_training_layout(aptos_dir)
    df = pd.read_csv(csv_path)
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain {sorted(required_columns)}; "
            f"found {df.columns.tolist()}"
        )

    records, missing = [], 0
    for row in df.loc[:, ["id_code", "diagnosis"]].itertuples(index=False):
        image_id, diagnosis = str(row.id_code), int(row.diagnosis)
        path = next(
            (os.path.join(img_dir, image_id + suffix)
             for suffix in (".png", ".jpg", ".jpeg")
             if os.path.isfile(os.path.join(img_dir, image_id + suffix))),
            None,
        )
        if path is None:
            missing += 1
            continue
        records.append({"id_code": image_id, "diagnosis": diagnosis, "path": path})

    if not records:
        raise FileNotFoundError(f"No APTOS images listed in {csv_path} were found.")
    if missing:
        print(f"Warning: skipped {missing} CSV rows with no matching image file.")
    return pd.DataFrame.from_records(records)


def crop_retina(image, tolerance=7):
    """Remove black camera borders without altering retinal content.

    The operation is deterministic and is applied to train, validation, test,
    and inference images.  It avoids the shortcut where border size correlates
    with a dataset source or class.
    """
    array = np.asarray(image.convert("RGB"))
    mask = array.mean(axis=2) > tolerance
    if not mask.any():
        return image
    rows, cols = np.where(mask)
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    # Keep pathological/invalid crops from producing tiny images.
    if bottom - top < 32 or right - left < 32:
        return image
    return image.crop((left, top, right, bottom))


class RetinaCrop:
    """Torchvision-compatible deterministic retinal-border crop."""

    def __call__(self, image):
        return crop_retina(image)


def ganomaly_transform(img_size, augment=False):
    """Normalise to [-1, 1], matching the GAN decoder's Tanh output."""
    transforms = [RetinaCrop(), T.Resize((img_size, img_size))]
    if augment:
        # A horizontal flip is anatomically plausible and does not erase small
        # lesions; more aggressive image edits hurt reconstruction objectives.
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return T.Compose(transforms)


class HealthyRetinaDataset(Dataset):
    """Verified APTOS Grade-0 images for healthy-only GANomaly training."""

    def __init__(self, aptos_dir, img_size=128, augment=False):
        df = aptos_dataframe(aptos_dir)
        self.paths = df.loc[df["diagnosis"] == 0, "path"].tolist()
        if not self.paths:
            raise ValueError("No Grade-0 APTOS images were found.")
        self.transform = ganomaly_transform(img_size, augment=augment)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            return self.transform(image.convert("RGB"))


class AnomalyRetinaDataset(Dataset):
    """APTOS images with any DR (diagnosis > 0) for binary evaluation only."""

    def __init__(self, aptos_dir, img_size=128):
        df = aptos_dataframe(aptos_dir)
        self.paths = df.loc[df["diagnosis"] > 0, "path"].tolist()
        if not self.paths:
            raise ValueError("No DR-positive APTOS images were found.")
        self.transform = ganomaly_transform(img_size, augment=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            return self.transform(image.convert("RGB"))


if __name__ == "__main__":
    import sys

    dataset = HealthyRetinaDataset(sys.argv[1], img_size=128)
    print(f"Found {len(dataset)} healthy images")
    sample = dataset[0]
    print("Sample shape:", sample.shape, "range:", sample.min().item(), sample.max().item())
