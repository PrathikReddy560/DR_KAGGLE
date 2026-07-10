import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


def _find_aptos_training_layout(aptos_dir):
    """Support both the official APTOS names and this project's Kaggle copy."""
    csv_names = ("train.csv", "train_1.csv")
    image_dirs = (
        os.path.join("train_images", "train_images"),
        "train_images",
    )

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
            "Could not find APTOS training data. Expected either "
            "train.csv or train_1.csv, and train_images/ or "
            "train_images/train_images/, under: " + aptos_dir
        )
    return csv_path, img_dir


def _aptos_paths_for_grades(aptos_dir, include_grade):
    csv_path, img_dir = _find_aptos_training_layout(aptos_dir)
    df = pd.read_csv(csv_path)
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain {sorted(required_columns)}; "
            f"found {df.columns.tolist()}"
        )

    paths = []
    for image_id in df.loc[include_grade(df["diagnosis"]), "id_code"]:
        # PNG is used by this mounted copy; the alternatives support other
        # APTOS repackagings without changing the training code.
        for suffix in (".png", ".jpg", ".jpeg"):
            path = os.path.join(img_dir, f"{image_id}{suffix}")
            if os.path.isfile(path):
                paths.append(path)
                break
    return paths


class HealthyRetinaDataset(Dataset):
    """Verified APTOS-2019 Grade-0 (No DR) images only.

    Raw ODIR folders are intentionally not scanned: they contain multiple
    pathologies, so adding every image would contaminate healthy-only training.
    """

    def __init__(self, aptos_dir, img_size=128):
        self.paths = _aptos_paths_for_grades(aptos_dir, lambda grades: grades == 0)
        assert len(self.paths) > 0, (
            "No APTOS Grade-0 images found. Check APTOS_DIR and confirm it "
            "contains train.csv/train_1.csv plus the matching train_images folder."
        )

        # Normalize to [-1, 1] to match the Decoder's Tanh output range
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


if __name__ == "__main__":
    import sys
    ds = HealthyRetinaDataset(sys.argv[1], img_size=128)
    print(f"Found {len(ds)} healthy images")
    sample = ds[0]
    print("Sample shape:", sample.shape, "range:", sample.min().item(), sample.max().item())


class AnomalyRetinaDataset(Dataset):
    """Loads APTOS images with DR (diagnosis > 0) to serve as the positive 
    (anomalous) class for AUC evaluation."""

    def __init__(self, aptos_dir, img_size=128):
        # diagnosis > 0 means DR is present
        self.paths = _aptos_paths_for_grades(aptos_dir, lambda grades: grades > 0)

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)
