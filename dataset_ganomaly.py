import os
import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HealthyRetinaDataset(Dataset):
    """Combines APTOS-2019 Grade-0 (No DR) images with ODIR-5K normal
    images. GANomaly trains EXCLUSIVELY on this healthy pool — no
    DR-positive image should ever appear here, or the gatekeeper stops
    being a gatekeeper."""

    def __init__(self, aptos_dir, odir_dir, img_size=128):
        self.paths = []

        # --- APTOS: keep only diagnosis == 0 (No DR) ---
        csv_path = os.path.join(aptos_dir, "train.csv")
        img_dir = os.path.join(aptos_dir, "train_images")
        
        if not os.path.exists(csv_path):
            # Fallback: search recursively for train.csv in aptos_dir
            for root, dirs, files in os.walk(aptos_dir):
                if "train.csv" in files:
                    csv_path = os.path.join(root, "train.csv")
                    img_dir = os.path.join(root, "train_images")
                    break

        if os.path.exists(csv_path) and os.path.exists(img_dir):
            df = pd.read_csv(csv_path)
            grade0_ids = df[df["diagnosis"] == 0]["id_code"].tolist()
            self.paths += [os.path.join(img_dir, f"{i}.png") for i in grade0_ids]

        # --- ODIR: assumed pre-filtered to normal-only images ---
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            self.paths += glob.glob(os.path.join(odir_dir, "**", ext), recursive=True)

        self.paths = [p for p in self.paths if os.path.exists(p)]
        assert len(self.paths) > 0, (
            "No healthy images found. Check APTOS_DIR/ODIR_DIR in config.py "
            "match your actual Kaggle dataset folder names."
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
    ds = HealthyRetinaDataset(sys.argv[1], sys.argv[2], img_size=128)
    print(f"Found {len(ds)} healthy images")
    sample = ds[0]
    print("Sample shape:", sample.shape, "range:", sample.min().item(), sample.max().item())
