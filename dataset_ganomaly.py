import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class HealthyRetinaDataset(Dataset):
    """Verified APTOS-2019 Grade-0 (No DR) images only.

    Raw ODIR folders are intentionally not scanned: they contain multiple
    pathologies, so adding every image would contaminate healthy-only training.
    """

    def __init__(self, aptos_dir, img_size=128):
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

        self.paths = [p for p in self.paths if os.path.exists(p)]
        assert len(self.paths) > 0, (
            "No APTOS Grade-0 images found. Check APTOS_DIR and confirm it "
            "contains train.csv plus train_images/."
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
        self.paths = []

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
            # diagnosis > 0 means DR is present
            dr_ids = df[df["diagnosis"] > 0]["id_code"].tolist()
            self.paths = [os.path.join(img_dir, f"{i}.png") for i in dr_ids]

        self.paths = [p for p in self.paths if os.path.exists(p)]

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
