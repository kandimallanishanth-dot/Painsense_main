"""
PainSense - PyTorch datasets.

FrameDataset      -> for CNN pretraining (binary pain / no_pain classification)
SequenceDataset   -> for CNN-LSTM training (synthetic onset sequences, regression target 0-10)
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_frame_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FrameDataset(Dataset):
    """Reads data/frames/{pain,no_pain}/*.jpg -> (image_tensor, label)"""

    def __init__(self, frames_dir=config.FRAMES_DIR, train=True):
        self.samples = []
        for label, cls in enumerate(["no_pain", "pain"]):  # 0 = no_pain, 1 = pain
            for path in glob.glob(os.path.join(frames_dir, cls, "*.jpg")):
                self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No frames found in {frames_dir}. Run data_prep.py --step frames first.")
        self.transform = train_frame_transform if train else frame_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


class SequenceDataset(Dataset):
    """Reads data/sequences/*.npz -> (seq_tensor [T,C,H,W], target_tensor [T])"""

    def __init__(self, seq_dir=config.SEQ_DIR, train=True):
        self.files = sorted(glob.glob(os.path.join(seq_dir, "*.npz")))
        if not self.files:
            raise RuntimeError(f"No sequences found in {seq_dir}. Run data_prep.py --step sequences first.")
        self.transform = train_frame_transform if train else frame_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        frames = data["frames"]     # (T, H, W, 3) uint8, BGR (from cv2)
        targets = data["targets"]   # (T,) float32, 0-10

        tensors = []
        for f in frames:
            f_rgb = f[:, :, ::-1]  # BGR -> RGB
            img = Image.fromarray(f_rgb)
            tensors.append(self.transform(img))
        seq_tensor = torch.stack(tensors)  # (T, C, H, W)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        return seq_tensor, target_tensor


def split_dataset(dataset, val_split=config.VAL_SPLIT, seed=42):
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
