"""
PainSense - training.

Stage 1 (real labels):
    python train.py --stage cnn --epochs 6

Stage 2 (synthetic onset sequences, initialized from stage-1 encoder):
    python train.py --stage lstm --epochs 10

Run on a GPU (e.g. Google Colab) if at all possible - CPU training on the
full SynPAIN set will be slow. For a same-day demo, --limit lets you train
on a subset quickly and still get a model that behaves sensibly on camera.
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from dataset import FrameDataset, SequenceDataset, split_dataset
from model import CNNClassifier, PainCNNLSTM


def maybe_limit(dataset, limit):
    if limit and limit < len(dataset):
        idx = torch.randperm(len(dataset))[:limit].tolist()
        return Subset(dataset, idx)
    return dataset


def train_cnn(epochs, limit=None):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    full_ds = FrameDataset(train=True)
    full_ds = maybe_limit(full_ds, limit)
    train_ds, val_ds = split_dataset(full_ds)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNNClassifier().to(config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        print(f"[CNN] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.CNN_CKPT)
            print(f"  saved checkpoint -> {config.CNN_CKPT}")

    print(f"Best CNN val accuracy: {best_acc:.4f}")


def train_lstm(epochs, limit=None):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    full_ds = SequenceDataset(train=True)
    full_ds = maybe_limit(full_ds, limit)
    train_ds, val_ds = split_dataset(full_ds)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = PainCNNLSTM().to(config.DEVICE)

    if os.path.exists(config.CNN_CKPT):
        cnn_state = torch.load(config.CNN_CKPT, map_location=config.DEVICE)
        model.load_cnn_weights(cnn_state)
        print(f"Initialized encoder from {config.CNN_CKPT}")
    else:
        print("WARNING: no CNN checkpoint found - training encoder from ImageNet weights only. "
              "Run `python train.py --stage cnn` first for a better result.")

    model.encoder.freeze_backbone(True)  # fine-tune only the LSTM + projection + head for speed
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.LR
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.DEVICE), y.to(config.DEVICE)
                preds = model(x)
                val_loss += loss_fn(preds, y).item() * x.size(0)
        val_loss /= max(len(val_ds), 1)
        print(f"[LSTM] epoch {epoch+1}/{epochs}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), config.CNNLSTM_CKPT)
            print(f"  saved checkpoint -> {config.CNNLSTM_CKPT}")

    print(f"Best LSTM val MSE: {best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cnn", "lstm"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of samples for a fast same-day training run")
    args = ap.parse_args()

    print(f"Using device: {config.DEVICE}")
    if args.stage == "cnn":
        train_cnn(epochs=args.epochs or config.CNN_EPOCHS, limit=args.limit)
    else:
        train_lstm(epochs=args.epochs or config.LSTM_EPOCHS, limit=args.limit)
