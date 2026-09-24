"""
PainSense - training using REAL SynPAIN video data.

Stage 1:
    Real SynPAIN face frames
    -> ResNet18 CNN
    -> NoPain / Pain classification

Stage 2:
    Real consecutive SynPAIN face frames
    -> CNN
    -> LSTM
    -> NoPain / Pain classification
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config

from dataset import (
    FrameDataset,
    SequenceDataset,
    IntensitySequenceDataset,
    export_intensity_manifest,
    split_dataset
)

from model import (
    CNNClassifier,
    PainCNNLSTM,
    PainIntensityLSTM
)


# =========================================================
# Utility
# =========================================================

def maybe_limit(dataset, limit):
    """
    Limit the number of samples AFTER the video-level
    train/validation split.

    This is important because split_dataset() needs access
    to the original dataset's video-group information.
    """

    if limit is not None and limit < len(dataset):

        idx = torch.randperm(
            len(dataset)
        )[:limit].tolist()

        return Subset(
            dataset,
            idx
        )

    return dataset


# =========================================================
# Stage 1 - CNN
# =========================================================

def train_cnn(
    epochs,
    limit=None
):

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    print()
    print("======================================")
    print("STAGE 1: CNN FRAME CLASSIFICATION")
    print("======================================")

    # -----------------------------------------------------
    # Load REAL SynPAIN face frames
    # -----------------------------------------------------

    full_ds = FrameDataset(
        train=True
    )

    print(
        f"Total real face frames: {len(full_ds)}"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Split by VIDEO first.
    #
    # This prevents frames from the same video from
    # appearing in both training and validation.
    # -----------------------------------------------------

    train_ds, val_ds = split_dataset(
        full_ds
    )

    # -----------------------------------------------------
    # Apply --limit AFTER the video-level split.
    #
    # This fixes the error:
    # "Dataset does not contain video group information."
    # -----------------------------------------------------

    train_ds = maybe_limit(
        train_ds,
        limit
    )

    print(
        f"Training samples used: {len(train_ds)}"
    )

    print(
        f"Validation samples: {len(val_ds)}"
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = CNNClassifier().to(
        config.DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR
    )

    # Binary classification:
    #
    # 0 = NoPain
    # 1 = Pain
    #
    # CNN outputs 2 logits.
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    # =====================================================
    # Training loop
    # =====================================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        for x, y in train_loader:

            x = x.to(
                config.DEVICE
            )

            y = y.to(
                config.DEVICE
            )

            optimizer.zero_grad()

            # CNN output:
            # (B, 2)
            logits = model(x)

            loss = loss_fn(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * x.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            total_correct += (
                predictions == y
            ).sum().item()

            total_samples += (
                y.size(0)
            )

        train_loss = (
            total_loss /
            max(len(train_ds), 1)
        )

        train_acc = (
            total_correct /
            max(total_samples, 1)
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    config.DEVICE
                )

                y = y.to(
                    config.DEVICE
                )

                logits = model(x)

                predictions = logits.argmax(
                    dim=1
                )

                correct += (
                    predictions == y
                ).sum().item()

                total += (
                    y.size(0)
                )

        val_acc = (
            correct /
            max(total, 1)
        )

        print(
            f"[CNN] "
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # -------------------------------------------------
        # Save best CNN checkpoint
        # -------------------------------------------------

        if val_acc >= best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                config.CNN_CKPT
            )

            print(
                f"  saved checkpoint -> "
                f"{config.CNN_CKPT}"
            )

    print()

    print(
        f"Best CNN validation accuracy: "
        f"{best_acc:.4f}"
    )


# =========================================================
# Stage 2 - CNN + LSTM
# =========================================================

def train_lstm(
    epochs,
    limit=None
):

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    print()
    print("======================================")
    print("STAGE 2: REAL VIDEO CNN-LSTM")
    print("======================================")

    # -----------------------------------------------------
    # Load REAL consecutive SynPAIN sequences
    # -----------------------------------------------------

    full_ds = SequenceDataset(
        train=True
    )

    print(
        f"Total real sequences: {len(full_ds)}"
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Split by VIDEO first.
    #
    # This prevents sequences from the same video from
    # appearing in both training and validation.
    # -----------------------------------------------------

    train_ds, val_ds = split_dataset(
        full_ds
    )

    # -----------------------------------------------------
    # Apply --limit AFTER video-level split.
    # -----------------------------------------------------

    train_ds = maybe_limit(
        train_ds,
        limit
    )

    print(
        f"Training sequences used: {len(train_ds)}"
    )

    print(
        f"Validation sequences: {len(val_ds)}"
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # CNN-LSTM model
    # -----------------------------------------------------

    model = PainCNNLSTM().to(
        config.DEVICE
    )

    # -----------------------------------------------------
    # Load Stage-1 CNN weights
    # -----------------------------------------------------

    if os.path.exists(
        config.CNN_CKPT
    ):

        cnn_state = torch.load(
            config.CNN_CKPT,
            map_location=config.DEVICE
        )

        model.load_cnn_weights(
            cnn_state
        )

        print(
            f"Initialized CNN encoder from "
            f"{config.CNN_CKPT}"
        )

    else:

        print(
            "WARNING: CNN checkpoint not found."
        )

        print(
            "The CNN encoder will use "
            "ImageNet pretrained weights."
        )

    # -----------------------------------------------------
    # Freeze ResNet backbone
    #
    # This reduces CPU training time.
    # The projection layer, LSTM and classification head
    # remain trainable.
    # -----------------------------------------------------

    model.encoder.freeze_backbone(
        True
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=config.LR
    )

    # -----------------------------------------------------
    # Binary classification loss
    #
    # 0 = NoPain
    # 1 = Pain
    # -----------------------------------------------------

    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # =====================================================
    # LSTM training loop
    # =====================================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        for x, y in train_loader:

            x = x.to(
                config.DEVICE
            )

            y = y.to(
                config.DEVICE
            )

            optimizer.zero_grad()

            # Model output:
            #
            # (B, T, 2)
            #
            # B = batch size
            # T = sequence length
            # 2 = NoPain/Pain
            logits = model(x)

            # -------------------------------------------------
            # Flatten model output
            #
            # (B, T, 2)
            #        ↓
            # (B*T, 2)
            # -------------------------------------------------

            logits_flat = logits.reshape(
                -1,
                2
            )

            # -------------------------------------------------
            # Flatten targets
            #
            # (B, T)
            #      ↓
            # (B*T,)
            # -------------------------------------------------

            targets_flat = y.reshape(
                -1
            ).long()

            # -------------------------------------------------
            # Calculate classification loss
            # -------------------------------------------------

            loss = loss_fn(
                logits_flat,
                targets_flat
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * x.size(0)
            )

            # -------------------------------------------------
            # Calculate training accuracy
            # -------------------------------------------------

            predictions = logits_flat.argmax(
                dim=1
            )

            total_correct += (
                predictions ==
                targets_flat
            ).sum().item()

            total_samples += (
                targets_flat.size(0)
            )

        train_loss = (
            total_loss /
            max(len(train_ds), 1)
        )

        train_acc = (
            total_correct /
            max(total_samples, 1)
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    config.DEVICE
                )

                y = y.to(
                    config.DEVICE
                )

                # (B,T,2)
                logits = model(x)

                # (B*T,2)
                logits_flat = logits.reshape(
                    -1,
                    2
                )

                # (B*T,)
                targets_flat = y.reshape(
                    -1
                ).long()

                loss = loss_fn(
                    logits_flat,
                    targets_flat
                )

                val_loss += (
                    loss.item() * x.size(0)
                )

                predictions = logits_flat.argmax(
                    dim=1
                )

                val_correct += (
                    predictions ==
                    targets_flat
                ).sum().item()

                val_total += (
                    targets_flat.size(0)
                )

        val_loss /= max(
            len(val_ds),
            1
        )

        val_acc = (
            val_correct /
            max(val_total, 1)
        )

        print(
            f"[LSTM] "
            f"epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # -------------------------------------------------
        # Save best CNN-LSTM checkpoint
        # -------------------------------------------------

        if val_acc >= best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                config.CNNLSTM_CKPT
            )

            print(
                f"  saved checkpoint -> "
                f"{config.CNNLSTM_CKPT}"
            )

    print()

    print(
        f"Best CNN-LSTM validation accuracy: "
        f"{best_val_acc:.4f}"
    )


# =========================================================
# Intensity regression on generated 0-10 targets
# =========================================================

def train_intensity(
    epochs,
    limit=None,
    sequences_per_video=6
):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print()
    print("======================================")
    print("INTENSITY: GENERATED 0-10 TARGETS")
    print("======================================")

    export_intensity_manifest()

    full_ds = IntensitySequenceDataset(
        sequences_per_video=sequences_per_video,
        train=True
    )

    train_ds, val_ds = split_dataset(full_ds)
    train_ds = maybe_limit(train_ds, limit)

    print(f"Training sequences used: {len(train_ds)}")
    print(f"Validation sequences: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = PainIntensityLSTM().to(config.DEVICE)

    # ImageNet features, not the binary pain checkpoint.
    # That checkpoint collapsed toward No Pain.
    print("Encoder: ImageNet ResNet18")

    # Freeze early ResNet layers and keep the last block trainable
    # so the features can follow expression strength.
    model.encoder.freeze_backbone(True)

    for name, param in model.encoder.backbone.named_parameters():
        if name.startswith("layer4"):
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )

    loss_fn = nn.MSELoss()
    best_mae = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / max(len(train_ds), 1)

        model.eval()
        abs_error = 0.0
        val_count = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)
                pred = model(x)
                abs_error += torch.abs(pred - y).sum().item()
                val_count += y.numel()

        val_mae = abs_error / max(val_count, 1)

        print(
            f"[INTENSITY] epoch {epoch + 1}/{epochs} "
            f"train_mse={train_loss:.4f} val_mae={val_mae:.4f}"
        )

        if val_mae <= best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.INTENSITY_CKPT)
            print(f"  saved checkpoint -> {config.INTENSITY_CKPT}")

    print()
    print(f"Best validation MAE: {best_mae:.4f}")

    model.load_state_dict(torch.load(
        config.INTENSITY_CKPT,
        map_location=config.DEVICE
    ))
    save_score_calibration(model, train_loader)


def save_score_calibration(model, loader):
    """
    Fit score = scale * raw + bias on training targets so the
    0-10 output uses the full generated range.
    """

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(config.DEVICE))[:, -1]
            target = y.to(config.DEVICE)[:, -1]
            preds.extend(pred.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())

    count = max(len(preds), 1)
    mean_pred = sum(preds) / count
    mean_target = sum(targets) / count
    variance = sum((pred - mean_pred) ** 2 for pred in preds)
    covariance = sum(
        (pred - mean_pred) * (target - mean_target)
        for pred, target in zip(preds, targets)
    )
    scale = covariance / variance if variance else 1.0
    bias = mean_target - scale * mean_pred

    path = os.path.join(
        config.CHECKPOINT_DIR,
        "pain_intensity_calibration.json"
    )

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"scale": scale, "bias": bias}, handle)

    print(
        f"Score calibration: scale={scale:.3f} bias={bias:.3f} -> {path}"
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=[
            "cnn",
            "lstm",
            "intensity"
        ],
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Limit training samples for a quick test run. "
            "The video-level train/validation split is "
            "performed before this limit."
        )
    )

    args = parser.parse_args()

    print(
        f"Using device: {config.DEVICE}"
    )

    if args.stage == "cnn":

        train_cnn(
            epochs=(
                args.epochs
                or config.CNN_EPOCHS
            ),
            limit=args.limit
        )

    elif args.stage == "intensity":

        train_intensity(
            epochs=(
                args.epochs
                or 3
            ),
            limit=args.limit
        )

    else:

        train_lstm(
            epochs=(
                args.epochs
                or config.LSTM_EPOCHS
            ),
            limit=args.limit
        )