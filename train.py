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
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import config
from dataset import (
    FrameDataset,
    SequenceDataset,
    split_dataset
)
from model import (
    CNNClassifier,
    PainCNNLSTM
)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def maybe_limit(dataset, limit):

    if limit and limit < len(dataset):

        idx = torch.randperm(
            len(dataset)
        )[:limit].tolist()

        return Subset(
            dataset,
            idx
        )

    return dataset


# ---------------------------------------------------------
# Stage 1 - CNN
# ---------------------------------------------------------

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

    # Real SynPAIN face frames.
    full_ds = FrameDataset(
        train=True
    )

    print(
        f"Total real face frames: {len(full_ds)}"
    )

    # IMPORTANT:
    # For the normal run we don't limit the dataset.
    # If --limit is used, it is only for a quick test.
    full_ds = maybe_limit(
        full_ds,
        limit
    )

    train_ds, val_ds = split_dataset(
        full_ds
    )

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

    model = CNNClassifier().to(
        config.DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LR
    )

    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:

            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()

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

            total_samples += y.size(0)

        train_loss = (
            total_loss /
            max(len(train_ds), 1)
        )

        train_acc = (
            total_correct /
            max(total_samples, 1)
        )

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(config.DEVICE)
                y = y.to(config.DEVICE)

                logits = model(x)

                predictions = logits.argmax(
                    dim=1
                )

                correct += (
                    predictions == y
                ).sum().item()

                total += y.size(0)

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

        # Save best CNN.
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


# ---------------------------------------------------------
# Stage 2 - CNN + LSTM
# ---------------------------------------------------------

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

    # Build REAL consecutive sequences directly
    # from SynPAIN video face frames.
    full_ds = SequenceDataset(
        train=True
    )

    print(
        f"Total real sequences: {len(full_ds)}"
    )

    full_ds = maybe_limit(
        full_ds,
        limit
    )

    train_ds, val_ds = split_dataset(
        full_ds
    )

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

    # Freeze ResNet backbone for faster CPU training.
    model.encoder.freeze_backbone(
        True
    )

    # Train the projection, LSTM and classification head.
    optimizer = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=config.LR
    )

    # Binary classification loss.
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train_loader:

            x = x.to(
                config.DEVICE
            )

            y = y.to(
                config.DEVICE
            )

            optimizer.zero_grad()

            # logits:
            # (B, T, 2)
            logits = model(x)

            # Convert:
            #
            # (B, T, 2)
            #
            # into:
            #
            # (B*T, 2)
            #
            logits_flat = logits.reshape(
                -1,
                2
            )

            # Targets:
            #
            # (B, T)
            #
            # into:
            #
            # (B*T,)
            targets_flat = y.reshape(
                -1
            ).long()

            loss = loss_fn(
                logits_flat,
                targets_flat
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * x.size(0)
            )

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

                logits = model(x)

                logits_flat = logits.reshape(
                    -1,
                    2
                )

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

        # Save best CNN-LSTM.
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=[
            "cnn",
            "lstm"
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
            "Limit number of samples for "
            "a quick test run."
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

    else:

        train_lstm(
            epochs=(
                args.epochs
                or config.LSTM_EPOCHS
            ),
            limit=args.limit
        )