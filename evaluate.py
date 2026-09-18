"""
PainSense - evaluation using REAL SynPAIN video data.

Usage:

    python evaluate.py --stage cnn
    python evaluate.py --stage lstm

CNN:
    Evaluates individual real face-frame classification.

LSTM:
    Evaluates real consecutive video-frame sequences.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

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


# =========================================================
# CNN Evaluation
# =========================================================

def eval_cnn():

    print()
    print("======================================")
    print("CNN EVALUATION")
    print("======================================")

    # Real SynPAIN face frames.
    ds = FrameDataset(
        train=False
    )

    # Use the same video-level split
    # used during training.
    _, val_ds = split_dataset(
        ds
    )

    loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CNNClassifier().to(
        config.DEVICE
    )

    model.load_state_dict(
        torch.load(
            config.CNN_CKPT,
            map_location=config.DEVICE
        )
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                config.DEVICE
            )

            logits = model(x)

            predictions = logits.argmax(
                dim=1
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

            y_true.extend(
                y.tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print()
    print(
        f"CNN Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NoPain",
                "Pain"
            ],
            zero_division=0
        )
    )


# =========================================================
# CNN-LSTM Evaluation
# =========================================================

def eval_lstm():

    print()
    print("======================================")
    print("REAL VIDEO CNN-LSTM EVALUATION")
    print("======================================")

    # Real consecutive video sequences.
    ds = SequenceDataset(
        train=False
    )

    # Same video-level validation split.
    _, val_ds = split_dataset(
        ds
    )

    loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    model = PainCNNLSTM().to(
        config.DEVICE
    )

    model.load_state_dict(
        torch.load(
            config.CNNLSTM_CKPT,
            map_location=config.DEVICE
        )
    )

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                config.DEVICE
            )

            logits = model(x)

            # logits:
            # (B, T, 2)
            #
            # Convert to predicted class.
            predictions = logits.argmax(
                dim=-1
            )

            all_pred.extend(
                predictions.cpu().reshape(-1).tolist()
            )

            all_true.extend(
                y.long().reshape(-1).tolist()
            )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    print()
    print(
        f"CNN-LSTM Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            all_true,
            all_pred,
            target_names=[
                "NoPain",
                "Pain"
            ],
            zero_division=0
        )
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
            "lstm"
        ],
        required=True
    )

    args = parser.parse_args()

    if args.stage == "cnn":

        eval_cnn()

    else:

        eval_lstm()