"""
PainSense - quick evaluation, useful for the "error analysis" part of the demo.

    python evaluate.py --stage cnn
    python evaluate.py --stage lstm
"""
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error

import config
from dataset import FrameDataset, SequenceDataset, split_dataset
from model import CNNClassifier, PainCNNLSTM


def eval_cnn():
    ds = FrameDataset(train=False)
    _, val_ds = split_dataset(ds)
    loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)

    model = CNNClassifier().to(config.DEVICE)
    model.load_state_dict(torch.load(config.CNN_CKPT, map_location=config.DEVICE))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            preds = model(x).argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))


def eval_lstm():
    ds = SequenceDataset(train=False)
    _, val_ds = split_dataset(ds)
    loader = DataLoader(val_ds, batch_size=8)

    model = PainCNNLSTM().to(config.DEVICE)
    model.load_state_dict(torch.load(config.CNNLSTM_CKPT, map_location=config.DEVICE))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(config.DEVICE)
            preds = model(x).cpu().numpy().ravel().tolist()
            all_pred.extend(preds)
            all_true.extend(y.numpy().ravel().tolist())

    print("MAE (0-10 scale):", mean_absolute_error(all_true, all_pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cnn", "lstm"], required=True)
    args = ap.parse_args()
    eval_cnn() if args.stage == "cnn" else eval_lstm()
