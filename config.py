"""
PainSense - central configuration.
Edit paths here once; every other script imports from this file.
"""
import os
import torch

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATASET_DIR = os.path.join(ROOT, "data", "raw", "SynPain", "Images")   # SynPAIN .jpg files go here
FRAMES_DIR = os.path.join(ROOT, "data", "frames")                          # ImageFolder: pain/ , no_pain/
SEQ_DIR = os.path.join(ROOT, "data", "sequences")                          # synthetic onset sequences (.npy)

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CNN_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_encoder.pt")
CNNLSTM_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_lstm.pt")

DB_PATH = os.path.join(ROOT, "logs", "painsense.db")

# ---------------- Image / model ----------------
IMG_SIZE = 128            # face crop resized to IMG_SIZE x IMG_SIZE
EMBED_DIM = 128           # CNN embedding dimension fed into LSTM
LSTM_HIDDEN = 64
SEQ_LEN = 8               # frames per temporal clip (both synthetic training + live inference buffer)

# ---------------- Training ----------------
BATCH_SIZE = 32
CNN_EPOCHS = 6
LSTM_EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Alerting ----------------
PAIN_SCORE_MAX = 10.0
ALERT_THRESHOLD = 6.0          # score >= this -> immediate alert
TREND_WINDOW = 5               # look at last N scores for trend rule
TREND_RISE = 2.0               # alert if score rose by >= this much over the window
CAPTURE_INTERVAL_SEC = 1.0     # seconds between frames sampled into the sequence buffer during live inference
