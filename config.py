"""
PainSense - central configuration.
Edit paths here once; every other script imports from this file.
"""
import os
import torch

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATASET_DIR = os.path.join(ROOT, "data", "raw", "SynPain", "Images")   # SynPAIN .jpg files go here
FRAMES_DIR = os.path.join(ROOT, "data", "frames")
SEQ_DIR = os.path.join(ROOT, "data", "sequences")

# Real SynPAIN video face frames
VIDEO_FACE_DIR = os.path.join(ROOT, "data", "videos", "face_frames")

# Generated 0-10 labels for those videos.
# Pain videos are neutral-to-pain transitions, so intensity follows
# frame progress from 0 to 10. NoPain videos stay at 0.
INTENSITY_DIR = os.path.join(ROOT, "data", "intensity")
INTENSITY_MANIFEST = os.path.join(INTENSITY_DIR, "manifest.csv")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CNN_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_encoder.pt")
CNNLSTM_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_lstm.pt")
INTENSITY_CKPT = os.path.join(CHECKPOINT_DIR, "pain_intensity.pt")

DB_PATH = os.path.join(ROOT, "logs", "painsense.db")

# ---------------- Image / model ----------------
IMG_SIZE = 224            # face crop resized to IMG_SIZE x IMG_SIZE
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

# ---------------- Intensity / Alerting ----------------
# 0 = start of a neutral-to-expression transition (or any NoPain video)
# 10 = end of a Pain video transition
# This is a generated target from SynPAIN morph progress, not PSPI or VAS.

PAIN_SCORE_MAX = 10.0
PAIN_SCORE_MIN = 0.0

# Alert when the estimated score reaches this value.
ALERT_THRESHOLD = 6.0

# Kept so older binary scripts still import.
PAIN_PROBABILITY_THRESHOLD = 0.50

# Number of consecutive valid face frames required
# before CNN-LSTM inference.
MIN_VALID_SEQUENCE = SEQ_LEN


CAPTURE_INTERVAL_SEC = 1.0     # seconds between frames sampled into the sequence buffer during live inference
