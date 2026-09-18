"""
PainSense - data preparation

Two jobs:
1. FRAME-LEVEL SET (real supervision):
   Parse SynPAIN filenames  ->  crop face  ->  save into
   data/frames/pain/*.jpg  and  data/frames/no_pain/*.jpg
   SynPAIN filename pattern: <id>_<Pain|NoPain>_<man|woman>_<Young|Old>.jpg

2. SYNTHETIC SEQUENCE SET (proxy temporal supervision, clearly a proxy — see README):
   SynPAIN gives 5,355 Neutral/Expressive PAIRS per identity, not video.
   We linearly alpha-blend each identity's NoPain (neutral) frame -> Pain frame
   across SEQ_LEN steps, producing a synthetic "onset" clip with a monotonically
   rising pseudo-intensity target (0 -> 10). This gives the LSTM head *something*
   legitimate to learn a temporal pattern from before real video data
   (UNBC-McMaster / BioVid) is available, as stated in the roadmap.

Run:
    python data_prep.py --step frames
    python data_prep.py --step sequences
    python data_prep.py --step all
"""
import os
import re
import argparse
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config

FILENAME_RE = re.compile(r"^(\d+)_(Pain|NoPain)_(man|woman)_(Young|Old)\.jpg$", re.IGNORECASE)

_face_cascade = None


def get_face_detector():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def crop_face(img_bgr, size=config.IMG_SIZE):
    """Detect the largest face and return a resized square crop.
    Falls back to a center crop if no face is detected (synthetic faces are
    front-facing and well lit, so this should rarely trigger)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = get_face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    h, w = img_bgr.shape[:2]
    if len(faces) > 0:
        # take the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.2 * fw)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
        crop = img_bgr[y0:y1, x0:x1]
    else:
        m = min(h, w)
        cy, cx = h // 2, w // 2
        crop = img_bgr[cy - m // 2: cy + m // 2, cx - m // 2: cx + m // 2]
    crop = cv2.resize(crop, (size, size))
    return crop


def parse_synpain_dir(raw_dir):
    """Return dict: identity_id -> {'pain': path, 'no_pain': path}"""
    identities = defaultdict(dict)
    for fname in os.listdir(raw_dir):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        ident, label, *_ = m.groups()
        key = "pain" if label.lower() == "pain" else "no_pain"
        identities[ident][key] = os.path.join(raw_dir, fname)
    return identities


def build_frame_dataset(raw_dir=config.RAW_DATASET_DIR, out_dir=config.FRAMES_DIR):
    identities = parse_synpain_dir(raw_dir)
    if not identities:
        raise RuntimeError(
            f"No SynPAIN images matched the expected filename pattern in {raw_dir}. "
            "Double-check you downloaded the 'Images' folder contents there."
        )

    for label in ("pain", "no_pain"):
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    n_written = 0
    for ident, paths in tqdm(identities.items(), desc="Cropping faces"):
        for label, path in paths.items():
            img = cv2.imread(path)
            if img is None:
                continue
            face = crop_face(img)
            out_path = os.path.join(out_dir, label, f"{ident}_{label}.jpg")
            cv2.imwrite(out_path, face)
            n_written += 1
    print(f"Wrote {n_written} cropped face frames to {out_dir}")


def build_synthetic_sequences(raw_dir=config.RAW_DATASET_DIR, out_dir=config.SEQ_DIR,
                               seq_len=config.SEQ_LEN):
    """For every identity with BOTH a pain and no_pain image, alpha-blend
    neutral -> pain across seq_len frames and save as a single .npy stack
    of shape (seq_len, IMG_SIZE, IMG_SIZE, 3) plus a target score file."""
    identities = parse_synpain_dir(raw_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_seq = 0
    manifest = []
    for ident, paths in tqdm(identities.items(), desc="Building synthetic sequences"):
        if "pain" not in paths or "no_pain" not in paths:
            continue
        neutral = cv2.imread(paths["no_pain"])
        pain = cv2.imread(paths["pain"])
        if neutral is None or pain is None:
            continue
        neutral = crop_face(neutral).astype(np.float32)
        pain = crop_face(pain).astype(np.float32)

        alphas = np.linspace(0.0, 1.0, seq_len)
        frames = np.stack([
            ((1 - a) * neutral + a * pain).astype(np.uint8) for a in alphas
        ])  # (seq_len, H, W, 3)
        targets = (alphas * config.PAIN_SCORE_MAX).astype(np.float32)  # 0 -> 10 ramp

        seq_path = os.path.join(out_dir, f"{ident}.npz")
        np.savez_compressed(seq_path, frames=frames, targets=targets)
        manifest.append(seq_path)
        n_seq += 1

    with open(os.path.join(out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest))
    print(f"Wrote {n_seq} synthetic onset sequences to {out_dir}")
    print("NOTE: these are alpha-blended proxy sequences, not real video. "
          "State this explicitly in the demo (see README).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["frames", "sequences", "all"], default="all")
    ap.add_argument("--raw_dir", default=config.RAW_DATASET_DIR)
    args = ap.parse_args()

    if args.step in ("frames", "all"):
        build_frame_dataset(raw_dir=args.raw_dir)
    if args.step in ("sequences", "all"):
        build_synthetic_sequences(raw_dir=args.raw_dir)
