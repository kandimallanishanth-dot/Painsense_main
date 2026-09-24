"""
PainSense - data preparation

Two jobs:

1. FRAME-LEVEL SET (real supervision):
   Parse SynPAIN filenames -> detect face -> crop face -> save into
   data/frames/pain/*.jpg and data/frames/no_pain/*.jpg

   SynPAIN filename pattern:
   <id>_<Pain|NoPain>_<man|woman>_<Young|Old>.jpg

2. SYNTHETIC SEQUENCE SET:
   Legacy/proxy sequence generation from static SynPAIN images.

IMPORTANT:
- If a face is NOT detected, crop_face() returns None.
- We NEVER use a random/center crop as a substitute for a missing face.
- This is especially important for real-time inference because a
  background/wall must never be sent to the CNN-LSTM as a face.

Run:
    python data_prep.py --step frames
    python data_prep.py --step sequences
    python data_prep.py --step all
"""

import os
import re
import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config


# ---------------------------------------------------------------------
# SynPAIN filename pattern
# ---------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(\d+)_(Pain|NoPain)_(man|woman)_(Young|Old)\.jpg$",
    re.IGNORECASE
)


# ---------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------

_face_cascade = None


def get_face_detector():
    """
    Load and cache the OpenCV Haar face detector.
    """

    global _face_cascade

    if _face_cascade is None:

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        _face_cascade = cv2.CascadeClassifier(cascade_path)

        if _face_cascade.empty():
            raise RuntimeError(
                "Could not load OpenCV Haar face detector.\n"
                f"Expected cascade file at:\n{cascade_path}"
            )

    return _face_cascade


# ---------------------------------------------------------------------
# Face cropping
# ---------------------------------------------------------------------

def crop_face(img_bgr, size=config.IMG_SIZE):
    """
    Detect the largest face and return a resized square crop.

    IMPORTANT:
        If no face is detected, return None.

    We intentionally DO NOT fall back to a center crop because a
    background image must never be passed to the CNN-LSTM as if it
    were a person's face.

    Returns:
        numpy array of shape (size, size, 3), or None
    """

    if img_bgr is None:
        return None

    if img_bgr.size == 0:
        return None

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detector = get_face_detector()

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # -------------------------------------------------------------
    # NO FACE
    # -------------------------------------------------------------

    if len(faces) == 0:
        return None

    # -------------------------------------------------------------
    # Select largest detected face
    # -------------------------------------------------------------

    x, y, fw, fh = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    # Add padding around the face.
    pad_x = int(0.20 * fw)
    pad_y = int(0.20 * fh)

    h, w = img_bgr.shape[:2]

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)

    x1 = min(w, x + fw + pad_x)
    y1 = min(h, y + fh + pad_y)

    crop = img_bgr[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    crop = cv2.resize(
        crop,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return crop


# ---------------------------------------------------------------------
# Parse SynPAIN directory
# ---------------------------------------------------------------------

def parse_synpain_dir(raw_dir):
    """
    Return:

        identity_id -> {
            'pain': path,
            'no_pain': path
        }
    """

    identities = defaultdict(dict)

    if not os.path.isdir(raw_dir):
        raise RuntimeError(
            f"SynPAIN directory does not exist:\n{raw_dir}"
        )

    for fname in os.listdir(raw_dir):

        m = FILENAME_RE.match(fname)

        if not m:
            continue

        ident, label, *_ = m.groups()

        key = (
            "pain"
            if label.lower() == "pain"
            else "no_pain"
        )

        identities[ident][key] = os.path.join(
            raw_dir,
            fname
        )

    return identities


# ---------------------------------------------------------------------
# Build frame-level dataset
# ---------------------------------------------------------------------

def build_frame_dataset(
    raw_dir=config.RAW_DATASET_DIR,
    out_dir=config.FRAMES_DIR
):
    """
    Create:

        data/frames/pain/
        data/frames/no_pain/

    Only frames where an actual face is detected are saved.
    """

    identities = parse_synpain_dir(raw_dir)

    if not identities:
        raise RuntimeError(
            f"No SynPAIN images matched the expected filename "
            f"pattern in:\n{raw_dir}\n\n"
            "Double-check that the SynPAIN Images folder is "
            "located there."
        )

    # Create output folders.

    for label in ("pain", "no_pain"):

        os.makedirs(
            os.path.join(out_dir, label),
            exist_ok=True
        )

    n_written = 0
    n_no_face = 0
    n_invalid = 0

    print()
    print("=" * 50)
    print("BUILDING REAL SYNPAIN FRAME DATASET")
    print("=" * 50)

    for ident, paths in tqdm(
        identities.items(),
        desc="Detecting/cropping faces"
    ):

        for label, path in paths.items():

            img = cv2.imread(path)

            if img is None:
                n_invalid += 1
                continue

            face = crop_face(img)

            # -----------------------------------------------------
            # No face detected
            # -----------------------------------------------------

            if face is None:
                n_no_face += 1
                continue

            out_path = os.path.join(
                out_dir,
                label,
                f"{ident}_{label}.jpg"
            )

            cv2.imwrite(
                out_path,
                face
            )

            n_written += 1

    print()
    print("=" * 50)
    print("FRAME DATASET COMPLETE")
    print("=" * 50)
    print(f"Faces written       : {n_written}")
    print(f"No face detected    : {n_no_face}")
    print(f"Invalid images      : {n_invalid}")
    print(f"Output directory    : {out_dir}")
    print()


# ---------------------------------------------------------------------
# Synthetic sequence generation
# ---------------------------------------------------------------------

def build_synthetic_sequences(
    raw_dir=config.RAW_DATASET_DIR,
    out_dir=config.SEQ_DIR,
    seq_len=config.SEQ_LEN
):
    """
    Legacy/proxy sequence generation.

    For every identity with BOTH a pain and no-pain image,
    alpha-blend neutral -> pain across seq_len frames.

    This is NOT real video data.

    It is retained only for compatibility with the original
    PainSense pipeline.
    """

    identities = parse_synpain_dir(raw_dir)

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    n_seq = 0
    n_no_face = 0

    manifest = []

    print()
    print("=" * 55)
    print("BUILDING SYNTHETIC PROXY SEQUENCES")
    print("=" * 55)

    for ident, paths in tqdm(
        identities.items(),
        desc="Building synthetic sequences"
    ):

        if "pain" not in paths or "no_pain" not in paths:
            continue

        neutral = cv2.imread(
            paths["no_pain"]
        )

        pain = cv2.imread(
            paths["pain"]
        )

        if neutral is None or pain is None:
            continue

        neutral = crop_face(neutral)

        pain = crop_face(pain)

        # ---------------------------------------------------------
        # Do not generate a sequence if either image has no face.
        # ---------------------------------------------------------

        if neutral is None or pain is None:

            n_no_face += 1
            continue

        neutral = neutral.astype(
            np.float32
        )

        pain = pain.astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Generate proxy temporal sequence.
        # ---------------------------------------------------------

        alphas = np.linspace(
            0.0,
            1.0,
            seq_len
        )

        frames = np.stack(
            [
                (
                    (1 - a) * neutral
                    + a * pain
                ).astype(np.uint8)

                for a in alphas
            ]
        )

        # 0 -> 10 pseudo-intensity target.

        targets = (
            alphas * config.PAIN_SCORE_MAX
        ).astype(
            np.float32
        )

        seq_path = os.path.join(
            out_dir,
            f"{ident}.npz"
        )

        np.savez_compressed(
            seq_path,
            frames=frames,
            targets=targets
        )

        manifest.append(
            seq_path
        )

        n_seq += 1

    # -------------------------------------------------------------
    # Save manifest
    # -------------------------------------------------------------

    manifest_path = os.path.join(
        out_dir,
        "manifest.txt"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(manifest)
        )

    print()
    print("=" * 55)
    print("SYNTHETIC SEQUENCE GENERATION COMPLETE")
    print("=" * 55)
    print(f"Sequences written  : {n_seq}")
    print(f"Skipped no-face    : {n_no_face}")
    print(f"Output directory   : {out_dir}")
    print()
    print(
        "NOTE: These sequences are alpha-blended proxy sequences, "
        "NOT real video."
    )
    print()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--step",
        choices=[
            "frames",
            "sequences",
            "all"
        ],
        default="all"
    )

    ap.add_argument(
        "--raw_dir",
        default=config.RAW_DATASET_DIR
    )

    args = ap.parse_args()

    # -------------------------------------------------------------
    # Frame dataset
    # -------------------------------------------------------------

    if args.step in (
        "frames",
        "all"
    ):

        build_frame_dataset(
            raw_dir=args.raw_dir
        )

    # -------------------------------------------------------------
    # Synthetic sequences
    # -------------------------------------------------------------

    if args.step in (
        "sequences",
        "all"
    ):

        build_synthetic_sequences(
            raw_dir=args.raw_dir
        )