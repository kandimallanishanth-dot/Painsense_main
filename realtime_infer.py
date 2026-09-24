"""
PainSense - real-time inference + web dashboard integration.

CURRENT MODEL OUTPUT:
    Intensity score from 0 to 10

The score estimates progress along the generated SynPAIN
neutral-to-pain transition. It is not a clinical PSPI or VAS score.

Pipeline:

    Camera
       ↓
    Face detection
       ↓
    Valid face?
       ↓
    8 consecutive valid face frames
       ↓
    CNN-LSTM
       ↓
    No Pain / Pain probability
       ↓
    Alert rules
       ↓
    Dashboard state
       ↓
    Flask dashboard

If no face is detected:
    - The frame is NOT added to the buffer
    - The existing buffer is cleared
    - CNN-LSTM is NOT executed
    - Prediction is cleared
    - Pain probability is cleared
    - Alert is cleared

Run:
    venv/Scripts/python.exe realtime_infer.py

Press 'q' to quit.
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import torch
from PIL import Image

import config
import db
import alert_rules
import dashboard_state

from data_prep import crop_face
from dataset import frame_transform
from model import PainIntensityLSTM


# =========================================================
# CONFIGURATION
# =========================================================

SCORE_THRESHOLD = config.ALERT_THRESHOLD


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

def load_calibration():
    path = os.path.join(
        config.CHECKPOINT_DIR,
        "pain_intensity_calibration.json"
    )

    if not os.path.exists(path):
        return 1.0, 0.0

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return (
        float(payload.get("scale", 1.0)),
        float(payload.get("bias", 0.0))
    )


def load_model():

    model = PainIntensityLSTM().to(config.DEVICE)

    try:

        state = torch.load(
            config.INTENSITY_CKPT,
            map_location=config.DEVICE
        )

        model.load_state_dict(state)

        print(
            f"Loaded trained checkpoint: "
            f"{config.INTENSITY_CKPT}"
        )

    except FileNotFoundError:

        print(
            "WARNING: No trained CNN-LSTM checkpoint found."
        )

        print(
            "The model will be untrained."
        )

    model.eval()

    return model


# =========================================================
# DRAW CAMERA OVERLAY
# =========================================================

def draw_overlay(
    frame,
    prediction,
    probability,
    face_detected,
    capturing,
    alert,
    alert_reason,
    buffer_fill,
    buffer_target
):

    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # TOP INFORMATION PANEL
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 125),
        (30, 30, 30),
        -1
    )

    # -----------------------------------------------------
    # CAPTURE INDICATOR
    # -----------------------------------------------------

    dot_color = (
        (0, 200, 0)
        if capturing and face_detected
        else (100, 100, 100)
    )

    cv2.circle(
        frame,
        (25, 25),
        10,
        dot_color,
        -1
    )

    cv2.putText(
        frame,
        "CAPTURE ACTIVE"
        if capturing and face_detected
        else "IDLE",
        (45, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # -----------------------------------------------------
    # NO FACE DETECTED
    # -----------------------------------------------------

    if not face_detected:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            "Waiting for a valid face...",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

    # -----------------------------------------------------
    # FACE DETECTED BUT BUFFERING
    # -----------------------------------------------------

    elif prediction is None:

        cv2.putText(
            frame,
            f"Buffering valid face frames: "
            f"{buffer_fill}/{buffer_target}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Waiting for CNN-LSTM inference...",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    else:

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        probability_text = (
            f"Pain score: "
            f"{probability:.1f} / {config.PAIN_SCORE_MAX:.0f}"
        )

        cv2.putText(
            frame,
            probability_text,
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # -----------------------------------------------------
    # ALERT
    # -----------------------------------------------------

    if alert:

        cv2.rectangle(
            frame,
            (0, h - 45),
            (w, h),
            (0, 0, 200),
            -1
        )

        cv2.putText(
            frame,
            f"ALERT: {alert_reason}",
            (15, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame


# =========================================================
# MAIN INFERENCE
# =========================================================

def run(
    camera_index=0,
    use_model=True
):

    # -----------------------------------------------------
    # INITIALIZE DATABASE
    # -----------------------------------------------------

    db.init_db()

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    model = (
        load_model()
        if use_model
        else None
    )

    score_scale, score_bias = load_calibration()

    # -----------------------------------------------------
    # OPEN WEBCAM
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        camera_index
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {camera_index}"
        )

    # =====================================================
    # FRAME BUFFER
    # =====================================================

    frame_buffer = deque(
        maxlen=config.SEQ_LEN
    )

    last_sample_t = 0.0

    # -----------------------------------------------------
    # Current prediction state
    # -----------------------------------------------------

    last_prediction = None
    last_probability = None

    last_alert = False
    last_reason = None

    face_detected = False

    # =====================================================
    # INITIAL DASHBOARD STATE
    # =====================================================

    dashboard_state.write_state(

        score=None,

        probability=None,

        prediction=None,

        sequence_current=0,

        sequence_total=config.SEQ_LEN,

        alert=False,

        alert_reason=None
    )

    print()
    print(
        "PainSense live inference running."
    )

    print(
        "Dashboard state integration active."
    )

    print(
        f"Pain score threshold: "
        f"{SCORE_THRESHOLD:.1f} / {config.PAIN_SCORE_MAX:.0f}"
    )

    print(
        "Press 'q' to quit."
    )

    print()

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    while True:

        ok, frame = cap.read()

        if not ok:

            print(
                "Camera read failed, stopping."
            )

            break

        now = time.time()

        capturing = (
            now - last_sample_t
        ) >= config.CAPTURE_INTERVAL_SEC

        # =================================================
        # SAMPLE FRAME
        # =================================================

        if capturing:

            last_sample_t = now

            # -------------------------------------------------
            # DETECT AND CROP FACE
            # -------------------------------------------------

            face = crop_face(frame)

            # =================================================
            # IMPORTANT:
            # NO FACE DETECTED
            # =================================================

            if face is None:

                face_detected = False

                # -------------------------------------------------
                # CLEAR BUFFER
                # -------------------------------------------------

                frame_buffer.clear()

                # -------------------------------------------------
                # CLEAR MODEL RESULT
                # -------------------------------------------------

                last_prediction = None
                last_probability = None

                # -------------------------------------------------
                # CLEAR ALERT
                # -------------------------------------------------

                last_alert = False
                last_reason = None

                # -------------------------------------------------
                # UPDATE DASHBOARD
                # -------------------------------------------------

                dashboard_state.write_state(

                    score=None,

                    probability=None,

                    prediction=None,

                    sequence_current=0,

                    sequence_total=config.SEQ_LEN,

                    alert=False,

                    alert_reason=None
                )

                # -------------------------------------------------
                # Terminal message
                # -------------------------------------------------

                print(
                    "[DASHBOARD] "
                    "No face detected -> "
                    "buffer cleared -> "
                    "no inference -> "
                    "no alert"
                )

            # =================================================
            # FACE DETECTED
            # =================================================

            else:

                face_detected = True

                # -------------------------------------------------
                # ADD ONLY VALID FACE TO BUFFER
                # -------------------------------------------------

                frame_buffer.append(face)

                # -------------------------------------------------
                # BEFORE MODEL INFERENCE
                # -------------------------------------------------

                if len(frame_buffer) < config.SEQ_LEN:

                    # We have a face but not enough frames yet.

                    last_prediction = None
                    last_probability = None

                    last_alert = False
                    last_reason = None

                    dashboard_state.write_state(

                        score=None,

                        probability=None,

                        prediction=None,

                        sequence_current=len(
                            frame_buffer
                        ),

                        sequence_total=config.SEQ_LEN,

                        alert=False,

                        alert_reason=None
                    )

                # =================================================
                # RUN CNN-LSTM
                # =================================================

                elif model is not None:

                    # -------------------------------------------------
                    # CONVERT FACE FRAMES TO TENSORS
                    # -------------------------------------------------

                    seq_tensors = torch.stack(
                        [
                            frame_transform(
                                Image.fromarray(
                                    cv2.cvtColor(
                                        f,
                                        cv2.COLOR_BGR2RGB
                                    )
                                )
                            )

                            for f in frame_buffer
                        ]
                    )

                    # -------------------------------------------------
                    # ADD BATCH DIMENSION
                    # -------------------------------------------------

                    seq_tensors = (
                        seq_tensors
                        .unsqueeze(0)
                        .to(config.DEVICE)
                    )

                    # =================================================
                    # MODEL INFERENCE
                    # =================================================

                    with torch.no_grad():

                        outputs = model(
                            seq_tensors
                        )

                    # =================================================
                    # LATEST FRAME SCORE, CLAMPED TO 0-10
                    # =================================================

                    if outputs.ndim != 2:

                        raise RuntimeError(
                            "Unexpected intensity output shape: "
                            f"{tuple(outputs.shape)}. "
                            "Expected [batch, sequence]."
                        )

                    raw_score = float(outputs[0, -1].item())
                    score = score_scale * raw_score + score_bias
                    score = max(
                        float(config.PAIN_SCORE_MIN),
                        min(float(config.PAIN_SCORE_MAX), score)
                    )

                    prediction = (
                        "PAIN"
                        if score >= SCORE_THRESHOLD
                        else "NO PAIN"
                    )

                    last_prediction = prediction
                    last_probability = score

                    # =================================================
                    # ALERT RULES
                    # =================================================

                    last_alert, last_reason = (
                        alert_rules.evaluate(
                            score,
                            face_detected=True
                        )
                    )

                    # =================================================
                    # DATABASE
                    # =================================================

                    try:

                        db.log_score(
                            score,
                            last_alert,
                            last_reason,
                            prediction=prediction
                        )

                    except Exception as db_error:

                        print(
                            "[DATABASE] "
                            f"Warning: {db_error}"
                        )

                    # =================================================
                    # SEND RESULT TO DASHBOARD
                    # =================================================

                    dashboard_state.write_state(

                        score=round(score, 2),

                        probability=score / float(config.PAIN_SCORE_MAX),

                        prediction=prediction,

                        sequence_current=config.SEQ_LEN,

                        sequence_total=config.SEQ_LEN,

                        alert=last_alert,

                        alert_reason=last_reason
                    )

                    # =================================================
                    # TERMINAL OUTPUT
                    # =================================================

                    print(
                        f"[DASHBOARD] "
                        f"Score={score:.1f} "
                        f"Prediction={prediction} "
                        f"Alert={last_alert}"
                    )

        # =================================================
        # DRAW CAMERA WINDOW
        # =================================================

        frame = draw_overlay(

            frame,

            last_prediction,

            last_probability,

            face_detected,

            capturing,

            last_alert,

            last_reason,

            len(frame_buffer),

            config.SEQ_LEN
        )

        # -------------------------------------------------
        # SHOW CAMERA
        # -------------------------------------------------

        cv2.imshow(
            "PainSense - live inference",
            frame
        )

        # -------------------------------------------------
        # QUIT WITH Q
        # -------------------------------------------------

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    # -----------------------------------------------------
    # Clear dashboard after stopping
    # -----------------------------------------------------

    dashboard_state.write_state(

        score=None,

        probability=None,

        prediction=None,

        sequence_current=0,

        sequence_total=config.SEQ_LEN,

        alert=False,

        alert_reason=None
    )

    print()
    print(
        "PainSense live inference stopped."
    )


# =========================================================
# PROGRAM ENTRY
# =========================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--camera",
        type=int,
        default=0
    )

    ap.add_argument(
        "--no-model",
        action="store_true",
        help=(
            "Run camera and dashboard "
            "without CNN-LSTM inference"
        )
    )

    args = ap.parse_args()

    print(
        f"Using device: "
        f"{config.DEVICE}"
    )

    run(

        camera_index=args.camera,

        use_model=not args.no_model

    )