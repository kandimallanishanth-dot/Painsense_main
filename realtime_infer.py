"""
PainSense - real-time inference (the actual demo script).

Camera -> face detection -> rolling buffer of SEQ_LEN frames -> CNN-LSTM
-> pain score -> log to DB -> alert rules -> on-screen overlay.

Run:
    python realtime_infer.py                 # uses webcam 0
    python realtime_infer.py --camera 1       # a different camera index
    python realtime_infer.py --no-model       # dry run without a trained checkpoint

Press 'q' to quit.
"""
import argparse
import time
from collections import deque

import cv2
import torch
from PIL import Image

import config
import db
import alert_rules
from data_prep import crop_face, get_face_detector
from dataset import frame_transform
from model import PainCNNLSTM


def load_model():
    model = PainCNNLSTM().to(config.DEVICE)
    try:
        state = torch.load(config.CNNLSTM_CKPT, map_location=config.DEVICE)
        model.load_state_dict(state)
        print(f"Loaded trained checkpoint: {config.CNNLSTM_CKPT}")
    except FileNotFoundError:
        print("No trained CNN-LSTM checkpoint found - running with an UNTRAINED model. "
              "Scores will not be meaningful; train first with train.py. "
              "Continuing so the pipeline can still be demonstrated end-to-end.")
    model.eval()
    return model


def draw_overlay(frame, score, capturing, alert, alert_reason, buffer_fill, buffer_target):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)

    # capture indicator (mirrors the physical LED on the Pi unit)
    dot_color = (0, 200, 0) if capturing else (100, 100, 100)
    cv2.circle(frame, (25, 25), 10, dot_color, -1)
    cv2.putText(frame, "CAPTURE ACTIVE" if capturing else "IDLE", (45, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    score_text = f"Pain score: {score:.1f}/10" if score is not None else \
        f"Buffering frames... {buffer_fill}/{buffer_target}"
    cv2.putText(frame, score_text, (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if alert:
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, f"ALERT: {alert_reason}", (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def run(camera_index=0, use_model=True):
    db.init_db()
    model = load_model() if use_model else None

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    frame_buffer = deque(maxlen=config.SEQ_LEN)
    score_history = []
    last_sample_t = 0.0
    last_score = None
    last_alert, last_reason = False, None

    print("PainSense live inference running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed, stopping.")
            break

        now = time.time()
        capturing = (now - last_sample_t) >= config.CAPTURE_INTERVAL_SEC
        if capturing:
            last_sample_t = now
            face = crop_face(frame)
            frame_buffer.append(face)

            if len(frame_buffer) == config.SEQ_LEN and model is not None:
                seq_tensors = torch.stack([
                    frame_transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    for f in frame_buffer
                ])
                seq_tensors = seq_tensors.unsqueeze(0).to(config.DEVICE)  # (1, T, C, H, W)
                with torch.no_grad():
                    scores = model(seq_tensors)  # (1, T)
                last_score = float(scores[0, -1].item())

                score_history.append(last_score)
                last_alert, last_reason = alert_rules.evaluate(last_score, score_history[:-1])
                db.log_score(last_score, last_alert, last_reason)

        frame = draw_overlay(frame, last_score, capturing, last_alert, last_reason,
                              len(frame_buffer), config.SEQ_LEN)
        cv2.imshow("PainSense - live inference", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-model", action="store_true", help="run the pipeline without inference (buffer/UI test)")
    args = ap.parse_args()
    run(camera_index=args.camera, use_model=not args.no_model)
