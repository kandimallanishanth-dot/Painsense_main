# PainSense — Prototype Backend

This is a working CNN-LSTM backend for the PainSense pipeline described in your
proposal and Review-2 deck:

`Camera → Face Detection → CNN (spatial) → LSTM (temporal) → Pain Score → Log + Alert → Dashboard`

## Important, read this before you train anything

**SynPAIN is not a video dataset.** It's 10,710 *static* synthetic images —
5,355 identities, each with one Neutral and one Pain (expressive) photo
(filenames like `1000012790_NoPain_man_Young.jpg`). There is no real temporal
sequence in it. So:

- The **CNN** is trained on **real, honest labels**: Pain vs No-Pain
  classification from SynPAIN. This part is legitimate supervised learning.
- The **LSTM** cannot be legitimately trained on real temporal dynamics from
  this dataset alone. `data_prep.py` builds **synthetic "onset" sequences**
  by alpha-blending each identity's Neutral → Pain image across 8 frames,
  with a rising 0→10 pseudo-intensity target. This gives the LSTM something
  real to learn a temporal *pattern* from (rising score as pain features
  intensify), but it is a **proxy, not real video**.

**Say this out loud in the demo.** Your own Review-2 slide 10 ("Responsible
Use") already commits to documenting limitations honestly — this is exactly
that limitation, and reviewers will respect you naming it more than they'd
respect you hiding it. Suggested line: *"For the prototype we trained the CNN
on real pain/no-pain labels from SynPAIN, and trained the LSTM's temporal
head on synthetically interpolated onset sequences as a proxy for video,
since SynPAIN itself only contains static image pairs — real video data
(UNBC-McMaster / BioVid) is the next roadmap step."*

## 1. Setup

```bash
cd painsense_backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Easiest way (no git-lfs needed):

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TaatiTeam/SynPAIN',
    repo_type='dataset',
    allow_patterns=['SynPain/Images/*'],
    local_dir='data/raw'
)
"
```

This should leave images at `data/raw/SynPain/Images/*.jpg`, which matches
`config.RAW_DATASET_DIR`. If your download lands somewhere else, either move
it or pass `--raw_dir` to `data_prep.py`.

> If you're short on time tonight, you don't need all 10,710 images. Ctrl-C
> the download after a few hundred files, or just delete extra files from
> the folder afterward — `data_prep.py` only looks at whatever's there.

## 3. Prepare data

```bash
python data_prep.py --step all
```

This does two things:
- Face-crops every image into `data/frames/{pain,no_pain}/` (for CNN training)
- Builds synthetic onset sequences into `data/sequences/*.npz` (for LSTM training)

## 4. Train

**Strongly recommended: do this on a GPU (Google Colab free tier is fine).**
CPU training works but is much slower — use `--limit` to cap dataset size for
a same-day run.

```bash
# Stage 1: CNN on real pain/no-pain labels
python train.py --stage cnn --epochs 6 --limit 2000

# Stage 2: CNN-LSTM on synthetic onset sequences (loads stage-1 encoder)
python train.py --stage lstm --epochs 10 --limit 600
```

Checkpoints land in `checkpoints/cnn_encoder.pt` and `checkpoints/cnn_lstm.pt`.

Sanity-check before the demo:

```bash
python evaluate.py --stage cnn     # accuracy + confusion matrix
python evaluate.py --stage lstm    # MAE on held-out synthetic sequences
```

## 5. Run the live demo

Two terminals, both from inside `painsense_backend/`:

```bash
# Terminal 1 — camera capture + inference + alert logic + logging
python realtime_infer.py

# Terminal 2 — dashboard
python dashboard.py
# then open http://localhost:5000
```

`realtime_infer.py` opens your webcam, shows a green "CAPTURE ACTIVE" dot
(your LED indicator, on-screen for the demo), samples a face crop roughly
once per second, and once it has 8 frames buffered, runs the CNN-LSTM and
shows a live pain score. If a score crosses the threshold (default 6/10) or
rises by ≥2 points over the last 5 readings, a red alert banner appears and
the dashboard flips to ALERT.

If training doesn't finish in time: `python realtime_infer.py --no-model`
still runs face detection and the buffering/UI pipeline end-to-end so you
can demonstrate the *architecture* live, and explain the score computation
separately with `evaluate.py` output. An honestly-labeled partial system
beats a claim of a system that doesn't actually run.

## 6. Swapping SQLite → PostgreSQL later

Everything logging-related goes through `db.py`'s three functions
(`init_db`, `log_score`, `get_recent`). To move to Postgres for the final
system, reimplement those three against a Postgres table with the same
columns (`timestamp`, `score`, `alert_flag`, `alert_reason`) — nothing else
in the codebase needs to change.

## File map

| File | Role |
|---|---|
| `config.py` | all paths, hyperparameters, alert thresholds |
| `data_prep.py` | face cropping, synthetic sequence construction |
| `dataset.py` | PyTorch `Dataset` classes |
| `model.py` | `CNNEncoder`, `CNNClassifier`, `PainCNNLSTM` |
| `train.py` | stage-1 CNN training, stage-2 LSTM training |
| `evaluate.py` | accuracy / confusion matrix / MAE |
| `alert_rules.py` | threshold + trend alert logic |
| `db.py` | SQLite score logging |
| `realtime_infer.py` | the actual live demo script |
| `dashboard.py` | Flask live dashboard (matches your deck's mockup) |

## What to say if asked "is this clinically validated?"

No — and your deck already says so (slide 10). This is a feasibility
prototype: real supervised CNN training on synthetic facial pain data, a
temporal head trained on a clearly-labeled synthetic proxy, and a working
end-to-end capture→inference→alert→dashboard loop. Clinical validation,
real video-based temporal training, and hardware integration are explicitly
future roadmap items, not claims being made today.
