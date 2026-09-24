import json
import os
import tempfile
from datetime import datetime

import config


STATE_FILE = os.path.join(
    config.ROOT,
    "logs",
    "dashboard_state.json"
)


def write_state(
    score=None,
    probability=None,
    prediction=None,
    sequence_current=0,
    sequence_total=8,
    alert=False,
    alert_reason=None
):
    """
    Save the latest CNN-LSTM prediction so the Flask
    dashboard can read it.
    """

    os.makedirs(
        os.path.dirname(STATE_FILE),
        exist_ok=True
    )

    state = {
        "score": score,
        "probability": probability,
        "prediction": prediction,
        "sequence_current": sequence_current,
        "sequence_total": sequence_total,
        "alert": alert,
        "alert_reason": alert_reason,
        "updated_at": datetime.now().strftime(
            "%H:%M:%S"
        )
    }

    # Write to a temporary file first.
    # This prevents Flask from reading a half-written file.

    fd, temp_path = tempfile.mkstemp(
        suffix=".json",
        dir=os.path.dirname(STATE_FILE)
    )

    try:

        with os.fdopen(
            fd,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                state,
                file,
                indent=2
            )

        os.replace(
            temp_path,
            STATE_FILE
        )

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


def read_state():
    """
    Read the latest prediction.
    """

    if not os.path.exists(STATE_FILE):

        return {
            "score": None,
            "probability": None,
            "prediction": None,
            "sequence_current": 0,
            "sequence_total": 8,
            "alert": False,
            "alert_reason": None,
            "updated_at": None
        }

    try:

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        return {
            "score": None,
            "probability": None,
            "prediction": None,
            "sequence_current": 0,
            "sequence_total": 8,
            "alert": False,
            "alert_reason": None,
            "updated_at": None
        }