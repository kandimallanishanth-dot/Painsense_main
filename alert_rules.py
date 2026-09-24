"""
PainSense - alert rules.

An alert is generated when:
    1. A valid face is present
    2. The estimated intensity score reaches ALERT_THRESHOLD
"""

import config


def evaluate(score, face_detected=True):
    """
    Return (alert, reason) for a 0-10 intensity score.
    """

    if not face_detected or score is None:
        return False, None

    score = max(
        float(config.PAIN_SCORE_MIN),
        min(float(config.PAIN_SCORE_MAX), float(score))
    )

    if score >= config.ALERT_THRESHOLD:
        return True, (
            f"Pain score {score:.1f} reached "
            f"threshold {config.ALERT_THRESHOLD:.1f}"
        )

    return False, None