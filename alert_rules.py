"""
PainSense - alert rules.

Two independent rules, matching the proposal / Review-2 deck:
  1. THRESHOLD rule: current score crosses ALERT_THRESHOLD.
  2. TREND rule: score has risen by >= TREND_RISE over the last TREND_WINDOW
     readings (sustained upward trend, even if still under threshold).
"""
import config


def check_threshold(score):
    return score >= config.ALERT_THRESHOLD


def check_trend(history_scores):
    """history_scores: list of recent scores, oldest -> newest."""
    if len(history_scores) < config.TREND_WINDOW:
        return False
    window = history_scores[-config.TREND_WINDOW:]
    return (window[-1] - window[0]) >= config.TREND_RISE


def evaluate(score, history_scores):
    """Returns (alert: bool, reason: str|None)"""
    if check_threshold(score):
        return True, f"Score {score:.1f} crossed threshold ({config.ALERT_THRESHOLD})"
    if check_trend(history_scores + [score]):
        return True, f"Sustained upward trend over last {config.TREND_WINDOW} readings"
    return False, None
