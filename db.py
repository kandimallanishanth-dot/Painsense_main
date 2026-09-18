"""
PainSense - score logging.

Uses SQLite for the prototype demo (zero setup, one file, works offline).
The proposal names PostgreSQL for the fuller system; swapping is a matter of
replacing these functions with psycopg2 calls against the same schema, since
the rest of the codebase only calls log_score() / get_recent() / init_db().
"""
import sqlite3
import time
import os

import config


def get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pain_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            score REAL NOT NULL,
            alert_flag INTEGER NOT NULL,
            alert_reason TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_score(score, alert_flag=False, alert_reason=None, ts=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO pain_scores (timestamp, score, alert_flag, alert_reason) VALUES (?, ?, ?, ?)",
        (ts or time.time(), float(score), int(alert_flag), alert_reason),
    )
    conn.commit()
    conn.close()


def get_recent(n=50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT timestamp, score, alert_flag, alert_reason FROM pain_scores ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    conn.close()
    rows.reverse()  # oldest -> newest
    return [
        {"timestamp": r[0], "score": r[1], "alert": bool(r[2]), "reason": r[3]}
        for r in rows
    ]


def clear_db():
    conn = get_conn()
    conn.execute("DELETE FROM pain_scores")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {config.DB_PATH}")
