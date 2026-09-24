from flask import Flask, render_template, jsonify

import dashboard_state
import db


app = Flask(__name__)


# =========================================================
# DASHBOARD HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# LIVE DASHBOARD STATE
# =========================================================

@app.route("/api/state")
def api_state():

    state = dashboard_state.read_state()

    return jsonify(
        state
    )


# =========================================================
# RECENT OBSERVATIONS / HISTORY
# =========================================================

@app.route("/api/history")
def api_history():

    history = db.get_recent(
        n=50
    )

    return jsonify(
        history
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    db.init_db()

    print(
        "PainSense web dashboard"
    )

    print(
        "History API: "
        "http://127.0.0.1:5000/api/history"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )