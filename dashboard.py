"""
PainSense - monitoring dashboard.

Run alongside realtime_infer.py:
    python dashboard.py
Then open http://localhost:5000

Polls the same SQLite DB that realtime_infer.py writes to, so the two
processes just need to run at the same time - no extra wiring required.
"""
from flask import Flask, jsonify, render_template_string

import config
import db

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>PainSense Dashboard</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; background: #eef2fb; margin: 0; padding: 24px; }
    h1 { color: #0b2f66; margin-bottom: 4px; }
    .sub { color: #4a5b7c; margin-top: 0; margin-bottom: 24px; }
    .cards { display: flex; gap: 16px; margin-bottom: 24px; }
    .card { background: white; border-radius: 10px; padding: 20px; flex: 1;
            box-shadow: 0 2px 8px rgba(11,47,102,0.08); text-align: center; }
    .card .label { color: #6b7a99; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .card .value { font-size: 32px; font-weight: bold; color: #0b2f66; margin-top: 6px; }
    .status-active { color: #1a9e46; }
    .status-idle { color: #9aa4b5; }
    .status-alert { color: #d1263b; }
    .status-standby { color: #9aa4b5; }
    .chart-box { background: white; border-radius: 10px; padding: 20px;
                 box-shadow: 0 2px 8px rgba(11,47,102,0.08); }
  </style>
</head>
<body>
  <h1>PainSense Dashboard</h1>
  <p class="sub">Live monitoring &middot; research prototype &middot; decision-support only</p>

  <div class="cards">
    <div class="card">
      <div class="label">Current Score</div>
      <div class="value" id="score">&mdash;</div>
    </div>
    <div class="card">
      <div class="label">Capture</div>
      <div class="value status-active" id="capture">ACTIVE</div>
    </div>
    <div class="card">
      <div class="label">Alert</div>
      <div class="value status-standby" id="alert">STANDBY</div>
    </div>
  </div>

  <div class="chart-box">
    <canvas id="chart" height="90"></canvas>
  </div>

  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [{
        label: 'Pain score over time',
        data: [],
        borderColor: '#1e5bd6',
        backgroundColor: 'rgba(30,91,214,0.1)',
        tension: 0.25,
        fill: true,
      }]},
      options: {
        scales: { y: { min: 0, max: 10, title: { display: true, text: 'Pain score' } } },
        plugins: { legend: { display: false } },
      }
    });

    async function poll() {
      try {
        const res = await fetch('/api/scores');
        const rows = await res.json();
        if (rows.length === 0) return;

        chart.data.labels = rows.map(r => new Date(r.timestamp * 1000).toLocaleTimeString());
        chart.data.datasets[0].data = rows.map(r => r.score);
        chart.update();

        const last = rows[rows.length - 1];
        document.getElementById('score').innerText = last.score.toFixed(1);

        const alertEl = document.getElementById('alert');
        if (last.alert) {
          alertEl.innerText = 'ALERT';
          alertEl.className = 'value status-alert';
        } else {
          alertEl.innerText = 'STANDBY';
          alertEl.className = 'value status-standby';
        }
      } catch (e) {
        console.error(e);
      }
    }
    setInterval(poll, 2000);
    poll();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/scores")
def api_scores():
    return jsonify(db.get_recent(50))


if __name__ == "__main__":
    db.init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
