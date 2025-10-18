# app.py
"""
Flask server for meeting analyzer.

Endpoints:
  GET  /            -> serves static/index.html if present, otherwise simple landing page
  GET  /health      -> basic health check
  POST /process     -> upload audio file, forward to worker, return JSON payload
  GET  /audio/<fn>  -> serve uploaded audio
  GET  /results/<fn> -> serve generated pies/metrics

Usage:
  python app.py
Then open http://localhost:7860 in your frontend.
"""
import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import tempfile
import time
import gc

# ---------------------------
# Config
# ---------------------------
ROOT = Path.cwd()
AUDIO_DIR = ROOT / "audio"
RESULTS_DIR = ROOT / "results"
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Maximum upload size (bytes)
MAX_UPLOAD_MB = 200
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

# Worker URL (make sure worker_server is running on this)
WORKER_URL = "http://127.0.0.1:9000/process"

# ---------------------------
# App & logging
# ---------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    index_path = Path(app.static_folder) / "index.html"
    if index_path.exists():
        return send_from_directory(app.static_folder, "index.html")
    else:
        return (
            "<h2>Meeting Analyzer</h2>"
            "<p>No <code>static/index.html</code> found.</p>"
            "<p>Available endpoints:</p>"
            "<ul>"
            "<li><a href='/static/'>/static/ (static files folder)</a></li>"
            "<li><a href='/process'>/process (POST - audio upload)</a> - POST only</li>"
            "<li><a href='/audio/'>/audio/ (uploaded audio files)</a></li>"
            "<li><a href='/results/'>/results/ (generated result files)</a></li>"
            "<li><a href='/health'>/health</a></li>"
            "</ul>"
            "<p>Upload audio via POST to <code>/process</code>.</p>"
        ), 200

@app.route("/health")
def health():
    return jsonify({"status": "ok", "audio_dir": str(AUDIO_DIR), "results_dir": str(RESULTS_DIR)}), 200

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=False)

@app.route("/process", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    upload = request.files["file"]
    if upload.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(upload.filename).suffix or ".wav"
    tmpf = tempfile.NamedTemporaryFile(prefix="meeting_", suffix=suffix, delete=False)
    tmp_path = Path(tmpf.name)
    tmpf.close()

    try:
        upload.save(str(tmp_path))

        # forward to worker
        with open(tmp_path, "rb") as fh:
            r = requests.post(WORKER_URL, files={"file": fh}, timeout=25*60)

        if r.status_code != 200:
            return jsonify({
                "error": "Worker failed",
                "status_code": r.status_code,
                "text": r.text,
            }), 500

        return jsonify(r.json()), 200

    finally:
        # cleanup temp file
        try:
            attempts = 10
            for i in range(attempts):
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    break
                except PermissionError:
                    gc.collect()
                    time.sleep(0.2 + 0.1 * i)
                except Exception:
                    break
        except Exception:
            pass

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
