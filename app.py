#!/usr/bin/env python3
"""
app.py — Flask frontend for Meeting Analyzer.

- Serves static frontend from ./static
- Exposes /demo.png for frontend demo image convenience
- Forwards uploads to worker at WORKER_URL/process
- Proxies status/result endpoints to worker
- Serves results/ and audio/ files from local folders
"""
import os
import uuid
import logging
import json
from pathlib import Path
from typing import Optional, Any, Dict
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from werkzeug.utils import secure_filename
import requests

# ------------------ Config ------------------
ROOT = Path.cwd()
AUDIO_DIR = ROOT / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STATIC_DIR = ROOT / "static"

WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:9000")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
WORKER_TIMEOUT = int(os.getenv("WORKER_TIMEOUT_S", "30"))  # seconds for worker requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------ App ------------------
# static_folder serves /static/<file> automatically.
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app, origins=FRONTEND_ORIGIN)

# ------------------ Helpers ------------------
def _make_safe_dest(filename: str) -> Path:
    """Return a unique safe path in AUDIO_DIR preserving extension if present."""
    ext = Path(filename).suffix or ".wav"
    name = uuid.uuid4().hex
    return AUDIO_DIR / f"{name}{ext}"

def _safe_send_from_directory(directory: Path, filename: str):
    """Wrapper to send file if exists, otherwise 404."""
    fp = directory / filename
    if not fp.exists() or not fp.is_file():
        abort(404)
    return send_from_directory(str(directory), filename, as_attachment=False)

def _worker_get_json(url: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ---------- New helper: rewrite worker-relative result URLs to absolute WORKER_URL ----------
def _ensure_worker_urls(payload: Any) -> Any:
    """
    If payload is a dict containing 'files': { ... }, convert any file paths that look like
    '/results/...' or 'results/...' into absolute URLs pointing at WORKER_URL.
    Operates in-place where possible and returns the payload.
    """
    try:
        if not isinstance(payload, dict):
            return payload
        files = payload.get("files")
        if isinstance(files, dict):
            for k, v in list(files.items()):
                if isinstance(v, str):
                    if v.startswith("/results/") or v.startswith("results/"):
                        files[k] = WORKER_URL.rstrip("/") + (v if v.startswith("/") else "/" + v)
                    # leave absolute URLs unchanged
        # Also handle an often-used top-level 'result_url' key from fast-path
        if isinstance(payload.get("result_url"), str):
            rv = payload["result_url"]
            if rv.startswith("/results/") or rv.startswith("results/"):
                payload["result_url"] = WORKER_URL.rstrip("/") + (rv if rv.startswith("/") else "/" + rv)
        return payload
    except Exception:
        # be conservative: if rewriting fails, return original payload
        return payload

# ------------------ Routes ------------------
@app.route("/")
def index():
    """Serve the frontend index if available, else return a simple JSON."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return send_from_directory(str(STATIC_DIR), "index.html")
    return jsonify({"message": "Meeting Analyzer frontend is running."})

@app.route("/demo.png")
def demo_png():
    """Convenience route so <img src="demo.png"> works when opening root."""
    demo_path = STATIC_DIR / "demo.png"
    if demo_path.exists():
        return send_from_directory(str(STATIC_DIR), "demo.png")
    # return 204 to avoid noisy 404s in dev if no demo image
    return "", 204

@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return _safe_send_from_directory(AUDIO_DIR, filename)

@app.route("/results/<path:filename>")
def serve_results(filename):
    return _safe_send_from_directory(RESULTS_DIR, filename)

@app.route("/health")
def health():
    """Return frontend health and whether worker is reachable."""
    worker_ok = False
    try:
        r = requests.get(f"{WORKER_URL.rstrip('/')}/health", timeout=2)
        worker_ok = r.ok
    except Exception:
        worker_ok = False
    return jsonify({
        "status": "ok",
        "worker_url": WORKER_URL,
        "worker_reachable": worker_ok
    })

@app.route("/process", methods=["POST"])
def process_audio():
    """
    Accept upload and forward to worker's /process endpoint.
    If results already exist for the calculated local stem, return quickly.
    Otherwise forward file to the worker and return worker response.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "empty filename"}), 400

    # save local copy
    dest = _make_safe_dest(upload.filename)
    try:
        secure_name = secure_filename(upload.filename) or (uuid.uuid4().hex + ".wav")
        upload.save(dest)
        logging.info("Saved upload -> %s", dest)
    except Exception as e:
        logging.exception("Failed saving uploaded file")
        return jsonify({"error": "failed to save upload", "detail": str(e)}), 500

    stem = dest.stem
        # fast-path: if worker already produced dialogue/analysis result for this stem
    possible_dialog = RESULTS_DIR / f"{stem}_dialogue_named.json"
    possible_analysis = RESULTS_DIR / f"{stem}_analysis.json"

    if possible_dialog.exists() or possible_analysis.exists():
        # attempt to load dialogue if present
        dlg = None
        if possible_dialog.exists():
            try:
                dlg = json.loads(possible_dialog.read_text(encoding="utf-8"))
            except Exception:
                dlg = None

        payload = {
            "note": "existing results found",
            "files": {},
            "dialogue": dlg
        }

        if possible_dialog.exists():
            payload["files"]["dialogue_named"] = f"/results/{possible_dialog.name}"
        if possible_analysis.exists():
            payload["files"]["analysis"] = f"/results/{possible_analysis.name}"

        # If you want a single top-level result_url for quick access (keep for backward comp)
        # prefer analysis if present, otherwise dialogue
        if possible_analysis.exists():
            payload["result_url"] = f"/results/{possible_analysis.name}"
        elif possible_dialog.exists():
            payload["result_url"] = f"/results/{possible_dialog.name}"

        # rewrite any /results paths to absolute worker URLs so browser hits worker
        payload = _ensure_worker_urls(payload)
        return jsonify(payload)


    # forward to worker
    worker_process_url = f"{WORKER_URL.rstrip('/')}/process"
    try:
        with open(dest, "rb") as fh:
            files = {"file": (secure_name, fh, "application/octet-stream")}
            resp = requests.post(worker_process_url, files=files, timeout=WORKER_TIMEOUT)
    except requests.exceptions.RequestException as e:
        logging.exception("Failed to contact worker at %s", worker_process_url)
        return jsonify({"error": "failed to contact worker", "detail": str(e)}), 502

    # clean up local audio copy if you want; keep it for debugging by default
    # try:
    #     dest.unlink()
    # except Exception:
    #     pass

    # pass through worker response
    try:
        data = resp.json()
    except Exception:
        return jsonify({"error": "invalid response from worker", "raw": resp.text}), 502

    data["audio_saved"] = str(dest)
    data.setdefault("note", "forwarded to worker")

    # rewrite any /results paths to absolute worker URLs so the browser requests the worker host
    data = _ensure_worker_urls(data)

    logging.info("Forwarded worker response for upload (status=%s)", resp.status_code)
    return jsonify(data), resp.status_code

@app.route("/job_status/<job_id>")
def job_status(job_id: str):
    """Proxy to worker /status/{job_id}"""
    worker_status_url = f"{WORKER_URL.rstrip('/')}/status/{job_id}"
    try:
        resp = requests.get(worker_status_url, timeout=10)
    except requests.exceptions.RequestException as e:
        logging.exception("Failed to query worker status")
        return jsonify({"error": "failed to contact worker", "detail": str(e)}), 502

    try:
        payload = resp.json()
        # If worker returned a job object containing 'files', rewrite them to absolute worker URLs
        payload = _ensure_worker_urls(payload)
        return jsonify(payload), resp.status_code
    except Exception:
        return jsonify({"error": "invalid response from worker", "raw": resp.text}), 502

@app.route("/job_result/<job_id>")
def job_result(job_id: str):
    """Proxy to worker /result/{job_id}"""
    worker_result_url = f"{WORKER_URL.rstrip('/')}/result/{job_id}"
    try:
        resp = requests.get(worker_result_url, timeout=20)
    except requests.exceptions.RequestException as e:
        logging.exception("Failed to contact worker for result")
        return jsonify({"error": "failed to contact worker", "detail": str(e)}), 502

    try:
        payload = resp.json()
        payload = _ensure_worker_urls(payload)
        return jsonify(payload), resp.status_code
    except Exception:
        return jsonify({"error": "invalid response from worker", "raw": resp.text}), 502

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "7860"))
    logging.info("Starting frontend on 0.0.0.0:%s, worker=%s", port, WORKER_URL)
    app.run(host="0.0.0.0", port=port)
