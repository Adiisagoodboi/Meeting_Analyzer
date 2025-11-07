#!/usr/bin/env python3
"""
worker_server.py — Worker server for Meeting Analyzer (FastAPI)

Behavior:
- Provides a simple save-only endpoint: POST /save_audio
    -> saves uploaded file into ./audio/ (no processing) and returns file URL.
- Keeps the processing pipeline (POST /process) and job system available for when you want to run diarization/ASR/visualize.
- Serves /audio/<file> and /results/<file> static files.

Run:
    uvicorn worker_server:app --host 0.0.0.0 --port 9000 --reload
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid, os, sys, json, threading, traceback, subprocess, tempfile, logging, shutil

# Optional local modules used by processing pipeline; presence not required if you only use /save_audio
try:
    import dia
    import asr
except Exception:
    dia = None
    asr = None

# ---- Configuration & Directories ----
ROOT = Path.cwd()
AUDIO_DIR = ROOT / "audio"
RESULTS_DIR = ROOT / "results"
AUDIO_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

WORKER_DEVICE = os.environ.get("WORKER_DEVICE", "cpu").strip()
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small").strip()
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")

DEFAULT_HOST = os.environ.get("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("PORT", "9000"))
RELOAD_ON_RUN = os.environ.get("RELOAD", "false").lower() in ("1", "true", "yes")

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("worker_server")

# ---- FastAPI app ----
app = FastAPI(title="Meeting Analyzer Worker (save-only / process)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount audio and results directories so frontend can fetch saved files
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# ---- Globals for processing pipeline (kept for /process) ----
jobs = {}
jobs_lock = threading.Lock()

DIAR_MODEL = None
ASR_MODEL = None
DIAR_LOCK = threading.Lock()
ASR_LOCK = threading.Lock()

# ---- Helpers ----
def atomic_write(path: Path, txt: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)

def _read_json_if_exists(path: Path):
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _write_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False))

def _public_file_url_for_audio(path: Path):
    # Return absolute if PUBLIC_BASE_URL set, else relative /audio/<name>
    rel = f"/audio/{path.name}"
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL + rel
    return rel

def _public_results_url(path: Path):
    rel = f"/results/{path.name}"
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL + rel
    return rel

def _build_result(stem: str):
    payload = {"stem": stem, "files": {}, "dialogue": None}
    dialogue_path = RESULTS_DIR / f"{stem}_dialogue_named.json"
    if dialogue_path.exists():
        payload["dialogue"] = _read_json_if_exists(dialogue_path)
        payload["files"]["dialogue_named"] = _public_results_url(dialogue_path)

    durations_path = RESULTS_DIR / f"{stem}_speaking_durations.json"
    if durations_path.exists():
        payload["files"]["durations"] = _public_results_url(durations_path)

    pie_png = RESULTS_DIR / f"{stem}_pie.png"
    pie_svg = RESULTS_DIR / f"{stem}_pie.svg"
    if pie_png.exists():
        payload["files"]["pie_png"] = _public_results_url(pie_png)
    elif pie_svg.exists():
        payload["files"]["pie_svg"] = _public_results_url(pie_svg)

    analysis = RESULTS_DIR / f"{stem}_analysis.json"
    if analysis.exists():
        payload["files"]["analysis"] = _public_results_url(analysis)

    return payload

# ---- Optional: model preload on startup (only if dia/asr available) ----
@app.on_event("startup")
def load_models_on_startup():
    global DIAR_MODEL, ASR_MODEL
    logger.info("Startup event: attempting to load models (if available).")
    if dia is None or asr is None:
        logger.info("dia/asr modules not available; skipping model preload.")
        return

    try:
        DIAR_MODEL = dia.load_model()
        logger.info("✅ Diarization model loaded.")
    except Exception:
        logger.exception("Failed to load diarization model; continue without it.")

    try:
        ASR_MODEL = asr.load_model(device=WORKER_DEVICE, model_name=WHISPER_MODEL, compute_type=WHISPER_COMPUTE_TYPE)
        logger.info("✅ ASR model loaded.")
    except Exception:
        logger.exception("Failed to load ASR model; continue without it.")

# ------------------------
# SAVE-ONLY endpoint
# ------------------------
@app.post("/save_audio")
async def save_audio(file: UploadFile = File(...)):
    """
    Save uploaded audio file directly into ./audio/ and return public URL.
    IMPORTANT: This is a *save-only* endpoint — it will NOT start diarization/ASR/visualization.
    """
    # ensure a filename and create a safe unique filename to avoid collisions
    orig_name = Path(file.filename).name if file.filename else "upload"
    # keep extension if present, otherwise default to .wav
    ext = Path(orig_name).suffix or ".wav"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = AUDIO_DIR / unique_name

    try:
        # write bytes atomically
        content = await file.read()
        dest_path.write_bytes(content)
    except Exception as e:
        logger.exception("Failed to save uploaded audio")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    public_url = _public_file_url_for_audio(dest_path)
    logger.info(f"Saved audio -> {dest_path} (public_url={public_url})")

    return JSONResponse(status_code=201, content={
        "saved": True,
        "filename": dest_path.name,
        "path": str(dest_path),
        "url": public_url
    })

# ------------------------
# Processing endpoints (unchanged) - POST /process starts pipeline
# ------------------------
# For convenience we keep the same processing pipeline code as before.
# If you don't need /process you can ignore the endpoints below.

def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def transcode_to_wav(input_path: Path, target_sr: int = 16000) -> Path:
    input_path = Path(input_path)
    wav_path = input_path.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", str(target_sr), "-sample_fmt", "s16", str(wav_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg transcode failed (rc={proc.returncode}): {proc.stderr.strip()}")
    return wav_path

def _process_job(job_id: str, audio_path: Path, original_filename: str):
    """
    Background worker that runs diarization -> asr -> visualize -> rag.
    This function will only be used by POST /process. If you only call /save_audio,
    this is never invoked.
    """
    try:
        stem = audio_path.stem
        with jobs_lock:
            jobs[job_id].update(status="running", progress=5, message="Starting processing")

        # Diarization
        with jobs_lock:
            jobs[job_id].update(progress=10, message="Running diarization")
        if dia is None:
            raise RuntimeError("dia module not available on server")
        with DIAR_LOCK:
            code, msg = dia.process_audio(DIAR_MODEL, str(audio_path), out_dir=str(RESULTS_DIR))
        with jobs_lock:
            jobs[job_id]["logs"].append({"step": "diarization", "code": code, "message": str(msg)})
        if code != 0:
            raise RuntimeError(f"diarization failed: {msg}")

        # ASR
        with jobs_lock:
            jobs[job_id].update(progress=60, message="Running ASR + speaker mapping")
        if asr is None:
            raise RuntimeError("asr module not available on server")
        with ASR_LOCK:
            code, msg = asr.process_audio(ASR_MODEL, str(audio_path), out_dir=str(RESULTS_DIR))
        with jobs_lock:
            jobs[job_id]["logs"].append({"step": "asr", "code": code, "message": str(msg)})
        if code != 0:
            raise RuntimeError(f"asr failed: {msg}")

        # durations, visualize, rag etc. (same pattern as earlier)
        # compute durations if missing
        with jobs_lock:
            jobs[job_id].update(progress=80, message="Computing speaking durations")
        dur_path = RESULTS_DIR / f"{stem}_speaking_durations.json"
        if not dur_path.exists():
            dialogue_path = RESULTS_DIR / f"{stem}_dialogue_named.json"
            if dialogue_path.exists():
                dlg = _read_json_if_exists(dialogue_path) or []
                durations = {}
                for seg in dlg:
                    sp = seg.get("speaker", "unknown") or "unknown"
                    s = float(seg.get("start", 0) or 0)
                    e = float(seg.get("end", s) or s)
                    durations[sp] = durations.get(sp, 0.0) + max(0.0, e - s)
                durations = {k: round(float(v), 3) for k, v in durations.items()}
                _write_json(dur_path, durations)
                with jobs_lock:
                    jobs[job_id]["logs"].append({"step": "durations_computed_from_dialogue", "count": len(durations)})

        # visualization
        with jobs_lock:
            jobs[job_id].update(progress=85, message="Building visualization")
        try:
            if dur_path.exists():
                cmd = [sys.executable, "visualize.py"]
                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
                with jobs_lock:
                    jobs[job_id]["logs"].append({
                        "step": "visualize_script_run",
                        "stdout": (proc.stdout or "")[:3000],
                        "stderr": (proc.stderr or "")[:3000],
                        "returncode": proc.returncode,
                    })
        except Exception as e:
            with jobs_lock:
                jobs[job_id]["logs"].append({"step": "visualize_error", "error": str(e)})

        # RAG (optional)
        with jobs_lock:
            jobs[job_id].update(progress=90, message="Running analysis")
        try:
            dialogue_json = RESULTS_DIR / f"{stem}_dialogue_named.json"
            rag_output = RESULTS_DIR / f"{stem}_analysis.json"
            if dialogue_json.exists():
                cmd = [sys.executable, "rag.py", "--stem", stem, "--results-dir", str(RESULTS_DIR)]
                proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
                with jobs_lock:
                    jobs[job_id]["logs"].append({
                        "step": "rag_script_run",
                        "returncode": proc.returncode,
                        "stdout": (proc.stdout or "")[:3000],
                        "stderr": (proc.stderr or "")[:3000]
                    })
                if rag_output.exists():
                    with jobs_lock:
                        jobs[job_id].setdefault("files", {})["analysis"] = _public_results_url(rag_output)
        except Exception:
            with jobs_lock:
                jobs[job_id]["logs"].append({"step": "rag_error"})

        # finalize
        with jobs_lock:
            jobs[job_id].update(progress=95, message="Finalizing results")
        result = _build_result(stem)
        with jobs_lock:
            jobs[job_id].update(status="done", progress=100, message="done", result=result)

    except Exception as e:
        tb = traceback.format_exc()
        with jobs_lock:
            jobs[job_id].update(status="error", progress=0, message=str(e), error=tb)
        logger.exception(f"[job {job_id}] failed:")
    finally:
        # cleanup uploaded file if still present
        try:
            tp = Path(audio_path)
            if tp.exists():
                try:
                    tp.unlink()
                except Exception:
                    pass
        except Exception:
            pass

async def _save_upload_and_start(file: UploadFile):
    """
    Save uploaded file to temp, transcode if needed, and start processing thread.
    This is used by /process (not used by /save_audio).
    """
    suffix = Path(file.filename).suffix or ".wav"
    tmp_dir = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(tmp_dir))
    tmp_path = Path(tmp.name)
    try:
        tmp.write(await file.read())
        tmp.flush()
        tmp.close()
    except Exception:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "message": "queued", "audio": str(tmp_path), "result": None, "error": None, "logs": []}

    # attempt transcode if non-wav and ffmpeg available (same logic as earlier implementations)
    audio_for_thread = tmp_path
    try:
        if tmp_path.suffix.lower() != ".wav":
            with jobs_lock:
                jobs[job_id]["logs"].append({"step": "transcode_check", "from": tmp_path.name})
            if not _ffmpeg_available():
                with jobs_lock:
                    jobs[job_id].update(status="error", progress=0, message="ffmpeg not found; cannot transcode non-wav uploads", error=None)
                    jobs[job_id]["logs"].append({"step": "transcode", "status": "ffmpeg_missing"})
                return {"job_id": job_id, "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"}
            try:
                wav_path = transcode_to_wav(tmp_path, target_sr=16000)
                with jobs_lock:
                    jobs[job_id]["logs"].append({"step": "transcode", "from": tmp_path.name, "to": wav_path.name})
                    jobs[job_id]["audio"] = str(wav_path)
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
                audio_for_thread = wav_path
            except Exception as e:
                tb = traceback.format_exc()
                with jobs_lock:
                    jobs[job_id].update(status="error", progress=0, message=f"transcode failed: {e}", error=tb)
                    jobs[job_id]["logs"].append({"step": "transcode_failed", "error": str(e)})
                return {"job_id": job_id, "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"}
    except Exception:
        tb = traceback.format_exc()
        with jobs_lock:
            jobs[job_id].update(status="error", progress=0, message="upload handling failed", error=tb)
            jobs[job_id]["logs"].append({"step": "upload_handling_failed", "traceback": tb})
        return {"job_id": job_id, "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"}

    # start background worker
    t = threading.Thread(target=_process_job, args=(job_id, audio_for_thread, file.filename), daemon=True)
    t.start()
    return {"job_id": job_id, "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    """
    Endpoint to start full processing pipeline. This is unchanged and will run diarization/ASR/etc.
    """
    resp = await _save_upload_and_start(file)
    return JSONResponse(status_code=202, content=resp)

# ------------------------
# Status / Result / Jobs / Health endpoints
# ------------------------
@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        return {"status": j["status"], "progress": j["progress"], "message": j["message"], "error": (j["error"][:300] if j.get("error") else None), "logs": j.get("logs", [])}

@app.get("/result/{job_id}")
def result(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job not found")
        if j["status"] != "done":
            return JSONResponse(status_code=409, content=j)
        return j["result"]

@app.get("/jobs")
def all_jobs():
    with jobs_lock:
        return {jid: {"status": j["status"], "progress": j["progress"], "message": j["message"]} for jid, j in jobs.items()}

@app.get("/health")
def health():
    return {"status": "ok", "device": WORKER_DEVICE, "model": WHISPER_MODEL, "models_loaded": (DIAR_MODEL is not None and ASR_MODEL is not None), "ffmpeg": shutil.which("ffmpeg") is not None}

# ---- Run directly convenience ----
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting worker_server on {DEFAULT_HOST}:{DEFAULT_PORT}, reload={RELOAD_ON_RUN}")
    uvicorn.run("worker_server:app", host=DEFAULT_HOST, port=DEFAULT_PORT, reload=RELOAD_ON_RUN)
