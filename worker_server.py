#!/usr/bin/env python3
# worker_server.py - Meeting Analyzer worker (improved)
"""
Run:
  python -m uvicorn worker_server:app --host 0.0.0.0 --port 9000 --reload
"""
from pathlib import Path
import os, sys, uuid, threading, time, tempfile, traceback, json, shutil, subprocess, re, unicodedata
from typing import Dict, Any
from collections import defaultdict
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---- paths ----
ROOT_PATH = Path.cwd()
RESULTS_DIR = ROOT_PATH / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---- config via env ----
WORKER_DEVICE = os.environ.get("WORKER_DEVICE", "cpu").strip()
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL", "small").strip()
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16").strip()

# ---- backend availability ----
FW_AVAILABLE = WHISPER_AVAILABLE = PYANNOTE_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FW_AVAILABLE = True
except Exception:
    pass
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    pass
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    PYANNOTE_AVAILABLE = True
except Exception:
    pass

# ---- app ----
app = FastAPI(title="Meeting Analyzer Worker")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

static_dir = Path(ROOT_PATH) / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "index.html not found"}, status_code=404)

@app.get("/health")
def health():
    return {"status": "ok"}

# ---- globals ----
asr_model = None
asr_backend = None
diar_pipeline = None

# ---- jobs ----
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

# ---------- helpers ----------
def run_cmd(cmd, timeout=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout or "", proc.stderr or ""

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def convert_to_wav(in_path: str, timeout: int = 300) -> str:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found")
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_tmp.close()
    out_path = out_tmp.name
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", in_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path]
    rc, out, err = run_cmd(cmd, timeout=timeout)
    if rc != 0:
        try: os.unlink(out_path)
        except Exception: pass
        raise RuntimeError(f"ffmpeg failed: {err}")
    return out_path

def _sanitize_stem(name: str):
    """Return a safe filename stem from original uploaded filename."""
    if not name:
        return None
    name = Path(name).stem  # drop extension
    name = unicodedata.normalize("NFKD", name)
    # remove characters that are not word/space/hyphen
    name = re.sub(r'[^\w\s-]', '', name, flags=re.UNICODE)
    name = re.sub(r'[\s\-]+', '_', name).strip('_')
    return name or None

@app.on_event("startup")
def load_models():
    global asr_model, asr_backend, diar_pipeline
    print("Loading models...")
    if FW_AVAILABLE:
        try:
            device = "cuda" if WORKER_DEVICE == "cuda" else "cpu"
            asr_model = WhisperModel(WHISPER_MODEL_NAME, device=device, compute_type=WHISPER_COMPUTE_TYPE)
            asr_backend = "faster_whisper"
            print("Loaded faster-whisper")
        except Exception as e:
            print("faster-whisper load failed:", repr(e))
    if asr_model is None and WHISPER_AVAILABLE:
        try:
            asr_model = whisper.load_model(WHISPER_MODEL_NAME)
            asr_backend = "whisper"
            print("Loaded openai/whisper")
        except Exception as e:
            print("whisper load failed:", repr(e))
    if PYANNOTE_AVAILABLE:
        try:
            diar_pipeline = PyannotePipeline.from_pretrained("pyannote/speaker-diarization")
            print("Loaded pyannote")
        except Exception as e:
            print("pyannote load failed:", repr(e))

# ---------- core processing ----------
def run_asr_on_file(wav_path: str):
    """
    Returns dict {"segments": [ {start,end,text}, ... ] }
    """
    if asr_backend == "faster_whisper":
        segments = []
        res = asr_model.transcribe(wav_path, beam_size=5, vad_filter=True)
        iterable = res[0] if isinstance(res, tuple) else res
        for seg in iterable:
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
        return {"segments": segments}
    elif asr_backend == "whisper":
        res = asr_model.transcribe(wav_path, verbose=False)
        segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]} for s in res["segments"]]
        return {"segments": segs}
    else:
        raise RuntimeError("ASR not available")

def run_diarization_on_file(wav_path: str):
    """
    Returns dict {"segments": [ {start,end,speaker}, ... ] } or None
    """
    if diar_pipeline is None:
        return None
    diar = diar_pipeline(wav_path)
    segs = []
    try:
        for t, _, sp in diar.itertracks(yield_label=True):
            segs.append({"start": float(t.start), "end": float(t.end), "speaker": sp})
    except Exception:
        try:
            for seg, _, label in diar.itersegments():
                segs.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})
        except Exception:
            traceback.print_exc()
            raise
    return {"segments": segs}

# Improved name extraction (safer normalization)
_NAME_TRASH = {"speaking", "here", "present", "on the line", "on the phone", "available", "now"}
NAME_PATTERNS = [
    r"\b(?:I(?:'m| am)|this is|it's|it is|hello,? I'm|hi,? I'm)\s+([A-Za-z][A-Za-z'`-]*(?:\s+[A-Za-z][A-Za-z'`-]*)?)\b",
    r"\b([A-Za-z][A-Za-z'`-]*(?:\s+[A-Za-z][A-Za-z'`-]*)?)\s+(?:here|speaking|on the line|present)\b",
    r"\bmy name is\s+([A-Za-z][A-Za-z'`-]*(?:\s+[A-Za-z][A-Za-z'`-]*)?)\b",
]
COMPILED_NAME_PATTERNS = [re.compile(p, flags=re.IGNORECASE) for p in NAME_PATTERNS]

def _first_name_from_candidate(cand: str):
    if not cand:
        return None
    cand = re.sub(r'[\.\?!,;:]+', '', cand).strip()
    parts = [p for p in re.split(r'\s+', cand) if p]
    if not parts:
        return None
    first = parts[0]
    if first.lower() in _NAME_TRASH or len(first) < 2:
        return None
    first = re.sub(r"[^A-Za-z\-']", "", first)
    if not first or first.lower() in ("ok","okay","yes","no"):
        return None
    return first.title()

def extract_name(text: str):
    if not text:
        return None
    for pat in COMPILED_NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r'\b(?:speaking|here|present|on the line|available|now)\b[\.!?,\s]*$', '', cand, flags=re.IGNORECASE).strip()
            first = _first_name_from_candidate(cand)
            if first:
                return first
    return None

def align_asr_diar(asr_segments, diar_segments):
    """Assign ASR segments to diar segments, build utterances, detect earliest names per speaker, merge nearby segments."""
    utterances = []
    speaker_candidates = {}  # spk -> list of (start, idx, name)

    def best_diar_for(seg):
        if not diar_segments:
            return None
        best, best_ov = None, 0.0
        for d in diar_segments:
            ov = max(0.0, min(seg["end"], d["end"]) - max(seg["start"], d["start"]))
            if ov > best_ov:
                best_ov, best = ov, d
        return best

    for idx, seg in enumerate(asr_segments):
        best = best_diar_for(seg) if diar_segments else None
        spk = best["speaker"] if best else "unknown"
        utterances.append({"speaker": spk, "start": seg["start"], "end": seg["end"], "text": seg["text"]})
        n = extract_name(seg["text"])
        if n:
            speaker_candidates.setdefault(spk, []).append((seg["start"], idx, n))

    # pick earliest candidate per speaker (if any)
    name_map = {}
    for spk, cand_list in speaker_candidates.items():
        cand_list.sort(key=lambda x: (x[0], x[1]))
        name_map[spk] = cand_list[0][2]

    # fallback defaults
    unique_spks = sorted({u["speaker"] for u in utterances})
    for idx, sp in enumerate(unique_spks):
        if sp not in name_map:
            name_map[sp] = f"Speaker_{idx+1}"

    # merge adjacent utterances of same speaker when gap is small (reduce UI fragmentation)
    merged_utts = []
    for u in utterances:
        if not merged_utts:
            merged_utts.append(u.copy()); continue
        prev = merged_utts[-1]
        gap = u["start"] - prev["end"]
        seglen = u["end"] - u["start"]
        if u["speaker"] == prev["speaker"] and (gap <= 0.25 or seglen < 0.4):
            prev["end"] = u["end"]
            prev["text"] = (prev["text"] + " " + u["text"]).strip()
        else:
            merged_utts.append(u.copy())
    utterances = merged_utts

    # durations aggregated by speaker id, then mapped to names
    dur_by_id = defaultdict(float)
    for u in utterances:
        dur_by_id[u["speaker"]] += max(0.0, (u["end"] - u["start"]))

    durations = {name_map[s]: round(dur_by_id[s], 2) for s in dur_by_id}

    return utterances, name_map, durations

# ---------- job runner ----------
def _process_job(job_id: str, uploaded_path: str, original_filename: str):
    try:
        with jobs_lock:
            jobs[job_id].update(progress=5, message="converting audio")
        wav_path = convert_to_wav(uploaded_path)
        temp_stem = Path(wav_path).stem
        user_stem = _sanitize_stem(original_filename)
        # prefer user_stem, attach short job id suffix to avoid accidental overwrite
        if user_stem:
            stem = f"{user_stem}_{job_id[:8]}"
        else:
            stem = temp_stem

        # run ASR
        with jobs_lock:
            jobs[job_id].update(progress=25, message="running ASR")
        asr_res = run_asr_on_file(wav_path)

        # write ASR segments (preferred stem) and keep temp-stem copy for debugging
        try:
            asr_file = RESULTS_DIR / f"{stem}_asr_segments.json"
            asr_file.write_text(json.dumps(asr_res["segments"], indent=2, ensure_ascii=False), encoding="utf-8")
            temp_asr = RESULTS_DIR / f"{temp_stem}_asr_segments.json"
            if not temp_asr.exists():
                temp_asr.write_text(json.dumps(asr_res["segments"], indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] wrote ASR segments -> {asr_file}")
        except Exception as ex:
            print("[warn] failed to write asr segments file:", repr(ex))

        # run diarization (optional)
        diar_res = None
        if diar_pipeline is not None:
            with jobs_lock:
                jobs[job_id].update(progress=55, message="running diarization")
            try:
                diar_res = run_diarization_on_file(wav_path)
                # persist diarization outputs using preferred stem
                if diar_res and "segments" in diar_res:
                    try:
                        merged_json_path = RESULTS_DIR / f"{stem}_diar_merged.json"
                        diar_csv_path = RESULTS_DIR / f"{stem}_diarization.csv"
                        mapping_json_path = RESULTS_DIR / f"{stem}_diar_mapping.json"

                        merged_json_path.write_text(json.dumps(diar_res["segments"], indent=2, ensure_ascii=False), encoding="utf-8")

                        try:
                            df = pd.DataFrame(diar_res["segments"])
                            cols = ["start", "end", "speaker"]
                            if all(c in df.columns for c in cols):
                                df = df[cols]
                            df.to_csv(diar_csv_path, index=False)
                        except Exception as ex_csv:
                            print("[warn] failed to write diar CSV:", repr(ex_csv))

                        mapping = {seg["speaker"]: seg["speaker"] for seg in diar_res["segments"]}
                        mapping_json_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
                        print(f"[INFO] wrote diar outputs -> {merged_json_path}, {diar_csv_path}, {mapping_json_path}")
                    except Exception as exwrite:
                        print("[warn] could not persist diarization outputs:", repr(exwrite))
            except Exception as ex:
                print("[warn] diarization failed:", repr(ex))
                diar_res = None

        # align ASR <-> diar and compute name_map & durations
        with jobs_lock:
            jobs[job_id].update(progress=75, message="mapping names and computing durations")
        utterances, name_map, durations = align_asr_diar(asr_res["segments"], diar_res["segments"] if diar_res else None)

        # normalize name_map values to first-name-only (defensive)
        clean_name_map = {}
        for k, v in (name_map or {}).items():
            if not v:
                continue
            v2 = re.sub(r'\b(?:speaking|here|present|on the line|available|now)\b[\.!?,\s]*$', '', str(v), flags=re.IGNORECASE).strip()
            first = re.split(r'\s+', v2.strip())[0] if v2 else v2
            first = re.sub(r"[^A-Za-z\-']", "", first or "")
            if first and len(first) > 1:
                clean_name_map[k] = first.title()
            else:
                # fallback: use v2 cleaned and title-cased
                if v2:
                    clean_name_map[k] = re.sub(r'[^A-Za-z\s\-]', '', v2).strip().title()
                else:
                    clean_name_map[k] = str(v).strip().title()
        name_map = clean_name_map

        # write outputs to results/ using preferred stem
        clean_csv = RESULTS_DIR / f"{stem}_clean_transcript.csv"
        try:
            pd.DataFrame([{
                "speaker_id": u["speaker"],
                "speaker_name": name_map.get(u["speaker"], u["speaker"]),
                "start": u["start"], "end": u["end"], "text": u["text"]
            } for u in utterances]).to_csv(clean_csv, index=False)
        except Exception as ex:
            print("[warn] failed to write clean transcript CSV:", repr(ex))

        (RESULTS_DIR / f"{stem}_name_map.json").write_text(json.dumps(name_map, indent=2, ensure_ascii=False), encoding="utf-8")
        (RESULTS_DIR / f"{stem}_speaking_durations.json").write_text(json.dumps(durations, indent=2, ensure_ascii=False), encoding="utf-8")

        # prepare dialogue payload (use cleaned CSV content)
        dialogue = []
        for u in utterances:
            dialogue.append({
                "start": float(u["start"]), "end": float(u["end"]),
                "text": u["text"], "speaker": name_map.get(u["speaker"], u["speaker"])
            })

        # optionally produce metrics CSV (compat)
        metrics_csv = RESULTS_DIR / f"{stem}_metrics.csv"
        try:
            with open(metrics_csv, "w", encoding="utf-8") as mf:
                mf.write("name,speaking_seconds\n")
                for n, sec in durations.items():
                    mf.write(f"{n},{sec}\n")
        except Exception:
            pass

        # build result structure
        result = {
            "audio_file": original_filename,
            "stages": {
                "asr": {"segments": asr_res["segments"], "segments_count": len(asr_res["segments"])},
                "diarization": {"segments": diar_res["segments"] if diar_res else [], "segments_count": len(diar_res["segments"]) if diar_res else 0},
                "analysis": {"durations": durations, "name_map": name_map, "clean_transcript_csv": str(clean_csv), "dialogue": dialogue}
            }
        }

        with jobs_lock:
            jobs[job_id].update(status="done", result=result, progress=100, message="done")

    except Exception as ex:
        tb = traceback.format_exc()
        with jobs_lock:
            jobs[job_id].update(status="error", message=str(ex), error=tb)
    finally:
        # cleanup uploaded file
        try:
            if os.path.exists(uploaded_path):
                os.unlink(uploaded_path)
        except Exception:
            pass

# ---------- API endpoints ----------
@app.post("/process")
async def process(file: UploadFile = File(...)):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix or ".wav")
    tmp.close()
    path = tmp.name
    with open(path, "wb") as f:
        f.write(await file.read())

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "running", "progress": 0, "message": "queued", "result": None, "error": None}

    thread = threading.Thread(target=_process_job, args=(job_id, path, file.filename), daemon=True)
    thread.start()

    return JSONResponse(status_code=202, content={"job_id": job_id, "status_url": f"/status/{job_id}", "result_url": f"/result/{job_id}"})

@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j: raise HTTPException(status_code=404, detail="job not found")
        return j

@app.get("/result/{job_id}")
def result(job_id: str):
    with jobs_lock:
        j = jobs.get(job_id)
        if not j: raise HTTPException(status_code=404, detail="job not found")
        if j["status"] != "done": return JSONResponse(status_code=409, content=j)
        return JSONResponse(j["result"])

@app.get("/jobs")
def list_jobs():
    with jobs_lock:
        return {jid: {"status": info["status"], "progress": info["progress"], "message": info["message"]} for jid, info in jobs.items()}
