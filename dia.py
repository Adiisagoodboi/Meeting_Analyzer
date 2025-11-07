#!/usr/bin/env python3
"""
dia.py — Diarization module for Meeting Analyzer (preloaded model style)

Exposes:
  - load_model() -> model_handle
  - process_audio(model_handle, audio_path, out_dir=None) -> (code:int, message:str)
  - warm_up(model_handle) -> bool

Behavior:
  - Loads a pyannote Pipeline from HF (requires HF_TOKEN env var).
  - Writes results/<stem>_diar_merged.json (list of segments: start,end,speaker).
  - Also writes results/<stem>_speaking_durations.json (per-speaker aggregated seconds).
  - Returns structured (code, message) so it can be called from worker_server.
"""
from pathlib import Path
import os, sys, json, traceback
from typing import Optional, Tuple

OUT = Path("results")
OUT.mkdir(exist_ok=True)

DIAR_PIPELINE_ID = os.getenv("DIAR_PIPELINE_ID", "pyannote/speaker-diarization-3.1")

def atomic_write(path: Path, txt: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)

def atomic_write_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False))

def merge_adjacent_same_speaker(segments):
    merged = []
    for seg in sorted(segments, key=lambda x: float(x["start"])):
        s = float(seg["start"])
        e = float(seg["end"])
        sp = seg.get("speaker")
        if not merged or merged[-1]["speaker"] != sp:
            merged.append({"start": s, "end": e, "speaker": sp})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], e)
    return merged

# -----------------------
# Model loader / runner
# -----------------------
def load_model() -> dict:
    """
    Load and return a model handle (dict). Raises on failure.
    Requires HF_TOKEN environment variable for auth.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set. Obtain a Hugging Face token and set HF_TOKEN.")
    device = os.getenv("WORKER_DEVICE", "cpu")

    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError("pyannote.audio not installed or import failed.") from e

    try:
        # Pipeline.from_pretrained signature may accept device param in some versions
        try:
            pipeline = Pipeline.from_pretrained(DIAR_PIPELINE_ID, use_auth_token=hf_token, device=device)
        except TypeError:
            pipeline = Pipeline.from_pretrained(DIAR_PIPELINE_ID, use_auth_token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline '{DIAR_PIPELINE_ID}': {e}") from e

    return {"pipeline_id": DIAR_PIPELINE_ID, "pipeline": pipeline, "device": device}

def warm_up(model_handle: dict) -> bool:
    """
    Optional warm-up. Running the pipeline on a tiny/short file could allocate memory.
    Here we no-op to avoid requiring an audio file at startup. Return True on success.
    """
    # Implementers can run a tiny sample if desired. Keep as no-op to be safe.
    return True

def diarize_with_pipeline(pipeline, wav_path: Path):
    """
    Run pipeline on wav_path and return list of rows: {'start', 'end', 'speaker'}
    """
    diar = pipeline(str(wav_path))
    rows = []
    # Two possible iteration APIs depending on pyannote version
    try:
        for turn, _, label in diar.itertracks(yield_label=True):
            rows.append({"start": float(turn.start), "end": float(turn.end), "speaker": label})
    except Exception:
        try:
            # older/later API fallback
            for seg, _, label in diar.itersegments():
                rows.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})
        except Exception as e:
            raise RuntimeError("Could not read diarization result: iteration failed.") from e

    if not rows:
        raise RuntimeError("No diarization output found (empty).")
    merged = merge_adjacent_same_speaker(rows)
    return merged

def _compute_and_write_durations(merged_segments, outp: Path, stem: str):
    """
    Compute per-speaker durations from merged_segments and write
    results/<stem>_speaking_durations.json (simple {speaker: seconds}).
    """
    durations = {}
    for seg in merged_segments:
        sp = seg.get("speaker", "unknown") or "unknown"
        s = float(seg.get("start", 0) or 0)
        e = float(seg.get("end", s) or s)
        dur = max(0.0, e - s)
        durations[sp] = durations.get(sp, 0.0) + dur
    # round durations to 3 decimals for neatness
    durations = {k: round(float(v), 3) for k, v in durations.items()}
    dur_path = outp / f"{stem}_speaking_durations.json"
    atomic_write_json(dur_path, durations)
    return dur_path, durations

def process_audio(model_handle: dict, audio_path: str, out_dir: Optional[str] = None) -> Tuple[int, str]:
    """
    Main processing function to be called by worker_server.
    Returns (code:int, message:str). 0 == success.
    """
    try:
        if out_dir:
            outp = Path(out_dir)
        else:
            outp = OUT
        outp.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return (2, f"audio not found: {audio_path}")

        pipeline = model_handle.get("pipeline") if isinstance(model_handle, dict) else None
        if pipeline is None:
            return (3, "invalid model handle (missing pipeline)")

        stem = audio_path.stem
        # Run diarization
        try:
            merged = diarize_with_pipeline(pipeline, audio_path)
        except Exception as e:
            tb = traceback.format_exc()
            return (4, f"diarization failed: {e}\n{tb}")

        out_path = outp / f"{stem}_diar_merged.json"
        atomic_write_json(out_path, merged)

        # NEW: compute and write per-speaker durations (so frontend/visualize can use them immediately)
        try:
            dur_path, durations = _compute_and_write_durations(merged, outp, stem)
            msg = f"wrote {out_path} (segments={len(merged)}), wrote durations {dur_path.name} (speakers={len(durations)})"
        except Exception as e:
            # don't fail entire process for duration write errors; include diagnostic in message
            msg = f"wrote {out_path} (segments={len(merged)}), durations_write_failed: {e}"

        return (0, msg)

    except Exception as e:
        tb = traceback.format_exc()
        return (99, f"unexpected error: {e}\n{tb}")

# -----------------------
# CLI compatibility
# -----------------------
def _cli_main():
    if len(sys.argv) < 2:
        print("Usage: python dia.py path/to/audio.wav")
        sys.exit(1)
    audio = Path(sys.argv[1]).resolve()
    if not audio.exists():
        print("Audio file not found:", audio); sys.exit(1)

    try:
        model = load_model()
    except Exception as e:
        print("[error] failed to load diarization model:", e)
        sys.exit(1)

    code, msg = process_audio(model, str(audio), out_dir=str(OUT))
    if code != 0:
        print("[error]", msg)
        sys.exit(code)
    else:
        print("[ok]", msg)
        # print summary similar to previous script
        stem = audio.stem
        merged_path = OUT / f"{stem}_diar_merged.json"
        try:
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            summary = {"file": str(audio), "segments": len(merged), "output": str(merged_path)}
            print("\n=== DIARIZATION SUMMARY ===")
            print(json.dumps(summary, indent=2))
        except Exception:
            pass

if __name__ == "__main__":
    _cli_main()
