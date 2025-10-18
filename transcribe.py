#!/usr/bin/env python3
"""
transcribe.py - produce results/<stem>_asr_segments.json
Supports faster-whisper (preferred) or openai-whisper as fallback.
"""
import sys, json, os
from pathlib import Path

OUT = Path("results"); OUT.mkdir(exist_ok=True)

# try faster-whisper first, then openai-whisper
FW_AVAILABLE = False
OW_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FW_AVAILABLE = True
except Exception:
    FW_AVAILABLE = False
try:
    import whisper
    OW_AVAILABLE = True
except Exception:
    OW_AVAILABLE = False

def usage():
    print("Usage: python transcribe.py path/to/audio.wav [model_name]")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()
    audio = Path(sys.argv[1]).resolve()
    if not audio.exists():
        print("Audio not found:", audio); sys.exit(1)

    model_name = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WHISPER_MODEL", "small") or "small").strip()
    print(f"[transcribe] chosen model: '{model_name}' (device={os.environ.get('WORKER_DEVICE','cpu')})")

    out_file = OUT / f"{audio.stem}_asr_segments.json"

    segments = []
    if FW_AVAILABLE:
        try:
            device = os.environ.get("WORKER_DEVICE", "cpu")
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
            model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print("[transcribe] Using faster-whisper:", model_name)
            res = model.transcribe(str(audio), beam_size=5, vad_filter=True)
            iterable = res[0] if isinstance(res, tuple) else res
            for s in iterable:
                segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
        except Exception as e:
            print("[transcribe] faster-whisper failed, falling back if possible:", repr(e))
            # try fallback to OW below if available
            if not OW_AVAILABLE:
                print("No other ASR backend available."); sys.exit(1)
    if not segments and OW_AVAILABLE:
        try:
            print("[transcribe] Using openai-whisper:", model_name)
            model = whisper.load_model(model_name)
            result = model.transcribe(str(audio), fp16=False)
            for s in result.get("segments", []):
                segments.append({"start": float(s["start"]), "end": float(s["end"]), "text": s["text"].strip()})
        except Exception as e:
            print("[transcribe] whisper failed:", repr(e)); sys.exit(1)

    if not segments:
        print("[transcribe] no segments produced"); sys.exit(1)

    # clean and normalize
    clean = []
    for seg in segments:
        try:
            st = float(seg["start"]); ed = float(seg["end"])
        except Exception:
            continue
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        clean.append({"start": round(st, 3), "end": round(ed, 3), "text": txt})

    out_file.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[transcribe] wrote {out_file} ({len(clean)} segments)")

if __name__ == "__main__":
    main()
