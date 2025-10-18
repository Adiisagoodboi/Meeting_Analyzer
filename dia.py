#!/usr/bin/env python3
"""
dia.py - minimal diarization pipeline wrapper.

Writes:
  results/<stem>_diar_merged.json
  results/<stem>_diarization.csv
  results/<stem>_diar_mapping.json
  <stem>.rttm (if available)
"""
from pathlib import Path
import os, sys, json, traceback

# optional pandas only used for CSV convenience
try:
    import pandas as pd
except Exception:
    pd = None

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

DIAR_PIPELINE_ID = os.getenv("DIAR_PIPELINE_ID", "pyannote/speaker-diarization-3.1")

def atomic_write(path: Path, txt: str):
    # simple atomic write using tmp file next to target
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)

def atomic_write_json(path: Path, obj):
    atomic_write(path, json.dumps(obj, indent=2, ensure_ascii=False))

def load_pipeline():
    hf = os.getenv("HF_TOKEN")
    if not hf:
        raise RuntimeError("HF_TOKEN environment variable not set.")
    device = os.getenv("WORKER_DEVICE", "cpu")
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        raise RuntimeError("pyannote.audio not available. Install pyannote-audio.") from e
    try:
        pipeline = Pipeline.from_pretrained(DIAR_PIPELINE_ID, use_auth_token=hf, device=device)
    except TypeError:
        pipeline = Pipeline.from_pretrained(DIAR_PIPELINE_ID, use_auth_token=hf)
    return pipeline

def merge_consecutive_same_speaker_list(recs):
    merged = []
    for r in recs:
        try:
            s = float(r["start"]); e = float(r["end"]); sp = r.get("speaker")
        except Exception:
            continue
        if not merged:
            merged.append({"start": s, "end": e, "speaker": sp})
            continue
        prev = merged[-1]
        # merge if same speaker
        if prev["speaker"] == sp:
            prev["end"] = max(prev["end"], e)
        else:
            merged.append({"start": s, "end": e, "speaker": sp})
    return merged

def diarize_file(pipeline, wav_path: Path):
    print(f"[diarize] processing {wav_path} ...")
    diar = pipeline(str(wav_path))

    raw_rows = []
    # two iteration strategies depending on pyannote object
    try:
        for turn, _, speaker in diar.itertracks(yield_label=True):
            raw_rows.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
    except Exception:
        try:
            for seg, _, label in diar.itersegments():
                raw_rows.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})
        except Exception:
            traceback.print_exc()
            raise RuntimeError("Unable to iterate diarization result.")

    if not raw_rows:
        raise RuntimeError("Diarization pipeline produced no turns.")

    stem = wav_path.stem
    raw_sorted = sorted(raw_rows, key=lambda r: float(r["start"]))
    merged_rows = merge_consecutive_same_speaker_list(raw_sorted)

    # create stable canonical speaker ids based on first-seen order
    raw_speakers = []
    for r in raw_sorted:
        sp = r.get("speaker")
        if sp not in raw_speakers:
            raw_speakers.append(sp)

    # Use canonical uppercase labels: SPEAKER_00, SPEAKER_01, ...
    mapping = {orig: f"SPEAKER_{i:02d}" for i, orig in enumerate(raw_speakers)}

    # map merged rows to canonical labels for the CSV/consumer
    canonical_mapped = []
    for r in merged_rows:
        s = round(float(r["start"]), 3)
        e = round(float(r["end"]), 3)
        sp = mapping.get(r.get("speaker"), r.get("speaker"))
        canonical_mapped.append({"start": s, "end": e, "speaker": sp})

    # write merged JSON (contains original speakers from pyannote)
    merged_json_path = OUT_DIR / f"{stem}_diar_merged.json"
    atomic_write_json(merged_json_path, merged_rows)

    # write canonical CSV (start,end,speaker)
    csv_path = OUT_DIR / f"{stem}_diarization.csv"
    lines = ["start,end,speaker"]
    for r in canonical_mapped:
        lines.append(f"{r['start']:.3f},{r['end']:.3f},{r['speaker']}")
    atomic_write(csv_path, "\n".join(lines))

    # write mapping JSON: original_label -> canonical SPEAKER_XX
    mapping_path = OUT_DIR / f"{stem}_diar_mapping.json"
    atomic_write_json(mapping_path, mapping)

    # try RTTM (best-effort)
    try:
        diar.to_rttm(OUT_DIR / f"{stem}.rttm")
    except Exception:
        pass

    print(f"[diarize] saved merged JSON: {merged_json_path.name}")
    print(f"[diarize] saved canonical CSV: {csv_path.name} (segments={len(canonical_mapped)})")
    print(f"[diarize] saved mapping JSON: {mapping_path.name}")
    return stem, csv_path, canonical_mapped

def resolve_audio_path():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    raise RuntimeError("No audio path provided. Call `python dia.py path/to/audio.wav`.")

def main():
    try:
        audio_path = resolve_audio_path()
    except Exception as e:
        print("[error]", e)
        return

    if not audio_path.exists():
        print(f"[error] audio file not found: {audio_path}")
        return

    try:
        pipeline = load_pipeline()
    except Exception as e:
        print("[error] failed to load diarization pipeline:", e)
        traceback.print_exc()
        return

    try:
        stem, out_csv, df = diarize_file(pipeline, audio_path)
    except Exception as e:
        print("[error] diarization failed:", e)
        traceback.print_exc()
        return

    summary = {"stem": stem, "diar_csv": str(out_csv), "segments_count": int(len(df))}
    print("\n=== DIARIZATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(json.dumps(summary))

if __name__ == "__main__":
    main()
