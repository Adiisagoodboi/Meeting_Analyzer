#!/usr/bin/env python3
"""
asr.py

ASR module for Meeting Analyzer.

Provides:
- load_model(device, model_name, compute_type) -> model_handle
- process_audio(model_handle, audio_path, out_dir) -> (code:int, message:str)
- warm_up(model_handle)

Behavior:
- Expects diarization output at results/<stem>_diar_merged.json
- Produces results/<stem>_dialogue_named.json and results/<stem>_name_map.json

Notes:
- Name extraction is purely pattern-based (no spaCy / NER).
"""
from pathlib import Path
import os
import sys
import json
import re
import subprocess
import tempfile
import traceback
from typing import Optional, Tuple

# Backend availability
FW_AVAILABLE = False
OW_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    FW_AVAILABLE = True
except Exception:
    FW_AVAILABLE = False

try:
    import whisper as openai_whisper
    OW_AVAILABLE = True
except Exception:
    OW_AVAILABLE = False

OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ----------------------------
# ----------------------------
# --- Name-capture patterns ---
# ----------------------------
_NAME_STOPWORDS = {
    "in", "on", "at", "yes", "no", "ok", "okay", "thanks", "thank",
    "hello", "speaker", "everyone", "to", "one", "all"
}

_TITLE_PREFIX_RE = re.compile(r'^(mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+', re.I)

# Unicode-aware name token pieces:
NAME_LETTER = r'[^\W\d_]'                       # any unicode letter
NAME_FOLLOW = r"[^\W\d_'\-\.]*"                 # following letters
NAME_TOKEN = NAME_LETTER + NAME_FOLLOW

MAX_NAME_TOKENS = 3
NAME_GROUP = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,{MAX_NAME_TOKENS}}}"

QUOTES = r"['\"\u2018\u2019\u201C\u201D]*"

def _alts(phrases):
    return r"(?:{})".format("|".join(
        re.escape(p).replace(" ", r"\s+") for p in phrases
    ))

# Self-introduction triggers (speaker introducing themselves)
SECONDARY_INTROS = [
    "my name is", "this is", "i'm", "i am", "it's", "its",
    "call me", "myself"
]
SECONDARY_RE = _alts(SECONDARY_INTROS)

# Suffixes like: "Aditya here", "Rahul speaking"
SUFFIXES = [
    "here", "speaking", "present", "on the line", "available",
    "in the house", "that's me", "that's who i am",
    "at the call", "now", "is what people call me", "this side"
]
SUFFIX_RE = _alts(SUFFIXES)

# ----------------------------
# Patterns (ordered by reliability)
# ----------------------------

# Pattern C: Strict self-intro
PAT_C = re.compile(
    rf"""(?xi)
    ^\s*
    (?:{SECONDARY_RE})
    [\s,:\-]* 
    {QUOTES}
    (?P<name>{NAME_GROUP})
    {QUOTES}
    (?:\s+(?:{SUFFIX_RE}))?
    [\s\.\?!,]*$
    """
)

# Pattern B: "Aditya here" / "Rahul speaking"
PAT_B = re.compile(
    rf"""(?xi)
    (?P<name>{NAME_GROUP})
    \s+
    (?:{SUFFIX_RE})\b
    """
)

# Pattern A (light): standalone name ONLY if secondary intro exists implicitly
# (e.g., "Aditya" as a full line, but NOT "to", "one", "all")
PAT_A = re.compile(
    rf"""(?xi)
    ^\s*
    {QUOTES}
    (?P<name>{NAME_GROUP})
    {QUOTES}
    [\s\.\?!,]*$
    """
)

NAME_TOKEN_RE = re.compile(rf"^{NAME_TOKEN}$", re.UNICODE)



# ----------------------------
# --- Utility helpers -------
# ----------------------------
def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.replace('\u201C', '"').replace('\u201D', '"').replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _clean_name(raw: str) -> str:
    name = raw.strip(" \"'.,\n\r\t")
    name = re.sub(r'\s+', ' ', name).strip()
    name = _TITLE_PREFIX_RE.sub('', name)
    return name

def _is_valid_name_token(tok: str) -> bool:
    """
    Conservative token-level validation:
    - length >= 2
    - not in defensive stopwords
    - matches NAME_TOKEN_RE (unicode aware)
    """
    if not tok or not isinstance(tok, str):
        return False
    t = tok.strip()
    if len(t) < 2:
        return False
    if t.lower() in _NAME_STOPWORDS:
        return False
    # require at least one unicode letter
    if not re.search(r"[^\W\d_]", t, flags=re.UNICODE):
        return False
    if not NAME_TOKEN_RE.fullmatch(t):
        # last-resort looser check (letters allowed, with internal -/'/.)
        if not re.fullmatch(r"[^\W\d_][^\W\d_'\-\.]*", t, flags=re.UNICODE):
            return False
    return True

def _sanitize_text_keep_ascii_letters(text: str) -> str:
    if not text:
        return ""
    text = "".join(ch for ch in text if ch.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _has_acceptable_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))

# ----------------------------
# --- Name extraction (pure patterns) ---
# ----------------------------
def extract_first_name_from_text(text: str) -> Optional[str]:
    """
    Pattern-only conservative extraction:
      1) try PAT_A, PAT_C, PAT_B in that order
      2) heuristic: token immediately before "here|speaking|present|on the line|available|now"
    Returns first token title-cased, or None.
    """
    if not text:
        return None
    txt = _normalize_text(text)

    # 1) pattern-based extraction
    for pat in (PAT_A, PAT_C, PAT_B):
        m = pat.search(txt)
        if not m:
            continue
        raw = m.group("name")
        if not raw:
            continue
        cand = _clean_name(raw)
        tokens = cand.split()
        # require all tokens plausible; return first as first-name
        if tokens and all(_is_valid_name_token(t) for t in tokens):
            return tokens[0].title()

    # 2) heuristic: name token immediately before certain suffixes
    before_here_re = re.compile(
        r"\b([A-Za-z][A-Za-z\-']{1,})\b(?=\s+(?:here|speaking|present|on the line|available|now)\b)",
        re.IGNORECASE,
    )
    m2 = before_here_re.search(txt)
    if m2:
        tok = re.sub(r"[^A-Za-z\-']", "", m2.group(1))
        if _is_valid_name_token(tok) and tok.lower() not in _NAME_STOPWORDS:
            return tok.title()

    return None

# -----------------------
# ASR backend helpers
# -----------------------
def _build_fw_transcriber(model: "WhisperModel"):
    def transcribe_f(path: Path) -> str:
        try:
            res = model.transcribe(str(path), beam_size=5, vad_filter=(os.environ.get("ASR_VAD_FILTER", "1").strip().lower() not in ("0","false","no")), language="en", task="transcribe")
        except TypeError:
            try:
                res = model.transcribe(str(path), beam_size=5, vad_filter=(os.environ.get("ASR_VAD_FILTER", "1").strip().lower() not in ("0","false","no")))
            except TypeError:
                res = model.transcribe(str(path), beam_size=5)
        segs = res[0] if isinstance(res, tuple) and len(res) > 0 else res
        texts = []
        for s in segs:
            t = getattr(s, "text", None)
            if t is None and isinstance(s, dict):
                t = s.get("text", "")
            if t is None:
                t = str(s)
            texts.append(t.strip())
        return " ".join(texts).strip()
    return transcribe_f

def _build_ow_transcriber(model):
    def transcribe_f(path: Path) -> str:
        r = model.transcribe(str(path), fp16=False)
        segs = r.get("segments", []) if isinstance(r, dict) else []
        return " ".join(s.get("text", "").strip() for s in segs).strip()
    return transcribe_f

def _normalize_model_name(name: str) -> str:
    return (name or "small").strip()

def load_model(device: str = "cpu", model_name: str = "small", compute_type: str = "int8_float16"):
    model_name = _normalize_model_name(model_name)
    backend = None
    if FW_AVAILABLE:
        backend = "faster-whisper"
    elif OW_AVAILABLE:
        backend = "openai-whisper"
    else:
        raise RuntimeError("No ASR backend available. Install faster-whisper or openai-whisper.")

    if backend == "faster-whisper":
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        transcribe = _build_fw_transcriber(model)
        return {"backend": "faster-whisper", "model": model, "transcribe": transcribe}
    else:
        model = openai_whisper.load_model(model_name)
        transcribe = _build_ow_transcriber(model)
        return {"backend": "openai-whisper", "model": model, "transcribe": transcribe}

def warm_up(model_handle):
    # no-op by default (implementers may transcribe a tiny file if they want)
    return True

# -----------------------
# ffmpeg helpers
# -----------------------
def atomic_write(path: Path, txt: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt, encoding="utf-8")
    tmp.replace(path)

def require_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg to extract audio snippets.")

def extract_segment_with_ffmpeg(src_audio: Path, start: float, end: float, out_path: Path):
    start_s = f"{start:.3f}"
    duration = max(0.001, end - start)
    dur_s = f"{duration:.3f}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_audio),
        "-ss", start_s,
        "-t", dur_s,
        "-ar", "16000", "-ac", "1", "-vn",
        "-f", "wav", str(out_path)
    ]
    subprocess.run(cmd, check=True)

# -----------------------
# Diarization loader
# -----------------------
def load_diar_merged(stem: str):
    j = OUT / f"{stem}_diar_merged.json"
    if not j.exists():
        raise FileNotFoundError(f"{j} not found. Run dia.py first.")
    data = json.loads(j.read_text(encoding="utf-8"))
    return sorted([{"start": float(r["start"]), "end": float(r["end"]), "speaker": r.get("speaker")} for r in data], key=lambda x: x["start"])

# -----------------------
# Filtering thresholds
# -----------------------
MIN_SEGMENT_SEC = float(os.environ.get("MIN_SEGMENT_SEC", "0.15"))
VAD_FILTER = os.environ.get("ASR_VAD_FILTER", "1").strip().lower() not in ("0", "false", "no")
MERGE_MAX_GAP = float(os.environ.get("MERGE_MAX_GAP", "0.25"))

# -----------------------
# Main processing
# -----------------------
def process_audio(model_handle, audio_path: str, out_dir: Optional[str] = None) -> Tuple[int, str]:
    try:
        require_ffmpeg()
    except Exception as e:
        return (1, f"ffmpeg missing: {e}")

    try:
        outp = Path(out_dir) if out_dir else OUT
        outp.mkdir(parents=True, exist_ok=True)

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return (2, f"audio not found: {audio_path}")

        stem = audio_path.stem

        try:
            diar = load_diar_merged(stem)
        except FileNotFoundError as e:
            return (3, str(e))
        except Exception as e:
            return (4, f"failed to load diarization: {e}")

        if not diar:
            return (5, "diarization empty")

        transcribe = model_handle.get("transcribe") if isinstance(model_handle, dict) else None
        if transcribe is None:
            return (6, "invalid model handle (no transcribe function)")

        cluster_to_name = {}
        utterances = []

        for seg in diar:
            sp_cluster = seg.get("speaker")
            s = float(seg["start"]); e = float(seg["end"])
            duration = max(0.0, e - s)

            if duration < MIN_SEGMENT_SEC:
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                extract_segment_with_ffmpeg(audio_path, s, e, tmp_path)
                try:
                    text = transcribe(tmp_path) or ""
                except Exception as trans_ex:
                    print(f"[warn] asr transcribe failed for {s:.3f}-{e:.3f}: {trans_ex}", file=sys.stderr)
                    text = ""
                text = _sanitize_text_keep_ascii_letters(text)
            except Exception as exc:
                print(f"[warn] failed to extract/transcribe {s:.3f}-{e:.3f}: {exc}", file=sys.stderr)
                text = ""
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            if not text:
                continue
            if not _has_acceptable_letters(text):
                continue

            detected_name = extract_first_name_from_text(text)

            if detected_name:
                normalized = re.sub(r"[^A-Za-z\-']", "", detected_name).strip().title()
                if not _is_valid_name_token(normalized):
                    detected_name = None

            if detected_name and sp_cluster is not None:
                if sp_cluster not in cluster_to_name:
                    cluster_to_name[sp_cluster] = detected_name
                    # retroactively update previous utterances belonging to this cluster
                    for u in utterances:
                        if u.get("cluster") == sp_cluster:
                            u["speaker"] = detected_name

            speaker_label = cluster_to_name.get(sp_cluster, sp_cluster)

            utt = {
                "speaker": speaker_label,
                "cluster": sp_cluster,
                "start": round(s, 3),
                "end": round(e, 3),
                "text": text
            }
            utterances.append(utt)

        # Finalize output (replace cluster labels where mapped)
        final_out = []
        for u in utterances:
            cl = u.get("cluster")
            sp = cluster_to_name.get(cl, u.get("speaker"))
            final_out.append({
                "speaker": sp,
                "start": u["start"],
                "end": u["end"],
                "text": u["text"]
            })

        # Merge adjacent segments for the same speaker if gap is small
        if final_out:
            merged = [final_out[0].copy()]
            for cur in final_out[1:]:
                prev = merged[-1]
                gap = cur["start"] - prev["end"]
                if cur["speaker"] == prev["speaker"] and gap <= MERGE_MAX_GAP:
                    prev["end"] = cur["end"]
                    if cur["text"]:
                        prev["text"] = (prev["text"] + " " + cur["text"]).strip()
                else:
                    merged.append(cur.copy())
            final_out = merged

        out_path = outp / f"{stem}_dialogue_named.json"
        name_map_path = outp / f"{stem}_name_map.json"
        atomic_write(out_path, json.dumps(final_out, indent=2, ensure_ascii=False))
        atomic_write(name_map_path, json.dumps(cluster_to_name, indent=2, ensure_ascii=False))

        msg = f"wrote {out_path} ({len(final_out)} segments)."
        if cluster_to_name:
            msg += f" wrote name map: {name_map_path}"
        return (0, msg)

    except Exception as e:
        tb = traceback.format_exc()
        return (99, f"unexpected error: {e}\n{tb}")

# -----------------------
# CLI entrypoint
# -----------------------
def _cli_main():
    if len(sys.argv) < 2:
        print("Usage: python asr.py path/to/audio.wav [model_name]")
        sys.exit(1)
    audio_path = Path(sys.argv[1]).resolve()
    if not audio_path.exists():
        print("Audio not found:", audio_path); sys.exit(1)
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("WHISPER_MODEL", "small")
    device = os.environ.get("WORKER_DEVICE", "cpu")
    compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8_float16")
    print(f"[info] loading ASR model: {model_name} on {device} ({compute_type})")
    model = load_model(device=device, model_name=model_name, compute_type=compute_type)
    code, msg = process_audio(model, str(audio_path), out_dir=str(OUT))
    if code != 0:
        print("[error]", msg)
        sys.exit(code)
    else:
        print("[ok]", msg)

if __name__ == "__main__":
    _cli_main()
