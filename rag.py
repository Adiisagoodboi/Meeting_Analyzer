#!/usr/bin/env python3
"""
rag.py — GPT-based Meeting Analyzer (results-folder-first, GPT-driven)

Behavior:
- Reads input from results/<stem>_dialogue_named.json
- Calls GPT to produce agenda, summary, per-speaker relevance, pros, cons, final_verdict, talk_time_seconds
- Does NOT compute or cap scores locally — GPT is fully responsible for scoring.
- Writes results/<stem>_analysis.json (always).
Exit codes:
 2 - input file not found
 3 - failed to load/validate input JSON
 4 - GPT/API analysis failed (script will still attempt to write best-effort file)
 5 - invalid analysis result object
 6 - failed to write output
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# OpenAI client (expects openai package with OpenAI class)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # We'll fail later with a clear message if unavailable

# ---------- config ----------
RESULTS_DIR = Path("results")
MODEL = "gpt-4o-mini"              # change if you prefer another model
MAX_TRANSCRIPT_CHARS = 200_000
# ----------------------------

def error(msg: str, code: int) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(code)

def find_dialogue_file(stem: Optional[str]) -> Tuple[Path, str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if stem:
        p = RESULTS_DIR / f"{stem}_dialogue_named.json"
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p, stem

    files = glob.glob(str(RESULTS_DIR / "*_dialogue_named.json"))
    if not files:
        raise FileNotFoundError(f"No dialogue_named.json files found in {RESULTS_DIR!s}")

    if len(files) == 1:
        p = Path(files[0])
        stem = p.stem.replace("_dialogue_named", "")
        return p, stem

    files_sorted = sorted(files, key=lambda f: Path(f).stat().st_mtime, reverse=True)
    chosen = Path(files_sorted[0])
    stem = chosen.stem.replace("_dialogue_named", "")
    print(f"[info] Multiple dialogue files found; selecting most recent: {chosen}")
    return chosen, stem

def load_transcript(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except Exception as e:
        raise RuntimeError(f"failed to load JSON: {e}")

    if not isinstance(obj, list):
        raise ValueError("transcript JSON must be an array of segments (list)")

    for i, seg in enumerate(obj):
        if not isinstance(seg, dict):
            raise ValueError(f"segment {i} is not an object")
        if "speaker" not in seg:
            seg["speaker"] = "unknown"
        if "start" not in seg:
            seg["start"] = 0.0
        if "end" not in seg:
            seg["end"] = seg.get("start", 0.0)
        if "text" not in seg and "transcript" in seg:
            seg["text"] = seg.get("transcript", "")
        if "text" not in seg:
            seg["text"] = ""

        # numeric coercion
        try:
            seg["start"] = float(seg.get("start") or 0.0)
        except Exception:
            seg["start"] = 0.0
        try:
            seg["end"] = float(seg.get("end") or seg["start"])
        except Exception:
            seg["end"] = seg["start"]

        # normalize speaker
        if not isinstance(seg["speaker"], str):
            seg["speaker"] = str(seg["speaker"] or "unknown")

    # safety limit
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) > MAX_TRANSCRIPT_CHARS:
        raise ValueError(
            f"transcript JSON too large ({len(s)} chars). Increase MAX_TRANSCRIPT_CHARS if necessary."
        )

    return obj

def compute_talk_times(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    times = defaultdict(float)
    for e in entries:
        try:
            sp = e.get("speaker") or "unknown"
            s = float(e.get("start") or 0.0)
            en = float(e.get("end") or s)
            if en >= s:
                times[sp] += en - s
        except Exception:
            continue
    return {k: round(float(v), 3) for k, v in times.items()}

def compute_word_counts(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = defaultdict(int)
    for e in entries:
        try:
            sp = e.get("speaker") or "unknown"
            text = e.get("text") or ""
            if not isinstance(text, str):
                text = str(text)
            wcount = len([w for w in text.split() if w.strip()])
            counts[sp] += wcount
        except Exception:
            continue
    return {k: int(v) for k, v in counts.items()}

def _coerce_float_0_10(val: Any) -> float:
    """Coerce GPT-provided score to float within [0,10], rounded to 1 decimal."""
    try:
        f = float(val)
    except Exception:
        f = 0.0
    if f != f:  # NaN
        f = 0.0
    return round(max(0.0, min(10.0, f)), 1)

def sanitize_and_trust_gpt(parsed_obj: Any,
                           entries: List[Dict[str, Any]],
                           talk_times: Dict[str, float],
                           word_counts: Dict[str, int]) -> Dict[str, Any]:
    """
    - Ensure fields exist and normalize speaker entries.
    - Trust GPT for relevance_score (coerced). If GPT doesn't provide a score, set 0.0.
    - Keep pros/cons/final_verdict/talk_time_seconds as provided by GPT when present.
    """
    parsed = parsed_obj if isinstance(parsed_obj, dict) else {"raw_output": parsed_obj}
    agenda = parsed.get("agenda") or ""
    summary = parsed.get("summary") or ""
    speakers = parsed.get("speakers") or []
    if not isinstance(speakers, list):
        speakers = []

    cleaned_speakers: List[Dict[str, Any]] = []

    for s in speakers:
        if not isinstance(s, dict):
            continue
        name = s.get("name") or s.get("speaker") or "unknown"

        # GPT-provided score (we trust GPT; coerce into 0-10)
        gscore_raw = s.get("relevance_score", s.get("score"))
        gscore = _coerce_float_0_10(gscore_raw) if gscore_raw is not None else 0.0

        pros = s.get("pros") or s.get("strengths") or s.get("feedback") or ""
        cons = s.get("cons") or s.get("weaknesses") or ""
        verdict = s.get("final_verdict") or s.get("verdict") or ""

        # talk time: prefer GPT value if present, else fall back to computed durations if available
        tt = s.get("talk_time_seconds")
        if tt is None:
            tt = talk_times.get(name, 0.0)
        try:
            tt = float(tt)
        except Exception:
            tt = float(talk_times.get(name, 0.0))
        tt = round(tt, 2)

        cleaned_speakers.append({
            "name": str(name),
            "relevance_score": float(gscore),
            "pros": str(pros or ""),
            "cons": str(cons or ""),
            "final_verdict": str(verdict or ""),
            "talk_time_seconds": tt
        })

    # If GPT returned no speakers, produce minimal speaker entries from transcript (scores = 0.0)
    if not cleaned_speakers and (talk_times or word_counts):
        all_names = sorted(set(talk_times.keys()) | set(word_counts.keys()))
        for nm in all_names:
            cleaned_speakers.append({
                "name": nm,
                "relevance_score": 0.0,
                "pros": "",
                "cons": "",
                "final_verdict": "No GPT analysis available.",
                "talk_time_seconds": round(float(talk_times.get(nm, 0.0)), 2)
            })

    cleaned = {
        "agenda": str(agenda),
        "summary": str(summary),
        "speakers": cleaned_speakers
    }
    if "raw_output" in parsed:
        cleaned["raw_output"] = parsed["raw_output"]
    return cleaned

def analyze_with_gpt(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Call GPT and return parsed JSON (or {"raw_output": text} if not JSON).
    The prompt instructs GPT to provide scores, pros, cons, final_verdict, and talk_time_seconds.
    """
    talk_times = compute_talk_times(entries)
    word_counts = compute_word_counts(entries)

    # Build transcript snippet (truncate if necessary)
    transcript_json = json.dumps(entries, ensure_ascii=False, indent=2)
    if len(transcript_json) > MAX_TRANSCRIPT_CHARS:
        transcript_json = transcript_json[:MAX_TRANSCRIPT_CHARS]

    prompt = f"""
You are an expert meeting analyst.

Below is a meeting transcript in JSON format (list of speaker entries with start, end, and text).

Your tasks:
1) Identify the main agenda of the meeting in one concise line.
2) Provide a short summary (2-4 sentences).
3) For each speaker, evaluate their contribution relative to the agenda and return:
   - "relevance_score": number (0.0–10.0) (one decimal, should look like proper analysis is done)
   - "pros": short positive points (1-2 sentences)
   - "cons": short improvement points, be firm (1-2 sentences)
   - "final_verdict": teacher-style verdict , if score is low be firm(1 sentence)
   - "talk_time_seconds": numeric seconds
4) Return ONLY a JSON object with fields:
{{
  "agenda": "string",
  "summary": "string",
  "speakers": [
    {{
      "name": "string",
      "relevance_score": number,
      "pros": "string",
      "cons": "string",
      "final_verdict": "string",
      "talk_time_seconds": number
    }}
  ]
}}

If you cannot produce valid JSON, return an object with a single key "raw_output" whose value is your analysis text.

Meeting transcript JSON:
{transcript_json}
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set. Set the env var before running rag.py")

    if OpenAI is None:
        raise RuntimeError("openai package not available (failed to import OpenAI)")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a professional meeting summarizer and evaluator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2500
    )

    text = ""
    try:
        text = resp.choices[0].message.content.strip()
    except Exception:
        text = str(resp)[:MAX_TRANSCRIPT_CHARS]

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {"raw_output": text}

    # Trust GPT output (no local score capping)
    return sanitize_and_trust_gpt(parsed, entries, talk_times, word_counts)

def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="RAG: read from results/<stem>_dialogue_named.json and write <stem>_analysis.json"
    )
    parser.add_argument("--stem", "-s", help="stem name (so rag reads results/<stem>_dialogue_named.json)", required=False)
    parser.add_argument("--results-dir", "-r", default=str(RESULTS_DIR), help="results directory (default: ./results)")
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        input_path, stem = find_dialogue_file(args.stem)
    except FileNotFoundError as e:
        error(str(e), code=2)

    output_path = RESULTS_DIR / f"{stem}_analysis.json"

    print(f"[info] Using input: {input_path}")
    print(f"[info] Will write output: {output_path}")

    try:
        entries = load_transcript(input_path)
    except Exception as e:
        error(f"Failed to load input JSON: {e}", code=3)

    write_err_code = 0

    # Call GPT (user requested GPT does it all)
    try:
        result = analyze_with_gpt(entries)
    except Exception as e:
        # If GPT fails unexpectedly, create a minimal fallback analysis WITHOUT computing/capping scores
        print(f"[error] GPT analysis failed: {e}", file=sys.stderr)
        talk_times = compute_talk_times(entries)
        word_counts = compute_word_counts(entries)
        fallback = {"agenda": "", "summary": "", "speakers": []}
        names = sorted(set(talk_times.keys()) | set(word_counts.keys()))
        for nm in names:
            fallback["speakers"].append({
                "name": nm,
                "relevance_score": 0.0,
                "pros": "",
                "cons": "",
                "final_verdict": "No GPT analysis available.",
                "talk_time_seconds": round(float(talk_times.get(nm, 0.0)), 2)
            })
        result = fallback
        write_err_code = 4

    if not isinstance(result, dict):
        error("analysis result is not an object", code=5)

    # Guarantee expected keys
    result.setdefault("agenda", "")
    result.setdefault("summary", "")
    result.setdefault("speakers", [])

    # Final normalization: coerce relevance_score to numeric 0-10 but do NOT cap or compute
    final_speakers = []
    for s in result.get("speakers", []):
        name = s.get("name") or s.get("speaker") or "unknown"
        gscore_raw = s.get("relevance_score", s.get("score"))
        gscore = _coerce_float_0_10(gscore_raw) if gscore_raw is not None else 0.0

        pros = s.get("pros") or s.get("strengths") or s.get("feedback") or ""
        cons = s.get("cons") or s.get("weaknesses") or ""
        verdict = s.get("final_verdict") or s.get("verdict") or ""
        tt = s.get("talk_time_seconds")
        if tt is None:
            tt = compute_talk_times(entries).get(name, 0.0)
        try:
            tt = float(tt)
        except Exception:
            tt = float(compute_talk_times(entries).get(name, 0.0))
        tt = round(tt, 2)

        final_speakers.append({
            "name": str(name),
            "relevance_score": float(gscore),
            "pros": str(pros or ""),
            "cons": str(cons or ""),
            "final_verdict": str(verdict or ""),
            "talk_time_seconds": tt
        })

    final_obj = {
        "agenda": str(result.get("agenda") or ""),
        "summary": str(result.get("summary") or ""),
        "speakers": final_speakers
    }
    if result.get("raw_output"):
        final_obj["raw_output"] = result.get("raw_output")

    # Write output file
    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(final_obj, fh, ensure_ascii=False, indent=2)
        print(f"[info] Saved analysis to {output_path}")
    except Exception as e:
        error(f"Failed to write output JSON: {e}", code=6)

    # exit with GPT error code if GPT failed earlier
    if write_err_code == 4:
        print("[warn] GPT call failed; wrote fallback analysis file.", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()
