#!/usr/bin/env python3
"""
asr.py - ASR -> diar mapping, writes:
  results/<stem>_assigned_words.json
  results/<stem>_dialogue.json
  results/<stem>_alignment_debug.txt
  results/<stem>_name_map.json

Behavior:
- When a diarization cluster self-introduces ("This is Ravi speaking"), assign
  FIRST NAME "Ravi" to that cluster and remember it.
- Retroactively update all previous assigned_words and utterances that were
  mapped to that cluster so their speaker becomes "Ravi".
- For future segments mapped to that cluster, always use "Ravi".
"""
from pathlib import Path
import json, os, re, sys
from collections import defaultdict

OUT = Path("results"); OUT.mkdir(exist_ok=True)

# Tunables (env overrides)
PREMERGE_MAX_GAP = float(os.getenv("PREMERGE_MAX_GAP", "0.6"))
PREMERGE_MAX_TOKENS = int(os.getenv("PREMERGE_MAX_TOKENS", "100"))
USE_NEAREST = os.getenv("USE_NEAREST", "0") == "1"
NEAREST_MAX_DISTANCE = float(os.getenv("NEAREST_MAX_DISTANCE", "0.15"))
MAX_MERGE_GAP = float(os.getenv("MAX_MERGE_GAP", "0.6"))

SENT_END_RE = re.compile(r'[\.?!]\s*$')
INTRO_RE = re.compile(r'^\s*(?:hi|hello|hey|i(?:\'m| am)|this is|my name is)\b', re.IGNORECASE)
ADDRESSING_PREFIX_RE = re.compile(r'^\s*[A-Za-z][a-z\-]+\s*(?:[,!:])', re.IGNORECASE)
ADDRESSING_HEAR_FROM_RE = re.compile(r'\b(?:hear from|from)\s+([A-Za-z][a-z\-]+)\b', re.IGNORECASE)

# Regex to remove leading self-intro fragments (defensive)
INTRO_REMOVE_RE = re.compile(
    r"""
    ^\s*(?:                          # beginning, optional whitespace
      (?:hi|hello|hey)[,!\.\s]*|     # greetings
      (?:this is|i(?:'m| am)|my name is)\b[\s,:-]*  # intro phrases
    )
    (?P<name>[A-Za-z][A-Za-z\-\']*(?:\s+[A-Za-z][A-Za-z\-\']*)?)   # name (one or two tokens)
    (?:\s+(?:speaking|here|present|on the line|available|now))?    # optional trailing words
    [\.\?!,]*\s*                      # optional punctuation and trailing whitespace
    """, re.IGNORECASE | re.VERBOSE
)

# Common stopwords / invalid name tokens
_NAME_STOPWORDS = {"in", "on", "at", "yes", "no", "ok", "okay", "thanks", "thank", "hello", "speaker"}

# Optional spaCy NER
NER_NLP = None
try:
    import spacy
    try:
        NER_NLP = spacy.load("en_core_web_sm")
    except Exception:
        NER_NLP = None
except Exception:
    NER_NLP = None

if NER_NLP is None:
    print("[INFO] spaCy NER not available — using regex fallback for name extraction.")
else:
    print("[INFO] spaCy NER loaded and will be used for name extraction.")

# ---------- I/O helpers ----------

def resolve_audio_path():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    raise SystemExit("Usage: python asr.py path/to/audio.wav")

def load_merged_diar(stem):
    j = OUT / f"{stem}_diar_merged.json"
    csv = OUT / f"{stem}_diarization.csv"
    diar = []
    if j.exists():
        arr = json.loads(j.read_text(encoding="utf-8"))
        for r in arr:
            diar.append({"start": float(r.get("start", 0.0)), "end": float(r.get("end", 0.0)), "speaker": r.get("speaker")})
        return diar
    if csv.exists():
        import csv as _csv
        with open(csv, newline="", encoding="utf-8") as f:
            r = _csv.DictReader(f)
            for row in r:
                try:
                    s = float(row.get("start") or 0.0); e = float(row.get("end") or 0.0)
                except:
                    continue
                sp = row.get("speaker") or row.get("label") or "spk_unknown"
                diar.append({"start": s, "end": e, "speaker": sp})
        return diar
    return diar

def load_asr_segments_or_transcribe(stem, audio_path):
    segfile = OUT / f"{stem}_asr_segments.json"
    if segfile.exists():
        return json.loads(segfile.read_text(encoding="utf-8"))
    raise SystemExit(f"ASR segments not found: {segfile}")

# ---------- low-level utils ----------

def overlap(a_s,a_e,b_s,b_e):
    return max(0.0, min(a_e,b_e) - max(a_s,b_s))

def premerge_asr_segments(asr_segments, dbg_lines):
    merged = []
    for i, seg in enumerate(asr_segments):
        st = float(seg.get("start", 0.0)); ed = float(seg.get("end", 0.0)); txt = (seg.get("text") or "").strip()
        toks = len(txt.split())
        item = {"start": st, "end": ed, "text": txt, "orig_idx": [i], "tokens": toks}
        if not merged:
            merged.append(item); continue
        cur = merged[-1]
        gap = st - cur["end"]
        if (not SENT_END_RE.search(cur["text"])) and gap <= PREMERGE_MAX_GAP and (cur["tokens"] + toks) <= PREMERGE_MAX_TOKENS:
            cur["end"] = ed
            cur["text"] = (cur["text"] + " " + txt).strip()
            cur["orig_idx"].append(i)
            cur["tokens"] = cur["tokens"] + toks
            dbg_lines.append(f"PREMERGE merged idx {i} -> tokens {cur['tokens']}")
        else:
            merged.append(item)
    return merged

def split_merged_piece_to_words(piece, asr_segments):
    return [{"start": float(asr_segments[i]["start"]),
             "end": float(asr_segments[i]["end"]),
             "text": (asr_segments[i].get("text") or "").strip()} for i in piece.get("orig_idx", [])]

def choose_speaker_by_majority_overlap(piece_items, diar):
    dur_by_sp = defaultdict(float)
    for item in piece_items:
        a_s = item["start"]; a_e = item["end"]
        for d in diar:
            ov = overlap(a_s, a_e, float(d["start"]), float(d["end"]))
            if ov > 0:
                dur_by_sp[d["speaker"]] += ov
    if not dur_by_sp:
        return None, {}
    sp, dur = max(dur_by_sp.items(), key=lambda kv: kv[1])
    return sp, dict(dur_by_sp)

# ---------- name extraction ----------

def looks_like_addressing(text):
    if not text: return False
    t = text.strip()
    if ADDRESSING_PREFIX_RE.search(t): return True
    if ADDRESSING_HEAR_FROM_RE.search(t): return True
    if re.search(r'\b(?:ask|call|ping)\s+[A-Za-z][a-z\-]+\b', t, re.IGNORECASE): return True
    return False

def _strip_punct(tok):
    return tok.strip(" ,.!?\"'()[]:;")

def _is_valid_name_token(tok):
    if not tok or not isinstance(tok, str):
        return False
    t = re.sub(r'[^A-Za-z\-\'`]', '', tok).strip()
    if not t:
        return False
    if t.lower() in _NAME_STOPWORDS:
        return False
    if len(t) < 2:
        return False
    return True

def extract_first_name_simple(text):
    if not text or not INTRO_RE.search(text):
        return None
    if looks_like_addressing(text):
        return None
    m = re.search(r"(?:this is|i am|i'm|im|my name is)\s+([A-Za-z][a-zA-Z\-']*)", text, re.IGNORECASE)
    if m:
        tok = _strip_punct(m.group(1))
        if not _is_valid_name_token(tok):
            return None
        return tok.title()
    for tok in text.strip().split():
        low = _strip_punct(tok).lower()
        if low in {"hi","hello","hey","i'm","im","i","am","this","is","my","name","me"}:
            continue
        if low in {"speaking","here","present","reporting","available","now"}:
            continue
        if not re.search(r'[A-Za-z]', tok):
            continue
        if _is_valid_name_token(tok):
            return _strip_punct(tok).title()
    return None

def extract_first_name_with_ner(text):
    if not text:
        return None
    if NER_NLP is not None:
        try:
            doc = NER_NLP(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    first = ent.text.strip().split()[0]
                    first = re.sub(r'[^A-Za-z\-\'`]', '', first)
                    if _is_valid_name_token(first):
                        return first.title()
        except Exception:
            pass
    return extract_first_name_simple(text)

# ---------- mapping core ----------

def _normalize_label_to_first_name(label):
    """Defensive: ensure label is a single first-name token (if possible)."""
    if not label or not isinstance(label, str):
        return None
    s = label.strip()
    # remove trailing junk if present
    s = re.sub(r'\b(?:speaking|here|present|reporting|available|now)\b[\.!?,\s]*$', '', s, flags=re.I).strip()
    if not s:
        return None
    # take first token only
    first = s.split()[0]
    first = re.sub(r'[^A-Za-z\-\'`]', '', first)
    if not _is_valid_name_token(first):
        return None
    return first.title()

def _retro_update_mappings(cluster_id, name, assigned_words, utterances):
    """Retroactively update previously created entries for cluster_id."""
    # assigned_words entries contain "spk_cluster" when cluster known
    for aw in assigned_words:
        if aw.get("spk_cluster") == cluster_id:
            aw["speaker"] = name
    # utterances may have spk_cluster saved; update those as well
    for utt in utterances:
        if utt.get("spk_cluster") == cluster_id:
            utt["speaker"] = name

def map_merged_to_speakers(merged, asr_segments, diar, dbg_lines):
    """
    Returns: assigned_words, utterances, cluster_to_name_map
    cluster_to_name_map: dict mapping diar cluster id -> first name (string)
    """
    diar_sorted = sorted(diar, key=lambda d: float(d["start"]))
    centers = [0.5*(d["start"]+d["end"]) for d in diar_sorted] if diar_sorted else []
    assigned_words = []
    utterances = []

    # persistent mapping for this run: cluster_id -> first-name
    cluster_to_name = {}

    for m in merged:
        piece_items = split_merged_piece_to_words(m, asr_segments)
        chosen_sp, dur_map = choose_speaker_by_majority_overlap(piece_items, diar_sorted)
        dbg_lines.append(f"VOTE [{m['start']:.3f}-{m['end']:.3f}] dur_map={dur_map} chosen={chosen_sp}")

        # fallback midpoint / nearest to choose_sp if voting failed
        if chosen_sp is None:
            mid = 0.5*(m["start"] + m["end"])
            contained = None
            for d in diar_sorted:
                if float(d["start"]) - 1e-9 <= mid <= float(d["end"]) + 1e-9:
                    contained = d["speaker"]; break
            if contained is not None:
                chosen_sp = contained
                dbg_lines.append(f" MIDPOINT -> {chosen_sp}")
            elif USE_NEAREST and centers:
                nearest_idx = min(range(len(centers)), key=lambda i: abs(centers[i] - mid))
                if abs(centers[nearest_idx] - mid) <= NEAREST_MAX_DISTANCE:
                    chosen_sp = diar_sorted[nearest_idx]["speaker"]
                    dbg_lines.append(f" NEAREST -> {chosen_sp}")

        merged_text = m.get("text", "") or ""

        # If the chosen_sp already has a mapped name, use it unconditionally
        if chosen_sp is not None and chosen_sp in cluster_to_name:
            final_label = cluster_to_name[chosen_sp]
            dbg_lines.append(f"MAPPED: cluster {chosen_sp} -> {final_label} (cached)")
            cleaned_text = merged_text  # do not try to strip intro anymore
        else:
            detected_name = extract_first_name_with_ner(merged_text)
            cleaned_text = merged_text
            if detected_name and chosen_sp is not None:
                # we have both a detected name and a cluster: persist mapping and use it
                normalized = _normalize_label_to_first_name(detected_name) or detected_name
                cluster_to_name[chosen_sp] = normalized
                final_label = normalized
                dbg_lines.append(f"NAME-DETECT+MAP: cluster {chosen_sp} -> '{normalized}' from '{merged_text[:120]}'")
                # remove leading self-intro phrase from merged_text
                nm = INTRO_REMOVE_RE.sub("", merged_text).strip()
                nm = re.sub(r'^[\s\-\:\,]+', '', nm)
                if nm:
                    cleaned_text = nm
                # retroactively update any previous entries for this cluster
                _retro_update_mappings(chosen_sp, normalized, assigned_words, utterances)
            elif detected_name and chosen_sp is None:
                # detected a name but we cannot map to a cluster — use it for this utterance only
                final_label = _normalize_label_to_first_name(detected_name) or detected_name
                dbg_lines.append(f"NAME-DETECT (no-cluster): using '{final_label}' for this utterance")
                nm = INTRO_REMOVE_RE.sub("", merged_text).strip()
                nm = re.sub(r'^[\s\-\:\,]+', '', nm)
                if nm:
                    cleaned_text = nm
            else:
                # no name detected — use the chosen_sp (can be None)
                if looks_like_addressing(merged_text):
                    dbg_lines.append(f"IGNORED-ADDRESSING: '{merged_text[:120]}'")
                final_label = chosen_sp
                cleaned_text = merged_text

        # defensive normalization: if final_label is a cluster id or odd token, try to return first name token
        if isinstance(final_label, str) and final_label not in cluster_to_name.values():
            norm = _normalize_label_to_first_name(final_label)
            if norm:
                final_label = norm

        # assign each small segment; if cluster has a mapped name, use that name
        for idx_it, it in enumerate(piece_items):
            seg_text = it["text"]
            # if we mapped cluster to name and this is the first small segment, strip intro if present
            if chosen_sp is not None and chosen_sp in cluster_to_name and idx_it == 0:
                st = INTRO_REMOVE_RE.sub("", seg_text).strip()
                st = re.sub(r'^[\s\-\:\,]+', '', st)
                if st:
                    seg_text = st
                else:
                    seg_text = it["text"]
            entry = {"start": it["start"], "end": it["end"], "text": seg_text, "speaker": final_label}
            if chosen_sp is not None:
                entry["spk_cluster"] = chosen_sp
            assigned_words.append(entry)

        # append utterance and store spk_cluster for retro updates later if needed
        utt_entry = {"speaker": final_label, "start": round(m["start"],3), "end": round(m["end"],3), "text": cleaned_text}
        if chosen_sp is not None:
            utt_entry["spk_cluster"] = chosen_sp
        utterances.append(utt_entry)

    return assigned_words, utterances, cluster_to_name

def merge_consecutive_same_speaker_utterances(utterances):
    if not utterances:
        return []
    out = []
    cur = dict(utterances[0]); cur_words = [cur["text"]]
    for u in utterances[1:]:
        if cur["speaker"] is not None and u["speaker"] == cur["speaker"] and (u["start"] - cur["end"]) <= MAX_MERGE_GAP:
            cur["end"] = u["end"]; cur_words.append(u["text"])
        else:
            cur["text"] = " ".join(cur_words).strip(); out.append(cur)
            cur = dict(u); cur_words = [cur["text"]]
    cur["text"] = " ".join(cur_words).strip(); out.append(cur)
    for r in out:
        r["start"] = round(float(r["start"]),3); r["end"] = round(float(r["end"]),3)
    return out

# ---------- main ----------

def _backup_if_exists(p: Path):
    if p.exists():
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

def main():
    audio = resolve_audio_path()
    stem = audio.stem
    diar = load_merged_diar(stem)
    if not diar:
        print("[ERROR] no diarization found (results/<stem>_diar_merged.json or CSV)."); sys.exit(1)
    asr_segments = load_asr_segments_or_transcribe(stem, audio)

    dbg_lines = []
    smalls = []
    for s in asr_segments:
        try:
            st = float(s.get("start", 0.0)); ed = float(s.get("end", 0.0))
        except:
            continue
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        smalls.append({"start": st, "end": ed, "text": txt})
    dbg_lines.append(f"ASR small segments: {len(smalls)}")

    merged = premerge_asr_segments(smalls, dbg_lines)
    dbg_lines.append(f"Premerge pieces: {len(merged)}")

    assigned_words, utterances, cluster_to_name = map_merged_to_speakers(merged, smalls, diar, dbg_lines)
    utterances = merge_consecutive_same_speaker_utterances(utterances)

    # Persist results (and name_map)
    aw_path = OUT / f"{stem}_assigned_words.json"
    dlg_path = OUT / f"{stem}_dialogue.json"
    dbg_path = OUT / f"{stem}_alignment_debug.txt"
    name_map_path = OUT / f"{stem}_name_map.json"

    _backup_if_exists(aw_path); _backup_if_exists(dlg_path); _backup_if_exists(dbg_path); _backup_if_exists(name_map_path)

    aw_path.write_text(json.dumps(assigned_words, indent=2, ensure_ascii=False), encoding="utf-8")
    dlg_path.write_text(json.dumps(utterances, indent=2, ensure_ascii=False), encoding="utf-8")
    # persist cluster -> name mapping (human-friendly)
    name_map_serializable = {str(k): v for k, v in cluster_to_name.items()}
    name_map_path.write_text(json.dumps(name_map_serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    dbg = {"diar_segments": len(diar), "asr_small_segments": len(smalls),
           "premerge_pieces": len(merged), "utterances": len(utterances), "debug": dbg_lines,
           "cluster_to_name": cluster_to_name}
    dbg_path.write_text(json.dumps(dbg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {stem}: wrote {len(assigned_words)} assigned words, {len(utterances)} utterances.")
    print("Wrote:", aw_path, dlg_path, dbg_path, name_map_path)

if __name__ == "__main__":
    main()
