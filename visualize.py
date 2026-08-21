#!/usr/bin/env python3
"""
visualize.py — Minimal dark-glass donut chart generator (no outer labels)

Saves: results/<stem>_pie.png
"""

from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import sys, traceback

ROOT = Path.cwd()
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

def safe_print(s: str):
    """Print without crashing on consoles that can't encode certain Unicode characters.
    Tries normal print(), falls back to writing UTF-8 bytes to stdout.buffer.
    """
    try:
        # try normal printing first
        print(s)
    except UnicodeEncodeError:
        try:
            # write UTF-8 bytes to buffer (bypass encoding step)
            sys.stdout.buffer.write((s + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            # last resort: ignore printing errors
            pass

def secs_to_hms(sec_total):
    sec = int(round(sec_total))
    m, s = divmod(sec, 60)
    return f"{m}:{s:02d}" if m else f"{s}s"

def text_contrast_color(col):
    r, g, b = mcolors.to_rgb(col)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.58 else "white"

def plot_donut_dark(names, durations, out_png: Path):
    total = sum(durations)
    if total <= 0:
        safe_print(f"[warn] no data → {out_png}")
        return

    # sort descending
    paired = sorted(zip(names, durations), key=lambda x: x[1], reverse=True)
    names_sorted, durations_sorted = zip(*paired)
    pct = [100.0 * d / total for d in durations_sorted]
    times_str = [secs_to_hms(d) for d in durations_sorted]

    # dark-friendly pastel palette
    palette = [
        "#7FB6FF", "#FFCF9A", "#A7E6A0", "#FF9AA2",
        "#C7B3FF", "#8CD7D9", "#FFDDE4", "#E8E29A"
    ]
    colors = [palette[i % len(palette)] for i in range(len(names_sorted))]

    # figure setup
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    plt.subplots_adjust(left=0.03, right=0.80)

    wedges, _ = ax.pie(
        durations_sorted,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.36, edgecolor=(0,0,0,0.18), linewidth=1.6),
        normalize=True
    )

    # subtle glow for each wedge
    for w in wedges:
        w.set_path_effects([path_effects.Stroke(linewidth=6, foreground=(0,0,0,0.1)), path_effects.Normal()])

    # inner dark circle
    centre_circle = plt.Circle((0,0), 0.34, fc=(0.03, 0.06, 0.12, 0.78), linewidth=0)
    ax.add_artist(centre_circle)
    ax.set(aspect="equal")
    ax.set_title("Talk-time Contribution", fontsize=30, weight="bold", pad=26, color="#f4f7fb")

    # inner wedge text: % and time
    for w, pval, tstr in zip(wedges, pct, times_str):
        ang = (w.theta2 + w.theta1) / 2.0
        ang_rad = math.radians(ang)
        tx, ty = 0.72 * math.cos(ang_rad), 0.72 * math.sin(ang_rad)
        ax.text(
            tx, ty,
            f"{pval:.1f}%\n({tstr})",
            ha="center", va="center",
            fontsize=13, weight="700",
            color="white",
            bbox=dict(boxstyle="round,pad=0.14", facecolor=(0.06,0.08,0.12,0.86), lw=0)
        )

    # right-side legend boxes only
    start_y = 0.82
    dy = 0.105
    for i, (name, d, perc, col) in enumerate(zip(names_sorted, durations_sorted, pct, colors)):
        y = start_y - i * dy
        txt = f" {name}   {secs_to_hms(d)}   •   {perc:.1f}% "
        text_color = text_contrast_color(col)
        ax.text(
            1.02, y,
            txt,
            transform=ax.transAxes,
            fontsize=14,
            weight="700",
            va="center",
            ha="left",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=mcolors.to_rgba(col, 0.92), edgecolor=(1,1,1,0.06), lw=0.9)
        )

    fig.text(0.02, 0.02, f"Total: {secs_to_hms(total)}", fontsize=12, color="#cbd2e6")

    # save (ensure parent exists)
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
        safe_print(f"[info] saved pie chart -> {out_png.resolve()}")
    except Exception as e:
        safe_print(f"[error] failed to save pie chart {out_png}: {e}")
        try:
            plt.close(fig)
        except Exception:
            pass

def main():
    json_files = sorted(OUT_DIR.glob("*_speaking_durations.json"))
    if not json_files:
        safe_print("[error] no *_speaking_durations.json found in results/.")
        return

    for jf in json_files:
        try:
            stem = jf.stem.replace("_speaking_durations", "")
            safe_print(f"[info] processing stem: {stem}")
            durations = json.loads(jf.read_text(encoding="utf-8"))
            if not durations:
                continue

            name_map_file = OUT_DIR / f"{stem}_name_map.json"
            name_map = json.loads(name_map_file.read_text(encoding="utf-8")) if name_map_file.exists() else {}
            names = [name_map.get(spk, spk) for spk in durations.keys()]
            times = [float(durations[k]) for k in durations.keys()]
            out_png = OUT_DIR / f"{stem}_pie.png"
            plot_donut_dark(names, times, out_png)
        except Exception as e:
            safe_print(f"[error] while processing {jf.name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
