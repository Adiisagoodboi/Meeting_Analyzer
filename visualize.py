# visualize.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def plot_pie(labels, sizes, out_png: Path, title: str = "Talk-time Contribution"):
    total = sum(sizes)
    if total <= 0:
        print(f"[warn] no valid durations, skipping plot → {out_png}")
        return

    def autopct(pct):
        secs = int(round(pct * total / 100.0))
        m, s = divmod(secs, 60)
        return f"{pct:.1f}%\n{m}:{s:02d}"

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct=autopct, startangle=90)
    plt.title(title)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"[info] saved pie chart → {out_png}")

def main():
    # Prefer JSON durations (new pipeline)
    json_files = sorted(OUT_DIR.glob("*_speaking_durations.json"))
    if json_files:
        for jf in json_files:
            stem = jf.stem.replace("_speaking_durations", "")
            data = json.loads(jf.read_text(encoding="utf-8"))
            labels = list(data.keys())
            sizes = list(data.values())
            out_png = OUT_DIR / f"{stem}_pie.png"
            plot_pie(labels, sizes, out_png, f"Talk-time Contribution — {stem}")
        return

    # Fallback: metrics.csv (old pipeline compatibility)
    metrics_files = sorted(OUT_DIR.glob("*_metrics.csv"))
    if metrics_files:
        for mf in metrics_files:
            stem = mf.stem.replace("_metrics", "")
            df = pd.read_csv(mf)
            if df.empty: 
                continue
            labels = df["name"].tolist()
            sizes = df["speaking_seconds"].tolist()
            out_png = OUT_DIR / f"{stem}_pie.png"
            plot_pie(labels, sizes, out_png, f"Talk-time Contribution — {stem}")
        return

    print("[error] no speaking durations found in results/.")

if __name__ == "__main__":
    main()
