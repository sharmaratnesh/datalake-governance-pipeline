"""
Sensitivity agreement on the 45 non-audio records versus the 10 audio records.

The audio inputs (faster-whisper transcripts) are not bit-identical between runs; the
other 45 inputs are. Scoring the same run on each subset separately shows whether a
difference between runs survives where the inputs are provably identical.

Run:
  python score_nonaudio_subset.py --run july=output/metadata_llm_phi3_mini_20260723_135524.csv \
                                  --run october=output/metadata_llm_phi3_mini_20261021_223725.csv
Output: evaluation/output/nonaudio_subset_<ts>.csv
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).parent
AUDIO = (".wav", ".mp3", ".m4a", ".flac")
LABELS = ["confidential", "general"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=CSV")
    ap.add_argument("--golden", default="evaluation/evaluation_template_golden.xlsx")
    args = ap.parse_args()

    g = pd.read_excel(BASE_DIR / args.golden, sheet_name="evaluation", header=1).dropna(subset=["File Name"])
    gold = g.set_index(g["File Name"].str.strip())["Security Classification"].str.strip().str.lower()
    audio = gold.index.str.lower().str.endswith(AUDIO)
    print(f"audio records ({audio.sum()}): {'; '.join(sorted(gold.index[audio]))}")

    rows = []
    for spec in args.run:
        label, _, csv = spec.partition("=")
        pred = pd.read_csv(BASE_DIR / csv).set_index("file_name")["security_class"].str.strip().str.lower()
        for subset, mask in [("all_55", audio | ~audio), ("non_audio_45", ~audio), ("audio_10", audio)]:
            yt = gold[mask]
            yp = pred.reindex(yt.index).fillna("general")
            p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=["general"], zero_division=0)
            rows.append({"run": label, "source_csv": Path(csv).name, "subset": subset, "n": int(mask.sum()),
                         "golden_general": int((yt == "general").sum()),
                         "sensitivity_kappa": round(cohen_kappa_score(yt, yp, labels=LABELS), 3),
                         "sensitivity_accuracy": round(accuracy_score(yt, yp), 3),
                         "general_precision": round(float(p[0]), 3), "general_recall": round(float(r[0]), 3),
                         "general_f1": round(float(f[0]), 3), "n_predicted_general": int((yp == "general").sum())})

    out = BASE_DIR / "evaluation" / "output" / f"nonaudio_subset_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
