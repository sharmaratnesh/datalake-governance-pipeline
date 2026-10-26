"""
Per-participant scores on the minority sensitivity class (non-confidential, "general").

For each annotator: how many records they labelled non-confidential, and their precision,
recall and F1 on that class against the golden labels. These are published as aggregates
(pseudonymous P1-P3), like the per-participant accuracies in evaluation_human_detail_*.csv;
the per-record answers they are computed from are not published. The mean row is Table 3's
human row.

Needs the withheld annotator sheets (evaluation/participant-*.xlsx), so it can only be run
where those exist.

  python scripts/human_minority_class.py evaluation/output/human_minority_class_<ts>.csv
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
import datalake_evaluate_all as ev  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        BASE_DIR / "evaluation" / "output" / f"human_minority_class_{datetime.now():%Y%m%d_%H%M%S}.csv"
    golden, _ = ev.read_baseline(BASE_DIR / "evaluation" / "evaluation_template_golden.xlsx")
    gold = golden.set_index("file_name")["golden_security"]
    sheets = sorted((BASE_DIR / "evaluation").glob("participant*.xlsx"))
    if not sheets:
        sys.exit("no evaluation/participant*.xlsx: the annotator sheets are withheld and not in the public repo")
    rows = []
    for i, path in enumerate(sheets, 1):
        pred, _ = ev.read_participant(path)
        s = pred.set_index("file_name").reindex(gold.index)["security"]
        tp = int(((s == "general") & (gold == "general")).sum())
        fp = int(((s == "general") & (gold != "general")).sum())
        fn = int(((s != "general") & (gold == "general")).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"participant": f"P{i}", "n_labelled_non_confidential": tp + fp,
                     "precision": round(p, 3), "recall": round(r, 3),
                     "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0})
    df = pd.DataFrame(rows)
    mean = {"participant": "mean", **{c: round(df[c].mean(), 3) for c in df.columns if c != "participant"}}
    mean["n_labelled_non_confidential"] = round(df["n_labelled_non_confidential"].mean(), 1)
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
