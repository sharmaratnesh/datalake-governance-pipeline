"""
Primary-corpus results as a distribution, not a single chosen run (A8).

Reads a variance_primary_<ts>.csv (score_variance_passes.py) and writes
evaluation/output/primary_distribution_<ts>.csv with, per model and metric:
  median / min / max over the five temperature-0.1 passes on Ollama 0.40.7,
  the temperature-0 value on 0.40.7 (the two T=0 passes are identical; checked),
  the July and October reference runs,
  and the temperature-0 value on each older Ollama build present in the input.

Run:
  python score_primary_distribution.py evaluation/output/variance_primary_<ts>.csv
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

EVAL_OUT = Path(__file__).parent / "evaluation" / "output"
METRICS = ["info_type_accuracy", "sensitivity_accuracy", "sensitivity_kappa", "general_precision",
           "general_recall", "general_f1", "linkage_accuracy", "overall_accuracy", "n_predicted_general"]
CURRENT = "0.40.7"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variance_csv", type=Path)
    args = ap.parse_args()
    v = pd.read_csv(args.variance_csv, dtype={"ollama_version": str, "run_ts": str})

    rows = []
    for model, d in v.groupby("model", sort=False):
        cur = d[d.ollama_version == CURRENT]
        t01 = cur[(cur.temperature == 0.1) & ~cur["pass"].str.startswith("ref_")]
        t0 = cur[cur.temperature == 0.0]
        if len(t01) != 5:
            sys.exit(f"{model}: expected 5 temperature-0.1 passes on {CURRENT}, found {len(t01)}")
        if len(t0[METRICS].drop_duplicates()) != 1:
            sys.exit(f"{model}: the temperature-0 passes on {CURRENT} disagree - report them separately")
        refs = d[d["pass"].str.startswith("ref_")].sort_values("run_ts")
        july, october = refs.iloc[0], refs.iloc[1]
        old = d[(d.ollama_version != CURRENT) & ~d["pass"].str.startswith("ref_")]
        for metric in METRICS:
            row = {"model": model, "metric": metric,
                   f"t0.1_median_{CURRENT}": t01[metric].median(),
                   f"t0.1_min_{CURRENT}": t01[metric].min(),
                   f"t0.1_max_{CURRENT}": t01[metric].max(),
                   f"t0_{CURRENT}": t0[metric].iloc[0],
                   "july": july[metric], "october": october[metric]}
            for r in old.sort_values("ollama_version").itertuples():
                row[f"t{r.temperature:g}_{r.ollama_version}"] = getattr(r, metric)
            row.update({"july_run": july.run_ts, "october_run": october.run_ts})
            rows.append(row)

    out = EVAL_OUT / f"primary_distribution_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
