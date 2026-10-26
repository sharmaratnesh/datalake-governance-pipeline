"""
Build one per-file evaluation table from several datalake_evaluate_all.py runs.

Starts from a base evaluation_per_file_<ts>.csv and replaces the columns of the named
methods (<method>_{category,security,linked,cat_ok,sec_ok,link_ok}) with those from
per-file CSVs of other runs - e.g. the October reference run with its two LLM columns
swapped for the median variance passes. Refuses to merge unless every input has the same
files in the same order with identical golden labels.

  python scripts/merge_per_file.py <base.csv> <out.csv> \
      --method llm_phi3_mini=<per_file_of_phi3_pass.csv> \
      --method llm_llama3.2_1b=<per_file_of_llama_pass.csv>
"""

import argparse
import sys

import pandas as pd

SUFFIXES = ["category", "security", "linked", "cat_ok", "sec_ok", "link_ok"]
GOLDEN = ["file_name", "golden_category", "golden_security", "golden_linked"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--method", action="append", required=True, metavar="NAME=CSV")
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    for spec in args.method:
        name, _, path = spec.partition("=")
        src = pd.read_csv(path)
        if not base[GOLDEN].astype(str).equals(src[GOLDEN].astype(str)):
            sys.exit(f"{path}: files, order or golden labels differ from {args.base}")
        cols = [f"{name}_{s}" for s in SUFFIXES]
        missing = [c for c in cols if c not in base or c not in src]
        if missing:
            sys.exit(f"{name}: columns missing: {missing}")
        base[cols] = src[cols].values
        print(f"{name}: columns taken from {path}")
    base.to_csv(args.out, index=False)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
