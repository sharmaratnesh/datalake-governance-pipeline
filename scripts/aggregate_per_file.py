"""
Make a publishable copy of evaluation_per_file_<ts>.csv.

The per-file CSV written by datalake_evaluate_all.py carries every annotator's
per-record response (participant_<id>_{category,security,linked,cat_ok,sec_ok,link_ok}).
Those are individual human-subject responses (ETH25-11255) and are withheld. This
replaces them with the three-annotator majority for information type and linkage:

  human_majority_{category,linked}   majority label ("no_majority" on a 3-way tie)
  human_majority_{cat_ok,link_ok}    majority label == golden label

Two things are deliberately NOT published:
  * a count of annotators correct: a count of 3 (or 0 on a binary task) means the record
    was unanimous, which discloses each annotator's response;
  * the per-record majority for sensitivity: combined with the published per-participant
    accuracies it identifies which records each annotator got wrong (P2 scored 1.000 and
    P1/P3 two errors each, so the records where the majority is wrong are exactly theirs).

  python scripts/aggregate_per_file.py <evaluation_per_file_ts.csv> <out.csv>
"""

import sys
from collections import Counter

import pandas as pd

TASKS = {"category": "cat_ok", "security": "sec_ok", "linked": "link_ok"}
PUBLISHED_MAJORITY = ("category", "linked")


def majority(row) -> str:
    (label, n), = Counter(row).most_common(1)
    return label if n >= 2 else "no_majority"


def main(src: str, dst: str) -> None:
    df = pd.read_csv(src)
    participants = sorted({c[: -len(f"_{t}")] for c in df.columns for t in TASKS
                           if c.startswith("participant_") and c.endswith(f"_{t}")})
    if len(participants) != 3:
        sys.exit(f"expected 3 participants, found {participants}")
    for task, ok in TASKS.items():
        if task not in PUBLISHED_MAJORITY:
            continue
        maj = df[[f"{p}_{task}" for p in participants]].apply(majority, axis=1)
        df[f"human_majority_{task}"] = maj
        df[f"human_majority_{ok}"] = maj.astype(str) == df[f"golden_{task}"].astype(str)
    dropped = [c for c in df.columns if c.startswith("participant_")]
    df = df.drop(columns=dropped)
    df.to_csv(dst, index=False)
    print(f"dropped {len(dropped)} columns: {dropped}")
    print(f"added: {[c for c in df.columns if c.startswith('human_majority_')]}")
    print(f"-> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
