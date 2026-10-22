"""
Where do LLaMA 3.2 1B's `general` labels come from?

When the model's reply contains no parseable JSON, datalake_llm_pipeline.py records
info_type=unknown / security_class=general with the reason "LLM parse failure — defaulted
to unknown/general". This counts, per run, the unparseable replies (from the DB and,
independently, from the run log's "JSON parse failed" warnings), how many `general` labels
are that fallback, how many the model produced itself, and the placeholder echoes
(security_class containing "<").

Run:
  python score_llama_fallbacks.py evaluation/results/variance_runs_*.csv
Output: evaluation/output/llama_fallbacks_<ts>.csv
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
JULY_VERSION = "unrecorded (<=0.32.3, the newest release at run time)"
REFERENCES = [("ref_20260723_134633", "20260723_134633", JULY_VERSION),
              ("ref_20261021_222833", "20261021_222833", "0.40.7")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", type=Path, nargs="+")
    ap.add_argument("--default-ollama-version", default="0.40.7",
                    help="for manifests written before the runner recorded the version")
    args = ap.parse_args()

    runs = list(REFERENCES)
    for m in args.manifests:
        f = pd.read_csv(m, dtype=str).rename(columns={"pass": "pass_id"})   # "pass" is a keyword
        for r in f[f.model == "llama3.2:1b"].itertuples():
            runs.append((r.pass_id, r.run_ts, getattr(r, "ollama_version", None) or args.default_ollama_version))

    rows = []
    for pass_id, ts, version in runs:
        with sqlite3.connect(OUTPUT_DIR / f"datalake_llm_llama32_1b_{ts}.db") as c:
            d = pd.read_sql("SELECT security_class, llm_reason FROM data_files", c)
        fallback = d.llm_reason.fillna("").str.startswith("LLM parse failure")
        general = d.security_class == "general"
        log = (OUTPUT_DIR / f"log_llm_llama32_1b_{ts}.txt").read_text(encoding="utf-8")
        rows.append({"pass": pass_id, "run_ts": ts, "ollama_version": version, "records": len(d),
                     "unparseable_db": int(fallback.sum()), "unparseable_log": log.count("JSON parse failed"),
                     "general_labels": int(general.sum()),
                     "general_from_fallback": int((general & fallback).sum()),
                     "general_from_model": int((general & ~fallback).sum()),
                     "placeholder_echoes": int(d.security_class.fillna("").str.contains("<").sum())})

    out = BASE_DIR / "evaluation" / "output" / f"llama_fallbacks_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
