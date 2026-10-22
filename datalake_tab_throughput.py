# -*- coding: utf-8 -*-
"""
Step 5c — Throughput / latency measurements, per method AND per platform (never merged).

Reads per-record `processing_sec` from each method's full-TAB-corpus metadata CSV (laptop
methods: Rule-Based, NLP, LLM-llama3.2-1b, LLM-phi3-mini — each must have exactly 1268 rows
to be counted as the full run) and from any cluster result JSONL files staged into
cluster_results/ (anchor / scale-probe runs, tagged with their own `platform` field by
cluster/ollama_tab_runner.py).

Emits evaluation/output/throughput_<ts>.csv: one row per (method, platform), with records
processed, total wall-clock, records/minute, mean/median/p95 per-record latency. Platform is
an explicit column on every row — rows are never averaged or merged across platforms.

Usage:
    python datalake_tab_throughput.py [--csv key=path ...]

Inputs are selected explicitly (datalake_artefacts.py): a method's metadata CSV is
auto-selected only if it is the ONLY full-corpus (1268-row) candidate in output/;
otherwise pin it with --csv <key>=<path>, key = prefix without "metadata_", e.g.
    --csv rule_based=output/metadata_rule_based_20261021_235159.csv \
    --csv nlp=output/metadata_nlp_20261021_235257.csv
"""
import argparse
import glob
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from datalake_artefacts import select_artefact, row_count_check, parse_pins, prefix_key

BASE_DIR = Path(__file__).parent
EXPECTED_FULL_CORPUS_SIZE = 1268

LAPTOP_METHODS = {
    "Rule-Based": "metadata_rule_based_",
    "NLP": "metadata_nlp_",
    "LLM-llama3.2-1b": "metadata_llm_llama32_1b_",
    "LLM-phi3-mini": "metadata_llm_phi3_mini_",
}


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tab_throughput")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def stats_from_latencies(latencies: list, platform: str, method: str, source: str, n_expected=None):
    arr = np.array(latencies, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    total = float(arr.sum())
    return {
        "method": method, "platform": platform, "source": source,
        "n_records": n, "n_expected_full_corpus": n_expected,
        "total_wallclock_sec": round(total, 2),
        "records_per_minute": round(60.0 * n / total, 2) if total > 0 else None,
        "mean_latency_sec": round(float(arr.mean()), 3),
        "median_latency_sec": round(float(np.median(arr)), 3),
        "p95_latency_sec": round(float(np.percentile(arr, 95)), 3),
    }


def main():
    ap = argparse.ArgumentParser(description="TAB throughput per method and platform")
    ap.add_argument("--csv", action="append", metavar="KEY=PATH", help="pin a metadata CSV (repeatable)")
    pins = parse_pins(ap.parse_args().csv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_tab_throughput_{ts}.txt")
    output_dir = BASE_DIR / "output"

    logger.info("=" * 70)
    logger.info("TAB THROUGHPUT (Step 5c) — laptop-cpu and hpc-gpu kept strictly separate")
    logger.info("=" * 70)

    rows = []
    for method, prefix in LAPTOP_METHODS.items():
        csv_path = select_artefact(output_dir, f"{prefix}*.csv", explicit=pins.get(prefix_key(prefix)),
                                   accept=row_count_check(EXPECTED_FULL_CORPUS_SIZE), what=method,
                                   option=f"--csv {prefix_key(prefix)}=")
        if not csv_path:
            logger.warning(f"{method}: no metadata CSV found — skipping")
            continue
        df = pd.read_csv(csv_path)
        if "processing_sec" not in df.columns:
            logger.warning(f"{method}: {csv_path.name} has no processing_sec column — skipping")
            continue
        if len(df) != EXPECTED_FULL_CORPUS_SIZE:
            logger.warning(f"{method}: {csv_path.name} has {len(df)} rows, expected "
                            f"{EXPECTED_FULL_CORPUS_SIZE} — NOT counted as the full TAB run, skipping. "
                            "(Likely still a smoke-test or partial checkpoint file.)")
            continue
        row = stats_from_latencies(df["processing_sec"].tolist(), "laptop-cpu", method,
                                    csv_path.name, EXPECTED_FULL_CORPUS_SIZE)
        if row:
            rows.append(row)
            logger.info(f"{method:<16} laptop-cpu  n={row['n_records']}  "
                        f"rpm={row['records_per_minute']}  mean={row['mean_latency_sec']}s  "
                        f"p95={row['p95_latency_sec']}s")

    cluster_dir = BASE_DIR / "cluster_results"
    cluster_files = sorted(glob.glob(str(cluster_dir / "*.jsonl"))) if cluster_dir.exists() else []
    if not cluster_files:
        logger.info("No cluster_results/*.jsonl found yet — cluster rows omitted "
                     "(stage results back from the HPC login node's ~/tab_experiment/results/ first)")
    for jsonl_path in cluster_files:
        recs = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
        if not recs:
            continue
        if "processing_sec" not in recs[0]:
            logger.info(f"{Path(jsonl_path).name}: no processing_sec field (not a timing "
                        "run, e.g. decoding ablations) — skipping for throughput")
            continue
        platform = recs[0].get("platform", "hpc-gpu-unknown")
        model = recs[0].get("model", "unknown")
        method = f"LLM-{model}"
        row = stats_from_latencies([r["processing_sec"] for r in recs], platform, method,
                                    Path(jsonl_path).name)
        if row:
            rows.append(row)
            logger.info(f"{method:<16} {platform:<22} n={row['n_records']}  "
                        f"rpm={row['records_per_minute']}  mean={row['mean_latency_sec']}s  "
                        f"p95={row['p95_latency_sec']}s")

    out_path = out_dir / f"throughput_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"\nOutput -> {out_path}  ({len(rows)} rows)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
