# -*- coding: utf-8 -*-
"""
Convert a cluster_results/*.jsonl file (from ollama_tab_runner.py — the anchor or
scale-probe run) into the same metadata_*.csv shape datalake_tab_evaluate.py,
datalake_tab_throughput.py and datalake_tab_span_eval.py expect for a method, so the
GPU-cluster rows score alongside the laptop methods. Writes to output/, named
`metadata_cluster_<model_slug>_<ts>.csv` — that prefix is deliberately distinct from the
laptop naming (`metadata_llm_<model_slug>_*.csv`) so the two platforms are never picked up
as the same "latest" file for a method.

Usage:
    python cluster/convert_cluster_results.py cluster_results/anchor_phi3_mini.jsonl
    python cluster/convert_cluster_results.py cluster_results/scale_probe_llama31_8b.jsonl
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("jsonl_path", type=Path)
a = ap.parse_args()

recs = [json.loads(line) for line in open(a.jsonl_path, encoding="utf-8") if line.strip()]
if not recs:
    raise SystemExit(f"{a.jsonl_path}: no records found")

df = pd.DataFrame(recs).rename(columns={"llm_customer_id": "customer_id"})
model_slug = recs[0]["model"].replace(":", "_").replace(".", "")
df["method"] = f"llm_{model_slug}_cluster"

keep = ["file_name", "file_type", "file_size_bytes", "info_type", "security_class",
        "customer_id", "llm_reason", "method", "model", "processing_sec", "platform"]
out = df[[c for c in keep if c in df.columns]]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = BASE_DIR / "output" / f"metadata_cluster_{model_slug}_{ts}.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(f"{len(out)} rows -> {out_path}")
