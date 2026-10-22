# -*- coding: utf-8 -*-
"""Score cluster_results/ablations.jsonl (raw per-doc predictions from
cluster/decoding_ablations_cluster.py) against TAB's adjudicated security_confidential_only
label. Produces the same summary shape as datalake_decoding_ablations.py's local run would,
for Step 7. Platform: hpc-gpu-ablations (LLaMA 3.2 1B, full 1268 corpus, 5 conditions).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "external"))
import tab_adapter as ta  # noqa: E402

docs_by_id = {}
for split in ta.SPLITS:
    for d in ta.load_split(split):
        docs_by_id[d["doc_id"]] = d

gold = {doc_id: ta.conf_only_adjudicated(doc) for doc_id, doc in docs_by_id.items()}

recs = [json.loads(l) for l in open(BASE_DIR / "cluster_results" / "ablations.jsonl", encoding="utf-8") if l.strip()]
df = pd.DataFrame(recs)

rows = []
for cond, grp in df.groupby("condition"):
    n_docs = len(grp)
    n_valid = int(grp["valid_output"].sum())
    scoreable = grp[grp["valid_output"] & grp["doc_id"].map(lambda d: gold.get(d) is not None)]
    n_scored = len(scoreable)
    n_correct = int((scoreable["pred"] == scoreable["doc_id"].map(gold)).sum())
    rows.append({
        "condition": cond, "platform": "hpc-gpu-ablations", "n_docs": n_docs,
        "n_valid_output": n_valid, "valid_output_rate": round(n_valid / n_docs, 3),
        "n_scored_vs_adjudicated": n_scored, "n_correct": n_correct,
        "accuracy_vs_adjudicated": round(n_correct / n_scored, 3) if n_scored else None,
        "mean_attempts_used": round(float(grp["attempts_used"].mean()), 3),
    })

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = BASE_DIR / "evaluation" / "output"
out_path = out_dir / f"decoding_ablations_{ts}.csv"
summary = pd.DataFrame(rows).sort_values("condition")
summary.to_csv(out_path, index=False)
print(summary.to_string(index=False))
print(f"\nOutput -> {out_path}")
