# -*- coding: utf-8 -*-
"""
Step 5c (scoring half) — score all four pipeline methods' TAB output against tab_golden.xlsx.

Primary: security_confidential_only (Adjudicated), scored only on the subset of documents
that have an adjudicated label (quality_checked non-empty — the primary evaluation set).
Sensitivity analyses: Union and Intersection derivations, scored on the full 1268.

Cohen's kappa is the HEADLINE metric (not accuracy) per the brief's revised Step 5 — accuracy
and its Wilson 95% CI are reported as secondary columns. The adjudicated row also carries the
human inter-annotator kappa ceiling (0.162, from external/tab_agreement.json) alongside every
method's score, so no classifier figure is read against a bare 1.0.

Output: evaluation/output/tab_evaluation_<ts>.csv

Usage:
    python datalake_tab_evaluate.py [--csv key=path ...]

Inputs are selected explicitly (datalake_artefacts.py): a method's metadata CSV is
auto-selected only if it is the ONLY full-corpus (1268-row) candidate in output/;
otherwise pin it with --csv <key>=<path>, key = prefix without "metadata_", e.g.
    --csv rule_based=output/metadata_rule_based_20261021_235159.csv \
    --csv nlp=output/metadata_nlp_20261021_235257.csv
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from datalake_artefacts import select_artefact, row_count_check, parse_pins, prefix_key

BASE_DIR = Path(__file__).parent
EXPECTED_FULL_CORPUS_SIZE = 1268

METHOD_CSV_PREFIXES = {
    "Rule-Based": "metadata_rule_based_",
    "NLP": "metadata_nlp_",
    "LLM-phi3-mini (laptop-cpu)": "metadata_llm_phi3_mini_",
    "LLM-llama3.2-1b (laptop-cpu)": "metadata_llm_llama32_1b_",
    "LLM-phi3-mini (hpc-gpu-anchor)": "metadata_cluster_phi3_mini_",
    "LLM-llama3.1-8b (hpc-gpu-scaleprobe)": "metadata_cluster_llama31_8b_",
}

DERIVATION_COLUMNS = {
    "adjudicated": "Security Confidential Only (Adjudicated)",
    "union": "Security Confidential Only (Union)",
    "intersection": "Security Confidential Only (Intersection)",
}


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tab_evaluate")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def wilson_ci(k: int, n: int):
    if n == 0:
        return None, None
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return round(lo, 3), round(hi, 3)


def read_golden() -> pd.DataFrame:
    wb = openpyxl.load_workbook(BASE_DIR / "evaluation" / "tab_golden.xlsx", data_only=True)
    ws = wb["evaluation"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df[df["File Name"].notna()]


def load_human_ceiling() -> float:
    p = BASE_DIR / "external" / "tab_agreement.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("cohens_kappa", {}).get("adjudicated_subset", {}).get("kappa")


def main():
    ap = argparse.ArgumentParser(description="Score TAB pipeline outputs against tab_golden.xlsx")
    ap.add_argument("--csv", action="append", metavar="KEY=PATH", help="pin a metadata CSV (repeatable)")
    pins = parse_pins(ap.parse_args().csv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_tab_evaluate_{ts}.txt")
    output_dir = BASE_DIR / "output"

    logger.info("=" * 70)
    logger.info("TAB EVALUATION — headline metric is Cohen's kappa, not accuracy")
    logger.info("=" * 70)

    golden = read_golden()
    human_ceiling = load_human_ceiling()
    logger.info(f"Human inter-annotator kappa ceiling (adjudicated subset): {human_ceiling}")

    rows = []
    for method, prefix in METHOD_CSV_PREFIXES.items():
        csv_path = select_artefact(output_dir, f"{prefix}*.csv", explicit=pins.get(prefix_key(prefix)),
                                   accept=row_count_check(EXPECTED_FULL_CORPUS_SIZE), what=method,
                                   option=f"--csv {prefix_key(prefix)}=")
        if not csv_path:
            logger.warning(f"{method}: no metadata CSV found — skipping")
            continue
        pred_df = pd.read_csv(csv_path)
        if len(pred_df) != EXPECTED_FULL_CORPUS_SIZE:
            logger.warning(f"{method}: {csv_path.name} has {len(pred_df)} rows, expected "
                            f"{EXPECTED_FULL_CORPUS_SIZE} — not the full TAB run, skipping")
            continue
        pred_df = pred_df[["file_name", "security_class"]].rename(columns={"file_name": "File Name"})
        merged = golden.merge(pred_df, on="File Name", how="inner")

        for deriv_name, col in DERIVATION_COLUMNS.items():
            sub = merged[merged[col].notna() & (merged[col].astype(str).str.strip() != "")]
            if len(sub) == 0:
                continue
            y_true = sub[col].tolist()
            y_pred = sub["security_class"].tolist()
            n = len(sub)
            acc = accuracy_score(y_true, y_pred)
            lo, hi = wilson_ci(int(round(acc * n)), n)
            try:
                kappa = round(float(cohen_kappa_score(y_true, y_pred, labels=["general", "confidential"])), 3)
            except Exception:
                kappa = None
            rows.append({
                "method": method, "derivation": deriv_name, "n": n,
                "cohens_kappa": kappa,
                "human_kappa_ceiling": human_ceiling if deriv_name == "adjudicated" else None,
                "accuracy": round(acc, 3), "accuracy_wilson_lo": lo, "accuracy_wilson_hi": hi,
                "source_csv": csv_path.name,
            })
            logger.info(f"{method:<16} {deriv_name:<13} n={n:<5} kappa={kappa}  "
                        f"(human ceiling={human_ceiling if deriv_name=='adjudicated' else '-'})  "
                        f"acc={acc:.3f} [{lo},{hi}]")

    out_path = out_dir / f"tab_evaluation_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"\nOutput -> {out_path}  ({len(rows)} rows)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
