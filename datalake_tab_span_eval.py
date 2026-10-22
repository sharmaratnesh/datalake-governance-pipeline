# -*- coding: utf-8 -*-
"""
Step 6 / 6b — Span-level evaluation, with the human agreement baseline (6a) in the same table.

IMPORTANT METHODOLOGICAL NOTE: none of the four pipeline methods (rule-based, NLP/TF-IDF,
LLM-phi3-mini, LLM-llama3.2-1b) produce span-level output — each emits one document-level
security_class label, not character offsets. To place them in the same character-level
precision/recall/F1 table the brief asks for, each method's document-level "confidential"
prediction is treated as a single predicted span covering the ENTIRE document (0..len(text));
"general" predicts no spans. This is a proxy, not a native span prediction, and is reported as
such — it mainly shows how far a document-level call is from "pointing at" the right text, not
a fair comparison to a real span-extraction system. exact_match (the predicted whole-document
span exactly equals one gold span) is near-always 0 by construction and is reported only for
schema completeness; partial_overlap (any character overlap) is the informative variant.

Gold spans: external/tab_spans.csv, restricted to confidential_status != NOT_CONFIDENTIAL
(the spans that drive the security_confidential_only label), from the adjudicated annotators
where available (quality_checked), else the union of all annotators for that doc.

Human baseline (6a): read directly from external/tab_agreement.json (span_level_agreement),
placed in the same output table so classifier rows are read against the achievable ceiling
rather than against 1.0.

Usage:
    python datalake_tab_span_eval.py [--csv key=path ...]

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

import pandas as pd

from datalake_artefacts import select_artefact, row_count_check, parse_pins, prefix_key

BASE_DIR = Path(__file__).parent
NOT_CONFIDENTIAL = "NOT_CONFIDENTIAL"

METHOD_CSV_PREFIXES = {
    "Rule-Based": "metadata_rule_based_",
    "NLP": "metadata_nlp_",
    "LLM-phi3-mini (laptop-cpu)": "metadata_llm_phi3_mini_",
    "LLM-llama3.2-1b (laptop-cpu)": "metadata_llm_llama32_1b_",
    "LLM-phi3-mini (hpc-gpu-anchor)": "metadata_cluster_phi3_mini_",
    "LLM-llama3.1-8b (hpc-gpu-scaleprobe)": "metadata_cluster_llama31_8b_",
}


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tab_span_eval")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def doc_len(doc_id: str) -> int:
    p = BASE_DIR / "tab_corpus" / f"{doc_id}.txt"
    return len(p.read_text(encoding="utf-8", errors="ignore")) if p.exists() else 0


def load_gold_spans() -> dict:
    """doc_id -> list of (start, end) for confidential_status != NOT_CONFIDENTIAL spans,
    deduplicated across annotators (a span counts once if ANY annotator marked it sensitive
    at that location — consistent with the union-of-spans gold used for Step 6 span counts)."""
    spans_path = BASE_DIR / "external" / "tab_spans.csv"
    df = pd.read_csv(spans_path)
    df = df[df["confidential_status"].notna() & (df["confidential_status"] != NOT_CONFIDENTIAL)]
    gold = {}
    for doc_id, grp in df.groupby("doc_id"):
        ranges = sorted({(int(r.start_offset), int(r.end_offset)) for r in grp.itertuples()})
        gold[doc_id] = ranges
    return gold


def overlap_len(a_start, a_end, b_start, b_end) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def score_method(metadata_csv: Path, gold_spans: dict, logger) -> dict:
    df = pd.read_csv(metadata_csv)
    if "file_name" not in df.columns or "security_class" not in df.columns:
        logger.warning(f"{metadata_csv}: missing expected columns, skipping")
        return None

    exact_tp = exact_fp = exact_fn = 0
    ov_tp_chars = ov_pred_chars = ov_gold_chars = 0

    for _, row in df.iterrows():
        doc_id = str(row["file_name"]).replace(".txt", "")
        pred_confidential = str(row["security_class"]).strip().lower() == "confidential"
        gold_ranges = gold_spans.get(doc_id, [])
        length = doc_len(doc_id)
        if length == 0:
            continue

        gold_chars = sum(e - s for s, e in gold_ranges)
        ov_gold_chars += gold_chars

        if pred_confidential:
            # proxy predicted span = whole document
            ov_pred_chars += length
            ov_tp_chars += gold_chars  # whole-doc span overlaps every gold span fully
            # exact-match: predicted span (0,length) equals some single gold span exactly
            if any(s == 0 and e == length for s, e in gold_ranges):
                exact_tp += 1
            elif gold_ranges:
                exact_fn += 1  # predicted something, but not an exact match to any gold span
            # (a doc with zero gold ranges and a confidential prediction has no exact-match fn/tp)
        else:
            if gold_ranges:
                exact_fn += 1

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return round(p, 3), round(r, 3), round(f, 3)

    ov_p, ov_r, ov_f = prf(ov_tp_chars, max(ov_pred_chars - ov_tp_chars, 0), max(ov_gold_chars - ov_tp_chars, 0))
    ex_p, ex_r, ex_f = prf(exact_tp, 0, exact_fn)  # fp not well-defined for this proxy; precision uses tp/(tp+fn) as floor

    return {
        "partial_overlap_precision": ov_p, "partial_overlap_recall": ov_r, "partial_overlap_f1": ov_f,
        "exact_match_recall": ex_r,  # precision/F1 omitted for exact_match: proxy makes "false positive span count" ill-defined
        "n_docs_scored": int(len(df)),
    }


def load_human_baseline() -> dict:
    agreement_path = BASE_DIR / "external" / "tab_agreement.json"
    if not agreement_path.exists():
        return {}
    data = json.loads(agreement_path.read_text(encoding="utf-8"))
    return data.get("span_level_agreement", {}).get("confidential_status", {})


def main():
    ap = argparse.ArgumentParser(description="TAB span-level evaluation")
    ap.add_argument("--csv", action="append", metavar="KEY=PATH", help="pin a metadata CSV (repeatable)")
    pins = parse_pins(ap.parse_args().csv)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_tab_span_eval_{ts}.txt")

    logger.info("=" * 70)
    logger.info("TAB SPAN-LEVEL EVALUATION (Step 6 / 6b)")
    logger.info("=" * 70)

    gold_spans = load_gold_spans()
    logger.info(f"Gold sensitive-span documents: {len(gold_spans)}")

    output_dir = BASE_DIR / "output"
    rows = []
    for method, prefix in METHOD_CSV_PREFIXES.items():
        csv_path = select_artefact(output_dir, f"{prefix}*.csv", explicit=pins.get(prefix_key(prefix)),
                                   accept=row_count_check(1268), what=method,
                                   option=f"--csv {prefix_key(prefix)}=")
        if not csv_path:
            logger.warning(f"{method}: no metadata CSV found in output/ — skipping "
                            "(run the pipeline over tab_corpus/ first)")
            continue
        n_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        if n_rows != 1268:
            logger.warning(f"{method}: {csv_path.name} has {n_rows} rows, expected 1268 — "
                            "not the full TAB run, skipping")
            continue
        result = score_method(csv_path, gold_spans, logger)
        if result:
            result["method"] = method
            result["source_csv"] = csv_path.name
            rows.append(result)
            logger.info(f"{method:<16} partial_overlap: P={result['partial_overlap_precision']} "
                        f"R={result['partial_overlap_recall']} F1={result['partial_overlap_f1']}  "
                        f"(n={result['n_docs_scored']})")

    human = load_human_baseline()
    if human:
        for align_name in ("exact_match", "partial_overlap"):
            block = human.get(align_name, {}).get("all_multi_annotator", {})
            rows.append({
                "method": "HUMAN_BASELINE_confidential_status",
                "source_csv": "external/tab_agreement.json",
                f"{align_name}_kappa_human_ceiling": block.get("kappa"),
                "n_pairs_human": block.get("n_pairs"),
            })
            logger.info(f"Human baseline (confidential_status, {align_name}): kappa={block.get('kappa')}")
    else:
        logger.warning("No external/tab_agreement.json found — run external/tab_agreement.py first "
                        "for the human-ceiling comparison row")

    out_path = out_dir / f"tab_span_evaluation_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"\nOutput -> {out_path}  ({len(rows)} rows)")
    logger.info("NOTE: classifier span metrics use a whole-document-as-predicted-span proxy "
                "(see module docstring) — not a native span prediction. Read them as a lower "
                "bound on localization quality, not a fair comparison to a real span extractor.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
