# -*- coding: utf-8 -*-
"""
TAB (Text Anonymization Benchmark) adapter — external validation corpus for RO1.

Reads external/tab/echr_{train,dev,test}.json and emits:
  - tab_corpus/<doc_id>.txt           raw case text, one file per document
  - evaluation/tab_golden.xlsx        golden labels, columns matching
                                       evaluation/evaluation_template_golden.xlsx
  - external/tab_spans.csv            every entity-mention span, union of ALL
                                       annotators, tagged by annotator_id
  - external/tab_stats.json           per-split corpus statistics + class balance
                                       for each derivation

Label derivations (see external/tab_agreement.json / tab_agreement.py for the
inter-annotator agreement analysis behind this design — Step 2b of the brief):

  security_classification      descriptive only, NOT a validation metric.
                                confidential if ANY annotator marked a DIRECT or
                                QUASI identifier_type span anywhere in the doc.
                                Structurally saturated on TAB (1268/1268 confidential)
                                because every ECHR judgment has a case number/date/
                                party name — reported as a descriptive finding, not
                                used for scoring.

  security_confidential_only   PRIMARY metric. Three derivations, each a column:
    (Adjudicated) — PRIMARY.   confidential if the union of exactly the
                                quality_checked-named annotators' spans has a
                                confidential_status != NOT_CONFIDENTIAL. Blank when
                                quality_checked is empty (no adjudicated label
                                exists for that document) — such documents are
                                excluded from the primary evaluation set.
    (Union)                    confidential if ANY annotator (of all annotators on
                                the doc) independently derives confidential.
    (Intersection)              confidential only if ALL annotators independently
                                derive confidential.

  data_class_category, linked_to_customer: left blank — TAB has no information-type
  or customer-linkage labels, and none is invented.

Usage:
    python external/tab_adapter.py --limit 6 --seed 42        # smoke test
    python external/tab_adapter.py                            # full 1268 docs
"""
import argparse
import csv
import json
import logging
import random
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook

BASE_DIR = Path(__file__).parent.parent
TAB_DIR = BASE_DIR / "external" / "tab"
SPLITS = ["train", "dev", "test"]

IDENTIFYING_TYPES = {"DIRECT", "QUASI"}  # brief's DIRECT_ID / QUASI_ID, confirmed with user
NOT_CONFIDENTIAL = "NOT_CONFIDENTIAL"


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tab_adapter")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_split(split: str) -> list:
    path = TAB_DIR / f"echr_{split}.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for doc in data:
        doc["_split"] = split
    return data


def annotator_spans(doc: dict, annotator: str) -> list:
    return doc["annotations"].get(annotator, {}).get("entity_mentions", [])


def all_spans(doc: dict) -> list:
    rows = []
    for ann in doc["annotations"].keys():
        for em in annotator_spans(doc, ann):
            rows.append({**em, "annotator_id": ann})
    return rows


def qc_spans(doc: dict) -> list:
    qc = doc.get("quality_checked") or []
    rows = []
    for ann in qc:
        if ann in doc["annotations"]:
            for em in annotator_spans(doc, ann):
                rows.append({**em, "annotator_id": ann})
    return rows


def security_classification_label(doc: dict) -> str:
    """Descriptive only — see module docstring. Not used for scoring."""
    spans = all_spans(doc)
    return "confidential" if any(s.get("identifier_type") in IDENTIFYING_TYPES for s in spans) else "general"


def conf_only_from_spans(spans: list) -> str:
    return "confidential" if any(
        s.get("confidential_status") not in (None, NOT_CONFIDENTIAL) for s in spans
    ) else "general"


def per_annotator_conf_only(doc: dict) -> dict:
    return {ann: conf_only_from_spans(annotator_spans(doc, ann)) for ann in doc["annotations"].keys()}


def conf_only_adjudicated(doc: dict):
    """None (blank) when quality_checked is empty — no adjudicated label exists."""
    if not (doc.get("quality_checked") or []):
        return None
    return conf_only_from_spans(qc_spans(doc))


def conf_only_union(doc: dict) -> str:
    labels = per_annotator_conf_only(doc)
    return "confidential" if any(v == "confidential" for v in labels.values()) else "general"


def conf_only_intersection(doc: dict) -> str:
    labels = per_annotator_conf_only(doc)
    return "confidential" if all(v == "confidential" for v in labels.values()) else "general"


def pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(BASE_DIR))
    ap.add_argument("--limit", type=int, default=None, help="sample N documents total (across splits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", default=",".join(SPLITS))
    a = ap.parse_args()

    repo = Path(a.repo)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = repo / "evaluation" / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir / f"log_tab_adapter_{ts}.txt")

    logger.info("=" * 70)
    logger.info("TAB ADAPTER")
    logger.info(f"seed={a.seed}  limit={a.limit}  splits={a.splits}")
    logger.info("=" * 70)

    splits = a.splits.split(",")
    all_docs = []
    for split in splits:
        docs = load_split(split)
        docs.sort(key=lambda d: d["doc_id"])  # deterministic order before shuffling
        logger.info(f"Loaded {split}: {len(docs)} documents")
        all_docs.extend(docs)

    if a.limit is not None:
        rng = random.Random(a.seed)
        rng.shuffle(all_docs)
        all_docs = all_docs[: a.limit]
        logger.info(f"Sampled {len(all_docs)} documents (seed={a.seed}) for this run")

    corpus_dir = repo / "tab_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    stale = list(corpus_dir.glob("*.txt"))
    if stale:
        logger.info(f"Clearing {len(stale)} stale file(s) from a previous run in {corpus_dir}")
        for f in stale:
            f.unlink()

    golden_rows = []
    span_rows = []
    primary_eval_count = 0
    stats_by_split = {s: {"doc_ids": [], "lengths": [], "sec_class": Counter(),
                          "conf_adj": Counter(), "conf_union": Counter(), "conf_inter": Counter(),
                          "span_types": Counter(), "qc_nonempty": 0}
                       for s in splits}

    for doc in all_docs:
        doc_id = doc["doc_id"]
        split = doc["_split"]
        text = doc["text"]
        qc = doc.get("quality_checked") or []

        (corpus_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

        sec_class = security_classification_label(doc)
        adj = conf_only_adjudicated(doc)
        union_lbl = conf_only_union(doc)
        inter_lbl = conf_only_intersection(doc)
        if adj is not None:
            primary_eval_count += 1

        golden_rows.append({
            "File Name": f"{doc_id}.txt",
            "File Type": "text",
            "Data Class Category": "",
            "Security Classification": sec_class,
            "Security Confidential Only (Adjudicated)": adj if adj is not None else "",
            "Security Confidential Only (Union)": union_lbl,
            "Security Confidential Only (Intersection)": inter_lbl,
            "Linked To Customer": "",
            "Total Time Taken Seconds": "",
            "Notes": f"TAB split={split}; n_annotators={len(doc['annotations'])}; quality_checked={','.join(qc)}",
            "Content Preview": text[:120].replace("\n", " "),
            "Classification Rationale": "",
        })

        spans = all_spans(doc)
        for s in spans:
            span_rows.append({
                "doc_id": doc_id,
                "start_offset": s["start_offset"],
                "end_offset": s["end_offset"],
                "span_text": s["span_text"],
                "entity_type": s["entity_type"],
                "identifier_type": s.get("identifier_type", ""),
                "confidential_status": s.get("confidential_status", ""),
                "annotator_id": s["annotator_id"],
            })

        st = stats_by_split[split]
        st["doc_ids"].append(doc_id)
        st["lengths"].append(len(text))
        st["sec_class"][sec_class] += 1
        st["conf_adj"][adj if adj is not None else "(no quality_checked)"] += 1
        st["conf_union"][union_lbl] += 1
        st["conf_inter"][inter_lbl] += 1
        if qc:
            st["qc_nonempty"] += 1
        for s in spans:
            st["span_types"][s["entity_type"]] += 1

    # ---- evaluation/tab_golden.xlsx (match evaluation_template_golden.xlsx layout) ----
    golden_path = repo / "evaluation" / "tab_golden.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "evaluation"
    columns = ["File Name", "File Type", "Data Class Category", "Security Classification",
               "Security Confidential Only (Adjudicated)", "Security Confidential Only (Union)",
               "Security Confidential Only (Intersection)", "Linked To Customer",
               "Total Time Taken Seconds", "Notes", "Content Preview", "Classification Rationale"]
    ws.append([f"GOLDEN LABELS — TAB (ECHR) corpus | {len(golden_rows)} files | derived, not expert-authored"]
               + [None] * (len(columns) - 1))
    ws.append(columns)
    for row in golden_rows:
        ws.append([row[c] for c in columns])
    wb.save(golden_path)
    logger.info(f"Golden XLSX  -> {golden_path}  ({len(golden_rows)} rows, "
                f"{primary_eval_count} with an adjudicated label)")

    # ---- external/tab_spans.csv ----
    spans_path = repo / "external" / "tab_spans.csv"
    with open(spans_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "start_offset", "end_offset", "span_text",
                                                "entity_type", "identifier_type", "confidential_status",
                                                "annotator_id"])
        writer.writeheader()
        writer.writerows(span_rows)
    logger.info(f"Spans CSV    -> {spans_path}  ({len(span_rows)} rows, union of all annotators)")

    # ---- external/tab_stats.json ----
    stats = {"generated": ts, "seed": a.seed, "limit": a.limit, "splits": {}}
    total_primary = 0
    for split in splits:
        st = stats_by_split[split]
        n = len(st["doc_ids"])
        lengths = st["lengths"]
        total_primary += st["qc_nonempty"]
        stats["splits"][split] = {
            "document_count": n,
            "primary_evaluation_set_size": st["qc_nonempty"],
            "class_balance": {
                "security_classification": dict(st["sec_class"]),
                "security_confidential_only_adjudicated": dict(st["conf_adj"]),
                "security_confidential_only_union": dict(st["conf_union"]),
                "security_confidential_only_intersection": dict(st["conf_inter"]),
            },
            "character_length": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": round(statistics.fmean(lengths), 1) if lengths else None,
                "median": statistics.median(lengths) if lengths else None,
            } if lengths else {},
            "span_counts_by_type": dict(st["span_types"]),
        }
    stats["primary_evaluation_set_total"] = total_primary
    stats["note"] = ("security_classification is descriptive only (saturated on TAB, not a "
                      "validation metric). security_confidential_only_adjudicated is the primary "
                      "metric, scored only on documents with a non-empty quality_checked list; "
                      "union/intersection are sensitivity analyses over all documents. See "
                      "external/tab_agreement.json for the inter-annotator agreement analysis "
                      "(Cohen's kappa, document- vs span-level disagreement) behind this design.")
    stats_path = repo / "external" / "tab_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"Stats JSON   -> {stats_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Documents processed         : {len(all_docs)}")
    logger.info(f"Corpus files written         : {len(golden_rows)}  -> {corpus_dir}")
    logger.info(f"Span rows written            : {len(span_rows)}")
    logger.info(f"Primary evaluation set (adjudicated label present): {primary_eval_count} / {len(all_docs)}")
    for split in splits:
        st = stats_by_split[split]
        logger.info(f"[{split}] docs={len(st['doc_ids'])}  qc_nonempty={st['qc_nonempty']}  "
                    f"sec_class={dict(st['sec_class'])}  "
                    f"conf_adjudicated={dict(st['conf_adj'])}  "
                    f"conf_union={dict(st['conf_union'])}  conf_intersection={dict(st['conf_inter'])}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
