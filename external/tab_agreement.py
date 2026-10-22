# -*- coding: utf-8 -*-
"""
Inter-annotator agreement analysis for TAB's security_confidential_only label (brief Step 2b).

Always runs over the FULL 1268-document corpus, independent of any tab_adapter.py sampling:
this establishes the human-agreement ceiling the overnight automated run will be read against,
so it must reflect the whole label population, not a smoke-test slice.

Emits external/tab_agreement.json with:
  1. Document-level vs span-level disagreement on security_confidential_only, computed
     separately, plus the inflation factor between them (a document label is an OR over
     spans, so one differing span flips the whole document).
  2. Both figures split by quality_checked present/absent.
  3. Cohen's kappa on the derived document label (pooled pairwise across annotators),
     for the full multi-annotator population and for the "adjudicated subset" (documents
     with >=2 quality_checked annotators) — the human ceiling for the primary metric.

Usage:
    python external/tab_agreement.py
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

BASE_DIR = Path(__file__).parent.parent
TAB_DIR = BASE_DIR / "external" / "tab"
SPLITS = ["train", "dev", "test"]
NOT_CONFIDENTIAL = "NOT_CONFIDENTIAL"
KAPPA_FLOOR = 0.4


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tab_agreement")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_all_docs() -> list:
    docs = []
    for split in SPLITS:
        with open(TAB_DIR / f"echr_{split}.json", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            d["_split"] = split
        docs.extend(data)
    return docs


def annotator_spans(doc: dict, ann: str) -> list:
    return doc["annotations"].get(ann, {}).get("entity_mentions", [])


def sensitive_spans(doc: dict, ann: str) -> list:
    return [s for s in annotator_spans(doc, ann) if s.get("confidential_status") not in (None, NOT_CONFIDENTIAL)]


def annotator_label(doc: dict, ann: str) -> str:
    return "confidential" if sensitive_spans(doc, ann) else "general"


def overlaps(a: dict, b: dict) -> bool:
    return a["start_offset"] < b["end_offset"] and b["start_offset"] < a["end_offset"]


def span_level_stats(doc: dict, annotators: list) -> tuple:
    """For every sensitive span, is it corroborated by an overlapping sensitive span from a
    DIFFERENT annotator in the same doc? Returns (n_spans, n_uncorroborated)."""
    tagged = []
    for ann in annotators:
        for s in sensitive_spans(doc, ann):
            tagged.append({**s, "_annotator": ann})
    n_uncorroborated = 0
    for i, s in enumerate(tagged):
        corroborated = any(
            other["_annotator"] != s["_annotator"] and overlaps(s, other)
            for j, other in enumerate(tagged) if j != i
        )
        if not corroborated:
            n_uncorroborated += 1
    return len(tagged), n_uncorroborated


def doc_level_disagrees(doc: dict, annotators: list) -> bool:
    labels = {annotator_label(doc, a) for a in annotators}
    return len(labels) > 1


def pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def summarize_group(group: list) -> dict:
    n = len(group)
    if n == 0:
        return {"n_documents": 0}
    doc_dis = sum(1 for d in group if doc_level_disagrees(d, list(d["annotations"].keys())))
    total_spans = total_uncorr = 0
    for d in group:
        ns, nu = span_level_stats(d, list(d["annotations"].keys()))
        total_spans += ns
        total_uncorr += nu
    doc_rate = pct(doc_dis, n)
    span_rate = pct(total_uncorr, total_spans)
    inflation = round(doc_rate / span_rate, 2) if (doc_rate is not None and span_rate) else None
    return {
        "n_documents": n,
        "document_level_disagreement_pct": doc_rate,
        "span_level_disagreement_pct": span_rate,
        "total_sensitive_spans": total_spans,
        "uncorroborated_sensitive_spans": total_uncorr,
        "inflation_factor_doc_over_span": inflation,
    }


def collect_pairs(docs_with_annotators: list) -> list:
    """docs_with_annotators: [(doc, [annotator_ids])]. Pooled (label_a, label_b) over every
    annotator pair within each doc."""
    pairs = []
    for doc, anns in docs_with_annotators:
        labels = [annotator_label(doc, a) for a in anns]
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                pairs.append((labels[i], labels[j]))
    return pairs


def pooled_kappa(pairs: list, labels=None):
    if len(pairs) < 2:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    try:
        k = cohen_kappa_score(a, b, labels=labels or ["general", "confidential"])
        return round(float(k), 3)
    except Exception:
        return None


IDENTIFIER_TYPE_LABELS = ["DIRECT", "QUASI", "NO_MASK"]
CONFIDENTIAL_STATUS_LABELS = ["NOT_CONFIDENTIAL", "ETHNIC", "POLITICS", "HEALTH", "BELIEF", "SEX"]


def span_overlap_len(a: dict, b: dict) -> int:
    return max(0, min(a["end_offset"], b["end_offset"]) - max(a["start_offset"], b["start_offset"]))


def _matches(s: dict, t: dict, exact: bool) -> bool:
    if exact:
        return s["start_offset"] == t["start_offset"] and s["end_offset"] == t["end_offset"]
    return span_overlap_len(s, t) > 0


def align_full(spans_a: list, spans_b: list, exact: bool) -> list:
    """Pairs covering EVERY span from both A and B exactly once. A span with no counterpart
    in the other annotator's set is paired with None (the caller backfills a default/negative
    label) — this is necessary because a missed/extra span is itself a disagreement, and
    silently dropping unmatched spans (as a naive overlap-join does) would hide exactly the
    cases that drive document-level disagreement, inflating apparent span-level kappa."""
    pairs = []
    matched_b_ids = set()
    for s in spans_a:
        best, best_ov = None, -1
        for t in spans_b:
            if exact:
                if _matches(s, t, True):
                    best, best_ov = t, 1
                    break
            else:
                ov = span_overlap_len(s, t)
                if ov > best_ov:
                    best, best_ov = t, ov
        if best is not None and best_ov > 0:
            pairs.append((s, best))
            matched_b_ids.add(id(best))
        else:
            pairs.append((s, None))
    for t in spans_b:
        if id(t) in matched_b_ids:
            continue
        if any(_matches(t, s, exact) for s in spans_a):
            continue  # has a counterpart (matched to a different best-pick above); not double-counted
        pairs.append((None, t))
    return pairs


def span_field_pairs(docs: list, field: str, exact: bool, default_value: str) -> list:
    """Pooled (value_a, value_b) pairs for `field` (identifier_type / confidential_status),
    over every annotator-pair's aligned spans in each multi-annotator doc. One direction
    (i, j) per unordered annotator pair, to avoid double-counting the same span event. A span
    with no counterpart in the other annotator's set is paired against `default_value`
    (the field's negative/no-mention category), not dropped."""
    pairs = []
    for doc in docs:
        anns = list(doc["annotations"].keys())
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                spans_a = annotator_spans(doc, anns[i])
                spans_b = annotator_spans(doc, anns[j])
                for sa, sb in align_full(spans_a, spans_b, exact):
                    va = sa.get(field) if sa is not None else default_value
                    vb = sb.get(field) if sb is not None else default_value
                    pairs.append((va, vb))
    return pairs


def span_level_agreement_section(multi_docs: list) -> dict:
    qc_docs = [d for d in multi_docs if d.get("quality_checked")]
    nonqc_docs = [d for d in multi_docs if not d.get("quality_checked")]

    def block(field, labels, default_value):
        out = {}
        for align_name, exact in (("exact_match", True), ("partial_overlap", False)):
            all_pairs = span_field_pairs(multi_docs, field, exact, default_value)
            qc_pairs = span_field_pairs(qc_docs, field, exact, default_value)
            nonqc_pairs = span_field_pairs(nonqc_docs, field, exact, default_value)
            out[align_name] = {
                "all_multi_annotator": {"kappa": pooled_kappa(all_pairs, labels), "n_pairs": len(all_pairs)},
                "quality_checked": {"kappa": pooled_kappa(qc_pairs, labels), "n_pairs": len(qc_pairs)},
                "non_quality_checked": {"kappa": pooled_kappa(nonqc_pairs, labels), "n_pairs": len(nonqc_pairs)},
            }
        return out

    return {
        "identifier_type": block("identifier_type", IDENTIFIER_TYPE_LABELS, "NO_MASK"),
        "confidential_status": block("confidential_status", CONFIDENTIAL_STATUS_LABELS, "NOT_CONFIDENTIAL"),
        "note": ("Cohen's kappa over span pairs between every annotator pair in each multi-"
                 "annotator document (one direction per unordered pair, to avoid double-"
                 "counting). A span with no counterpart in the other annotator's set is paired "
                 "against that field's own negative category (NO_MASK / NOT_CONFIDENTIAL) rather "
                 "than dropped — a missed or extra span is itself a disagreement, and dropping it "
                 "would inflate kappa by only scoring the easy cases where both annotators agreed "
                 "a span existed at all. exact_match pairs spans with identical character offsets only; "
                 "partial_overlap pairs each span with the other annotator's max-overlapping span. "
                 "identifier_type (DIRECT/QUASI/NO_MASK) is compared against confidential_status "
                 "(NOT_CONFIDENTIAL + 5 special categories) over the SAME span alignment, so any "
                 "kappa gap between the two fields is attributable to the label, not the alignment."),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = BASE_DIR / "evaluation" / "output"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir / f"log_tab_agreement_{ts}.txt")

    logger.info("=" * 70)
    logger.info("TAB AGREEMENT ANALYSIS — security_confidential_only")
    logger.info("=" * 70)

    docs = load_all_docs()
    multi = [d for d in docs if len(d["annotations"]) > 1]
    qc_multi = [d for d in multi if d.get("quality_checked")]
    nonqc_multi = [d for d in multi if not d.get("quality_checked")]

    overall = summarize_group(multi)
    qc_group = summarize_group(qc_multi)
    nonqc_group = summarize_group(nonqc_multi)

    all_pairs = collect_pairs([(d, list(d["annotations"].keys())) for d in multi])
    kappa_full = pooled_kappa(all_pairs)

    qc_multi_rater_docs = [(d, d["quality_checked"]) for d in docs if len(d.get("quality_checked") or []) >= 2]
    qc_pairs = collect_pairs(qc_multi_rater_docs)
    kappa_adjudicated = pooled_kappa(qc_pairs)

    primary_eval_size = sum(1 for d in docs if d.get("quality_checked"))
    primary_by_split = {s: sum(1 for d in docs if d["_split"] == s and d.get("quality_checked")) for s in SPLITS}

    result = {
        "generated": ts,
        "metric": "security_confidential_only",
        "primary_evaluation_set": {
            "definition": "documents with non-empty quality_checked",
            "size": primary_eval_size,
            "by_split": primary_by_split,
        },
        "disagreement": {
            "all_multi_annotator_documents": overall,
            "quality_checked_multi_annotator_documents": qc_group,
            "non_quality_checked_multi_annotator_documents": nonqc_group,
            "note": ("document_level_disagreement_pct: % of multi-annotator docs where annotators' "
                     "independently-derived document labels disagree. span_level_disagreement_pct: "
                     "% of individual sensitive spans with no overlapping corroborating span from a "
                     "different annotator in the same doc. inflation_factor_doc_over_span quantifies "
                     "how much the doc-level OR-of-spans disjunction inflates apparent disagreement "
                     "relative to the span level."),
        },
        "cohens_kappa": {
            "full_multi_annotator_population": {
                "kappa": kappa_full,
                "n_pairwise_comparisons": len(all_pairs),
                "n_documents": len(multi),
            },
            "adjudicated_subset": {
                "kappa": kappa_adjudicated,
                "n_pairwise_comparisons": len(qc_pairs),
                "n_documents": len(qc_multi_rater_docs),
                "note": ("Pairwise Cohen's kappa pooled across quality_checked-annotator pairs in "
                         "documents with >=2 quality_checked annotators — the human agreement "
                         "ceiling for the primary evaluation set."),
            },
        },
    }

    result["span_level_agreement"] = span_level_agreement_section(multi)

    if kappa_adjudicated is not None and kappa_adjudicated < KAPPA_FLOOR:
        warning = (
            f"Kappa={kappa_adjudicated} is below the conventional {KAPPA_FLOOR} 'fair agreement' "
            "threshold. The primary evaluation set's security_confidential_only label is noisy: "
            "automated-method scores against it measure agreement with an inconsistently-applied "
            "human label, not ground-truth correctness. State this plainly in the paper and adjust "
            "the external-validation framing accordingly."
        )
        result["cohens_kappa"]["adjudicated_subset"]["warning"] = warning
        logger.warning(warning)

    out_path = BASE_DIR / "external" / "tab_agreement.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info(f"Primary evaluation set       : {primary_eval_size}  (by split: {primary_by_split})")
    logger.info(f"All multi-annotator docs     : {overall}")
    logger.info(f"  quality_checked subset     : {qc_group}")
    logger.info(f"  non-quality_checked subset : {nonqc_group}")
    logger.info(f"Cohen's kappa (full population)    : {kappa_full}  "
                f"(n_pairs={len(all_pairs)}, n_docs={len(multi)})")
    logger.info(f"Cohen's kappa (adjudicated subset) : {kappa_adjudicated}  "
                f"(n_pairs={len(qc_pairs)}, n_docs={len(qc_multi_rater_docs)})")
    sla = result["span_level_agreement"]
    logger.info("Span-level agreement (6a):")
    for field in ("identifier_type", "confidential_status"):
        for align_name in ("exact_match", "partial_overlap"):
            b = sla[field][align_name]["all_multi_annotator"]
            logger.info(f"  {field:<20} {align_name:<16} kappa={b['kappa']}  n_pairs={b['n_pairs']}")
    logger.info(f"Output -> {out_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
