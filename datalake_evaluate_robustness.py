"""
datalake_evaluate_robustness.py — Reviewer-requested robustness analyses (RO1)

Implements the reviewers' items 1, 2, 3, 5 and 6:
  1. Wilson 95% CI on headline accuracies (category/security/overall), one paired
     McNemar test on the security decision between the two best automated methods.
  2. Phase 1 (7 manually authored files) vs Phase 2 (48 programmatically generated
     files) accuracy breakdown, per method.
  3. Per-file-type (.txt/.csv/.json/.wav/.jpg) accuracy breakdown, per method.
  5. Generator-intended labels vs expert-consensus golden labels, for the 48 Phase 2
     files whose intended category/security follow deterministically from which
     generation function produced them.
  6. LLaMA 3.2 1B failure taxonomy: a strict 5-way partition of all 55 files into
     placeholder echo / schema failure / invalid enum / entity-extraction error / correct.

Item 4 (OCR field accuracy + audio WER against a ground-truth manifest) and the
reproducibility decisions in item 7 (Ollama seed, public repository) are NOT
implemented here — both require an explicit go-ahead from the paper's author first,
per the brief.

This script does NOT modify datalake_evaluate_all.py and does NOT re-run any of the
four classification pipelines. It reads:
  evaluation/output/evaluation_per_file_<ts>.csv   (--per-file: golden + all methods)
  output/log_llm_llama32_1b_<ts>.txt               (--llama-log: raw LLM responses)
Each is auto-selected only if exactly one 55-file data_swamp candidate exists; otherwise
it must be pinned. The paper's analysis used
  --per-file  evaluation/output/evaluation_per_file_20260723_155341.csv
  --llama-log output/log_llm_llama32_1b_20260723_134633.txt

Outputs (written to evaluation/output/):
  evaluation_confidence_intervals_<ts>.csv
  evaluation_mcnemar_security_<ts>.csv
  evaluation_phase_split_<ts>.csv
  evaluation_modality_split_<ts>.csv
  evaluation_generator_vs_golden_<ts>.csv
  evaluation_llama1b_failure_taxonomy_<ts>.csv
  log_evaluate_robustness_<ts>.txt

Run:
  python datalake_evaluate_robustness.py --per-file <csv> --llama-log <txt>
"""

import argparse
import ast
import io
import logging
import re
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from datalake_artefacts import select_artefact, row_count_check

warnings.filterwarnings("ignore", category=UserWarning)

# -- Paths ----------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
EVAL_DIR   = BASE_DIR / "evaluation"
EVAL_OUT   = EVAL_DIR / "output"
OUTPUT_DIR = BASE_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

AUTOMATED_METHODS = ["rule_based", "nlp", "llm_llama3.2_1b", "llm_phi3_mini"]
METHOD_DISPLAY = {
    "rule_based": "Rule-Based",
    "nlp": "NLP",
    "llm_llama3.2_1b": "LLM-LLaMA3.2:1b",
    "llm_phi3_mini": "LLM-Phi3:mini",
}

# The 7 manually authored Phase 1 files (confirmed against the current data_swamp/
# listing and Section 3.2.1 of the paper). All other files are Phase 2.
PHASE1_FILES = {
    "DL.jpg",
    "identity driving license.jpg",
    "Transaction 1.txt",
    "Transactions.csv",
    "Recording.wav",
    "call2.wav",
    "jane complaint.wav",
}

# Generator-intended (category, security) per Phase 2 filename prefix, read directly
# from the generation functions in generate_data_swamp.py (make_transaction_txt,
# make_complaint_txt, etc.). `ambiguous=True` rows are cases where the generator's
# own field content plausibly supports more than one category label - the intended
# label is still given (the closer / more conservative reading), but the disagreement
# analysis in item 5 should not be read as a pipeline "error" for these files without
# also reading the note.
PHASE2_INTENDED_LABELS = {
    "transaction_":       ("financial_transactions", "confidential", False, ""),
    "complaint_":         ("service_requests", "confidential", False, ""),
    "internal_note_":     ("service_requests", "confidential", True,
                            "Internal bank employee note about an account action (KYC "
                            "update, fraud alert raised, address change). Could be read "
                            "as service_requests (documents a customer-account service "
                            "action) or unknown (it is an internal record, not a "
                            "customer-initiated request)."),
    "customer_email_":    ("service_requests", "confidential", False, ""),
    "transactions_":      ("financial_transactions", "confidential", False, ""),
    "customers_":         ("identities", "confidential", False, ""),
    "fraud_flags_":       ("unknown", "confidential", True,
                            "Fraud-flag list keyed by account_id/alert_id only, no "
                            "customer name. Could equally be read as "
                            "financial_transactions (transaction-anomaly monitoring) or "
                            "service_requests (fraud handling); 'unknown' is used here as "
                            "the most conservative default since no single theme "
                            "dominates the field content."),
    "customer_profile_":  ("identities", "confidential", False, ""),
    "transaction_record_": ("financial_transactions", "confidential", False, ""),
    "fraud_alert_":       ("service_requests", "confidential", True,
                            "Fraud alert JSON includes customer_name and account_number "
                            "(customer/account-specific, like a service request), but "
                            "risk_score/recommended_action framing also resembles "
                            "financial-transaction monitoring. Treated as "
                            "service_requests here since it centres on a specific "
                            "customer's flagged event requiring action."),
    "kyc_report_":        ("identities", "confidential", False, ""),
    "call_recording_":    ("service_requests", "confidential", False, ""),
    "voice_complaint_":   ("service_requests", "confidential", False, ""),
    "driving_licence_":   ("identities", "confidential", False, ""),
    "passport_":          ("identities", "confidential", False, ""),
    "utility_bill_":      ("financial_transactions", "confidential", True,
                            "Utility bill has billing/amount-due fields "
                            "(financial-transaction-like) but is also commonly used as a "
                            "proof-of-address identity-verification document "
                            "(KYC-adjacent); ambiguous between financial_transactions "
                            "and identities."),
    "bank_statement_":    ("financial_transactions", "confidential", False, ""),
}

VALID_CATEGORIES = {"identities", "service_requests", "financial_transactions", "unknown"}
VALID_SECURITY   = {"confidential", "general"}


# -- Logging ----------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("evaluate_robustness")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def primary_corpus_log(path: Path) -> str | None:
    """accept() for pipeline logs: must be a run over the 55-file data_swamp/."""
    head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    if "Files found in data swamp: 55" not in head or "data_swamp" not in head:
        return "not a 55-file data_swamp run"
    return None


def required(path: Path | None, what: str) -> Path:
    if path is None:
        raise FileNotFoundError(f"No {what} found")
    return path


def file_type_of(file_name: str) -> str:
    return Path(file_name).suffix.lower().lstrip(".")


def intended_label_for(file_name: str):
    """Return (category, security, ambiguous, note) for a Phase 2 file, matched by
    the longest prefix (before the trailing digit run and extension) that appears in
    PHASE2_INTENDED_LABELS. Returns None for Phase 1 files or unrecognised names."""
    m = re.match(r"^(.+?)\d+\.\w+$", file_name)
    if not m:
        return None
    prefix = m.group(1)
    return PHASE2_INTENDED_LABELS.get(prefix)


# -- Item 1: Wilson CI + McNemar --------------------------------------------------
def human_majority_correct(df: pd.DataFrame, task: str) -> pd.Series:
    """Per-file boolean: did the majority of the 3 participants match the golden
    label for `task` ('category' or 'security')? Ties (impossible with 3 voters)
    would default to False. This is the SAME majority-vote convention already used
    for the human-vs-golden Kappa comparison in datalake_evaluate_all.py, chosen here
    (rather than the mean-of-individual-accuracies used for the headline 'Human
    Average' figure in evaluation_summary.csv) because Wilson CI needs a single
    correct/incorrect count out of N=55, not a fractional score. Treat this as a
    distinct, explicitly-labelled definition from the headline Human Average.
    """
    gold_col = f"golden_{task}"
    # The published per-file CSV carries only the majority label (scripts/aggregate_per_file.py);
    # the per-participant columns exist only in the private working copy.
    if f"human_majority_{task}" in df.columns:
        return df[f"human_majority_{task}"].astype(str) == df[gold_col].astype(str)
    participants = sorted(c[: -len(f"_{task}")] for c in df.columns
                          if c.startswith("participant_") and c.endswith(f"_{task}"))
    votes = df[[f"{p}_{task}" for p in participants]].apply(
        lambda row: Counter(row).most_common(1)[0][0], axis=1
    )
    return votes == df[gold_col]


def run_wilson_ci(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 1a: Wilson 95% Confidence Intervals (n = 55)")
    logger.info("=" * 70)
    logger.info(
        "NOTE: 'Human Average' rows use per-file MAJORITY VOTE across the 3 "
        "participants (same convention as the Kappa human-average elsewhere in this "
        "project), not the mean-of-individual-accuracies used for the headline Human "
        "Average figure in evaluation_summary.csv. The two will not be numerically "
        "identical; this is disclosed for reproducibility."
    )
    rows = []
    n = len(df)

    method_bool_cols = {}
    for m in AUTOMATED_METHODS:
        method_bool_cols[METHOD_DISPLAY[m]] = {
            "category": df[f"{m}_cat_ok"],
            "security": df[f"{m}_sec_ok"],
            "overall":  df[f"{m}_cat_ok"] & df[f"{m}_sec_ok"],
        }
    method_bool_cols["Human (avg, majority vote)"] = {
        "category": human_majority_correct(df, "category"),
        "security": human_majority_correct(df, "security"),
    }
    hc = method_bool_cols["Human (avg, majority vote)"]
    hc["overall"] = hc["category"] & hc["security"]

    logger.info(f"{'Method':<28} {'Metric':<10} {'Correct/N':>10} {'Acc':>7} {'95% CI':>18}")
    logger.info("-" * 78)
    for method_name, tasks in method_bool_cols.items():
        for metric, bool_series in tasks.items():
            count = int(bool_series.sum())
            low, high = proportion_confint(count, n, method="wilson", alpha=0.05)
            acc = count / n
            logger.info(
                f"{method_name:<28} {metric:<10} {count:>3}/{n:<6} {acc:>7.3f} "
                f"  [{low:.3f}, {high:.3f}]"
            )
            rows.append({
                "method": method_name, "metric": metric,
                "correct": count, "n": n, "accuracy": round(acc, 4),
                "ci_low": round(low, 4), "ci_high": round(high, 4),
            })
    return pd.DataFrame(rows)


def run_mcnemar(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 1b: Paired McNemar Test - Security Decision (top-2 automated methods)")
    logger.info("=" * 70)

    sec_acc = {m: df[f"{m}_sec_ok"].mean() for m in AUTOMATED_METHODS}
    ranked = sorted(sec_acc.items(), key=lambda kv: kv[1], reverse=True)
    (method_a, acc_a), (method_b, acc_b) = ranked[0], ranked[1]
    logger.info(
        f"Security accuracy ranking: "
        + ", ".join(f"{METHOD_DISPLAY[m]}={a:.3f}" for m, a in ranked)
    )
    logger.info(f"Selected pair (top 2): {METHOD_DISPLAY[method_a]} vs {METHOD_DISPLAY[method_b]}")

    a_ok = df[f"{method_a}_sec_ok"]
    b_ok = df[f"{method_b}_sec_ok"]
    both_correct = int((a_ok & b_ok).sum())
    a_only        = int((a_ok & ~b_ok).sum())
    b_only        = int((~a_ok & b_ok).sum())
    both_wrong    = int((~a_ok & ~b_ok).sum())

    table = [[both_correct, a_only], [b_only, both_wrong]]
    result = mcnemar(table, exact=True)

    logger.info(f"Contingency table (rows={METHOD_DISPLAY[method_a]} correct/incorrect, "
                f"cols={METHOD_DISPLAY[method_b]} correct/incorrect):")
    logger.info(f"  both_correct={both_correct}  {METHOD_DISPLAY[method_a]}_only={a_only}  "
                f"{METHOD_DISPLAY[method_b]}_only={b_only}  both_wrong={both_wrong}")
    logger.info(f"McNemar exact test: statistic={result.statistic:.4f}  p-value={result.pvalue:.4f}")
    if a_only + b_only < 10:
        logger.info(
            f"NOTE: only {a_only + b_only} discordant pairs (files where the two methods "
            f"disagree) out of n=55 - the exact binomial test is appropriate here, but "
            f"treat the p-value as low-powered evidence given so few discordant cases."
        )

    return pd.DataFrame([{
        "method_a": METHOD_DISPLAY[method_a], "method_b": METHOD_DISPLAY[method_b],
        "both_correct": both_correct, "a_only": a_only, "b_only": b_only, "both_wrong": both_wrong,
        "statistic": round(float(result.statistic), 4), "p_value": round(float(result.pvalue), 4),
        "test_type": "mcnemar_exact",
    }])


# -- Item 2: Phase 1 vs Phase 2 ---------------------------------------------------
def run_phase_split(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 2: Phase 1 (manually authored, n=7) vs Phase 2 (generated, n=48)")
    logger.info("=" * 70)

    is_phase1 = df["file_name"].isin(PHASE1_FILES)
    unmatched = PHASE1_FILES - set(df.loc[is_phase1, "file_name"])
    if unmatched:
        logger.warning(f"  Phase 1 files not found in per-file CSV: {unmatched}")
    logger.info(f"Phase 1 files matched: {is_phase1.sum()} (expected 7)")
    logger.info(f"Phase 2 files matched: {(~is_phase1).sum()} (expected 48)")

    rows = []
    method_cols = {**{m: m for m in AUTOMATED_METHODS}}
    all_methods = list(AUTOMATED_METHODS)

    logger.info(f"\n{'Method':<20} {'Phase':<8} {'n':>4} {'Category':>10} {'Security':>10} {'Linkage':>10}")
    logger.info("-" * 68)
    for m in all_methods:
        for phase_label, mask in (("Phase 1", is_phase1), ("Phase 2", ~is_phase1)):
            sub = df[mask]
            cat_acc = sub[f"{m}_cat_ok"].mean()
            sec_acc = sub[f"{m}_sec_ok"].mean()
            link_acc = sub[f"{m}_link_ok"].mean()
            logger.info(f"{METHOD_DISPLAY[m]:<20} {phase_label:<8} {len(sub):>4} "
                        f"{cat_acc:>10.3f} {sec_acc:>10.3f} {link_acc:>10.3f}")
            rows.append({
                "method": METHOD_DISPLAY[m], "phase": phase_label, "n": len(sub),
                "category_accuracy": round(cat_acc, 4), "security_accuracy": round(sec_acc, 4),
                "linkage_accuracy": round(link_acc, 4),
            })

    # Human average (majority vote) and Expert Baseline, for completeness
    for phase_label, mask in (("Phase 1", is_phase1), ("Phase 2", ~is_phase1)):
        sub = df[mask]
        h_cat = human_majority_correct(sub, "category").mean()
        h_sec = human_majority_correct(sub, "security").mean()
        logger.info(f"{'Human (avg, maj.vote)':<20} {phase_label:<8} {len(sub):>4} "
                    f"{h_cat:>10.3f} {h_sec:>10.3f} {'n/a':>10}")
        rows.append({
            "method": "Human (avg, majority vote)", "phase": phase_label, "n": len(sub),
            "category_accuracy": round(h_cat, 4), "security_accuracy": round(h_sec, 4),
            "linkage_accuracy": None,
        })
        rows.append({
            "method": "Expert Baseline", "phase": phase_label, "n": len(sub),
            "category_accuracy": 1.0, "security_accuracy": 1.0, "linkage_accuracy": 1.0,
        })

    result_df = pd.DataFrame(rows)

    # Flag the specific pattern the reviewer is checking for: Phase 2 accuracy
    # substantially higher than Phase 1 for the same method (generator-regularity
    # exploitation signature). "Substantially" = more than 0.15 absolute difference,
    # a judgement threshold disclosed here rather than silently applied.
    logger.info("\nGenerator-regularity check (Phase 2 accuracy minus Phase 1 accuracy, category):")
    pivot = result_df[result_df["method"] != "Expert Baseline"].pivot(
        index="method", columns="phase", values="category_accuracy"
    )
    if "Phase 1" in pivot.columns and "Phase 2" in pivot.columns:
        pivot["diff"] = pivot["Phase 2"] - pivot["Phase 1"]
        for method, row in pivot.iterrows():
            flag = "  <-- FLAG: Phase 2 >> Phase 1" if row["diff"] > 0.15 else ""
            logger.info(f"  {method:<28} Phase1={row['Phase 1']:.3f}  Phase2={row['Phase 2']:.3f}  "
                        f"diff={row['diff']:+.3f}{flag}")
    return result_df


# -- Item 3: modality (file-type) split -------------------------------------------
def run_modality_split(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 3: Accuracy by File Type")
    logger.info("=" * 70)

    df = df.copy()
    df["file_type"] = df["file_name"].apply(file_type_of)
    type_counts = df["file_type"].value_counts()
    logger.info(f"File-type counts: {type_counts.to_dict()}")

    rows = []
    logger.info(f"\n{'Method':<20} {'Type':<6} {'n':>4} {'Category':>10} {'Security':>10} {'Linkage':>10}")
    logger.info("-" * 66)
    for m in AUTOMATED_METHODS:
        for ftype in sorted(df["file_type"].unique()):
            sub = df[df["file_type"] == ftype]
            cat_acc = sub[f"{m}_cat_ok"].mean()
            sec_acc = sub[f"{m}_sec_ok"].mean()
            link_acc = sub[f"{m}_link_ok"].mean()
            logger.info(f"{METHOD_DISPLAY[m]:<20} {ftype:<6} {len(sub):>4} "
                        f"{cat_acc:>10.3f} {sec_acc:>10.3f} {link_acc:>10.3f}")
            rows.append({
                "method": METHOD_DISPLAY[m], "file_type": ftype, "n": len(sub),
                "category_accuracy": round(cat_acc, 4), "security_accuracy": round(sec_acc, 4),
                "linkage_accuracy": round(link_acc, 4),
            })

    result_df = pd.DataFrame(rows)
    if "wav" in result_df["file_type"].values:
        logger.info("\nWAV vs non-WAV category accuracy check (transcription-error compounding):")
        for m in AUTOMATED_METHODS:
            sub = result_df[result_df["method"] == METHOD_DISPLAY[m]]
            wav_acc = sub.loc[sub["file_type"] == "wav", "category_accuracy"]
            other_acc = sub.loc[sub["file_type"] != "wav", "category_accuracy"]
            if not wav_acc.empty and not other_acc.empty:
                logger.info(f"  {METHOD_DISPLAY[m]:<20} wav={wav_acc.values[0]:.3f}  "
                            f"other_types_avg={other_acc.mean():.3f}  "
                            f"diff={wav_acc.values[0] - other_acc.mean():+.3f}")
    return result_df


# -- Item 5: generator-intended vs golden -----------------------------------------
def run_generator_vs_golden(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 5: Generator-Intended Labels vs Expert-Consensus Golden Labels (Phase 2 only)")
    logger.info("=" * 70)

    rows = []
    unmatched_prefixes = set()
    for _, row in df.iterrows():
        fn = row["file_name"]
        if fn in PHASE1_FILES:
            continue
        intended = intended_label_for(fn)
        if intended is None:
            unmatched_prefixes.add(fn)
            continue
        gen_cat, gen_sec, ambiguous, note = intended
        cat_match = gen_cat == row["golden_category"]
        sec_match = gen_sec == row["golden_security"]
        rows.append({
            "file_name": fn,
            "generator_category": gen_cat, "golden_category": row["golden_category"],
            "category_match": cat_match,
            "generator_security": gen_sec, "golden_security": row["golden_security"],
            "security_match": sec_match,
            "ambiguous_prefix": ambiguous, "ambiguity_note": note,
        })

    if unmatched_prefixes:
        logger.warning(f"  Files with no recognised Phase 2 prefix mapping: {unmatched_prefixes}")

    result_df = pd.DataFrame(rows)
    n = len(result_df)
    cat_agree = result_df["category_match"].sum()
    sec_agree = result_df["security_match"].sum()
    logger.info(f"\nPhase 2 files compared: {n}")
    logger.info(f"Category agreement (generator-intended vs golden): {cat_agree}/{n} = {cat_agree/n:.3f}")
    logger.info(f"Security agreement (generator-intended vs golden): {sec_agree}/{n} = {sec_agree/n:.3f}")

    n_ambig = result_df["ambiguous_prefix"].sum()
    logger.info(f"Files from an a-priori ambiguous prefix (flagged before comparison, not "
                f"after seeing disagreement): {n_ambig}/{n}")

    logger.info("\nAll category disagreements (generator-intended vs golden):")
    disagreements = result_df[~result_df["category_match"]]
    if disagreements.empty:
        logger.info("  (none)")
    else:
        for _, row in disagreements.iterrows():
            amb = " [FROM AMBIGUOUS PREFIX]" if row["ambiguous_prefix"] else ""
            logger.info(f"  {row['file_name']:<28} generator={row['generator_category']:<24} "
                        f"golden={row['golden_category']:<24}{amb}")
            if row["ambiguity_note"]:
                logger.info(f"      note: {row['ambiguity_note']}")

    logger.info("\nAll security disagreements (generator-intended vs golden):")
    sec_disagreements = result_df[~result_df["security_match"]]
    if sec_disagreements.empty:
        logger.info("  (none)")
    else:
        for _, row in sec_disagreements.iterrows():
            logger.info(f"  {row['file_name']:<28} generator={row['generator_security']:<14} "
                        f"golden={row['golden_security']:<14}")

    return result_df


# -- Item 6: LLaMA 3.2 1B failure taxonomy ----------------------------------------
FILE_HEADER_RE = re.compile(r"File \d+/\d+: (.+)$", re.MULTILINE)
LLM_RESULT_RE  = re.compile(r"LLM result: (\{.*\})\s*$", re.MULTILINE)
PARSE_FAIL_RE  = re.compile(r"JSON parse failed")

PLACEHOLDER_MARKERS = ("<identities", "<confidential", "|", ">")


def looks_like_placeholder(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return "<" in value and ">" in value


def looks_malformed_id(value: str) -> bool:
    """Same heuristic already used for graph-node labels in the pipeline scripts:
    flags customer_id/name values that are clearly not a real short identifier
    (contain template syntax, commas, or are implausibly long/sentence-like)."""
    if not value:
        return False
    return (
        "<" in value or ">" in value or "|" in value or "," in value
        or value.count(" ") > 2 or len(value) > 24
    )


def parse_llm_log(log_path: Path, logger: logging.Logger) -> dict:
    """Return {file_name: {'raw_dict': dict|None, 'parse_failed': bool}} for every
    file block found in the log."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"─{5,}", text)

    per_file = {}
    current_file = None
    for block in blocks:
        header_match = FILE_HEADER_RE.search(block)
        if header_match:
            current_file = header_match.group(1).strip()
        if current_file is None:
            continue
        parse_failed = bool(PARSE_FAIL_RE.search(block))
        result_match = LLM_RESULT_RE.search(block)
        raw_dict = None
        if result_match:
            try:
                raw_dict = ast.literal_eval(result_match.group(1))
            except Exception as e:
                logger.warning(f"  Could not parse LLM result dict for {current_file}: {e}")
        if header_match:
            per_file[current_file] = {"raw_dict": raw_dict, "parse_failed": parse_failed}
    return per_file


def run_failure_taxonomy(df: pd.DataFrame, logger: logging.Logger, log_path: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 70)
    logger.info("Item 6: LLaMA 3.2 1B Failure Taxonomy")
    logger.info("=" * 70)

    logger.info(f"Reading raw LLM log: {log_path}")
    parsed = parse_llm_log(log_path, logger)
    logger.info(f"File blocks parsed from log: {len(parsed)} (expected 55)")

    rows = []
    for _, row in df.iterrows():
        fn = row["file_name"]
        entry = parsed.get(fn)
        if entry is None:
            rows.append({"file_name": fn, "category": "not_found_in_log",
                          "detail": "No matching block found in the raw LLM log"})
            continue

        raw = entry["raw_dict"]
        if entry["parse_failed"] or raw is None:
            rows.append({"file_name": fn, "category": "schema_serialisation_failure",
                          "detail": "No valid JSON object could be extracted from the raw response"})
            continue

        info_type = raw.get("info_type", "")
        security_class = raw.get("security_class", "")

        if looks_like_placeholder(info_type) or looks_like_placeholder(security_class):
            rows.append({"file_name": fn, "category": "placeholder_echo",
                         "detail": f"info_type={info_type!r} security_class={security_class!r}"})
            continue

        if info_type not in VALID_CATEGORIES or security_class not in VALID_SECURITY:
            rows.append({"file_name": fn, "category": "invalid_enum_value",
                         "detail": f"info_type={info_type!r} security_class={security_class!r}"})
            continue

        cat_correct = info_type == row["golden_category"]
        if cat_correct:
            first_name = (raw.get("first_name") or "").strip()
            customer_id = (raw.get("customer_id") or "").strip()
            entity_flag = False
            detail = ""
            if info_type == "identities" and not first_name:
                entity_flag = True
                detail = "identities file but first_name empty"
            elif customer_id and looks_malformed_id(customer_id):
                entity_flag = True
                detail = f"malformed/hallucinated customer_id: {customer_id!r}"
            if entity_flag:
                rows.append({"file_name": fn, "category": "entity_extraction_error", "detail": detail})
                continue

        rows.append({
            "file_name": fn,
            "category": "valid_and_correct" if cat_correct else "valid_but_wrong_category",
            "detail": f"info_type={info_type} (golden={row['golden_category']})",
        })

    result_df = pd.DataFrame(rows)
    counts = result_df["category"].value_counts()
    n = len(result_df)
    logger.info(f"\nFailure taxonomy over {n} files:")
    for cat, cnt in counts.items():
        logger.info(f"  {cat:<32}: {cnt:>3} ({100*cnt/n:.0f}%)")

    placeholder_n = counts.get("placeholder_echo", 0)
    logger.info(
        f"\nPlaceholder-echo count check: {placeholder_n}/{n} = {100*placeholder_n/n:.0f}% "
        f"(paper/reviewer reference figure: 28/55 = 51%; confirm this still matches or "
        f"note the discrepancy explicitly)."
    )
    logger.warning(
        "NOTE: 'entity_extraction_error' uses an automated proxy (empty name on an "
        "identity document, or a malformed/hallucinated customer_id) rather than full "
        "manual verification against source content, as the brief itself only requires "
        "a coarse taxonomy but flags this sub-category as needing 'eyeballing.' Spot-check "
        "this category's rows before treating counts as final."
    )
    return result_df


# -- Main --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="RO1 robustness analyses")
    ap.add_argument("--per-file", help="evaluation/output/evaluation_per_file_<ts>.csv")
    ap.add_argument("--llama-log", help="output/log_llm_llama32_1b_<ts>.txt")
    ap.add_argument("--label", default="", help="suffix for output names, e.g. median_current_runtime")
    ap.add_argument("--no-taxonomy", action="store_true",
                    help="skip the LLaMA failure taxonomy (it then needs no --llama-log)")
    args = ap.parse_args()
    per_file_path = required(select_artefact(EVAL_OUT, "evaluation_per_file_*.csv", explicit=args.per_file,
                                             accept=row_count_check(55), what="per-file CSV",
                                             option="--per-file"), "evaluation_per_file_*.csv")
    llama_log = None if args.no_taxonomy else required(
        select_artefact(OUTPUT_DIR, "log_llm_llama32_1b_*.txt", explicit=args.llama_log,
                        accept=primary_corpus_log, what="LLaMA log", option="--llama-log"),
        "log_llm_llama32_1b_*.txt")
    stem = f"{TIMESTAMP}_{args.label}" if args.label else TIMESTAMP
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    log_path = EVAL_OUT / f"log_evaluate_robustness_{stem}.txt"
    logger = setup_logging(log_path)

    logger.info("=" * 70)
    logger.info("RO1 ROBUSTNESS ANALYSES (reviewer-requested, items 1/2/3/5/6)")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info("=" * 70)

    logger.info(f"Reading per-file evaluation CSV: {per_file_path}")
    df = pd.read_csv(per_file_path)
    logger.info(f"Rows loaded: {len(df)} (expected 55)")

    ci_df = run_wilson_ci(df, logger)
    ci_df.to_csv(EVAL_OUT / f"evaluation_confidence_intervals_{stem}.csv", index=False)

    mcnemar_df = run_mcnemar(df, logger)
    mcnemar_df.to_csv(EVAL_OUT / f"evaluation_mcnemar_security_{stem}.csv", index=False)

    phase_df = run_phase_split(df, logger)
    phase_df.to_csv(EVAL_OUT / f"evaluation_phase_split_{stem}.csv", index=False)

    modality_df = run_modality_split(df, logger)
    modality_df.to_csv(EVAL_OUT / f"evaluation_modality_split_{stem}.csv", index=False)

    gen_gold_df = run_generator_vs_golden(df, logger)
    gen_gold_df.to_csv(EVAL_OUT / f"evaluation_generator_vs_golden_{stem}.csv", index=False)

    if llama_log is not None:
        taxonomy_df = run_failure_taxonomy(df, logger, llama_log)
        taxonomy_df.to_csv(EVAL_OUT / f"evaluation_llama1b_failure_taxonomy_{stem}.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("Robustness analyses complete. All outputs written to:")
    logger.info(f"  {EVAL_OUT}")
    logger.info("Items NOT implemented (require explicit go-ahead first):")
    logger.info("  Item 4 - OCR field accuracy + audio WER (needs a ground-truth manifest)")
    logger.info("  Item 7 - Ollama seed pinning + public repository decisions")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
