"""
datalake_evaluate_all.py  -  RO1 Comprehensive Evaluation
==========================================================

Reads golden labels from  evaluation/evaluation_template_golden.xlsx  (new format,
55 files, researcher-filled) or falls back to  evaluation/archive_8_participants/Baseline.xlsx
(old 7-file labels) if the new golden template has not yet been completed.

Human participant responses are read from  evaluation/participant*.xlsx  (3 participants,
new underscore-column format).  Old archived files are in  evaluation/archive_8_participants/.

Automated pipeline CSVs are read from  output/metadata_*.csv.

Evaluates 4 factors for every method (automated + human average + expert baseline):
  1. Data class category accuracy  (info_type: identities / service_requests /
                                    financial_transactions / unknown)
  2. Security classification accuracy  (confidential / general)
  3. Customer linkage accuracy         (binary: correctly linked vs unknown)
  4. Time taken                        (seconds to classify all files)

Outputs (written to  evaluation/output/ ):
  evaluation_summary_<ts>.csv      - 4-factor scores per method
  evaluation_per_file_<ts>.csv     - per-file breakdown for all methods
  evaluation_human_detail_<ts>.csv - per-participant scores
  evaluation_kappa_<ts>.csv        - inter-rater Cohen's Kappa between methods
  evaluation_category_prf_<ts>.csv - per-class precision/recall/F1 for data class
                                     category (identities / service_requests /
                                     financial_transactions / unknown), per method
  evaluation_security_prf_<ts>.csv - per-class precision/recall/F1 for security
                                     classification (confidential / general), per method
  log_evaluation_<ts>.txt

Pipeline CSVs are selected explicitly (see datalake_artefacts.py). A CSV is only
auto-selected when it is the sole candidate over the golden file set; output/ normally
holds several (July run, October run, variance passes), so pin them:

Usage:
  python datalake_evaluate_all.py       --rule-based output/metadata_rule_based_20261021_222646.csv       --nlp        output/metadata_nlp_20261021_222739.csv       --llama      output/metadata_llm_llama32_1b_20261021_222833.csv       --phi3       output/metadata_llm_phi3_mini_20261021_223725.csv
  python datalake_evaluate_all.py --methods phi3 --phi3 <csv> --out-dir <dir>
"""

import argparse
import io
import re
import sys
import logging
import itertools
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import openpyxl
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

from datalake_artefacts import select_artefact, file_set_check

warnings.filterwarnings("ignore", category=UserWarning)

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
EVAL_DIR   = BASE_DIR / "evaluation"
OUTPUT_DIR = BASE_DIR / "output"
EVAL_OUT   = EVAL_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

# (CLI key, display name, CSV prefix) - the CSV used is chosen in main(), never by recency
PIPELINE_METHODS = [
    ("rule-based", "Rule-Based",       "metadata_rule_based_"),
    ("nlp",        "NLP",              "metadata_nlp_"),
    ("llama",      "LLM-llama3.2-1b",  "metadata_llm_llama32_1b_"),
    ("phi3",       "LLM-phi3-mini",    "metadata_llm_phi3_mini_"),
]

INFO_TYPE_CLASSES = ["identities", "service_requests", "financial_transactions", "unknown"]
SECURITY_CLASSES  = ["confidential", "general"]

# Canonical customer names (lower-case fuzzy normalisation)
KNOWN_CUSTOMERS = {
    "jane phillip": "Jane Phillip",
    "jane philip":  "Jane Phillip",
    "jane":         "Jane Phillip",
    "john murray":  "John Murray",
    "john":         "John Murray",
}


# -- Logging -------------------------------------------------------------------
def setup_logging(log_path: Path) -> logging.Logger:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("evaluate_all")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# -- Time parsing --------------------------------------------------------------
def parse_seconds(value) -> float | None:
    """Parse varied time formats from participant sheets into total seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower()
    s = s.rstrip(".")
    # "6.5 minutes" or "4 minutes"
    m = re.match(r"^([\d.]+)\s*min(?:ute)?s?$", s)
    if m:
        return float(m.group(1)) * 60
    # "5 min 9 seconds" / "4 minute 50 seconds" / "4 min 33 seconds"
    m = re.match(r"^(\d+)\s*min(?:ute)?s?\s+(\d+)\s*sec", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # "355 seconds"
    m = re.match(r"^([\d.]+)\s*sec", s)
    if m:
        return float(m.group(1))
    # bare number string
    try:
        return float(s)
    except ValueError:
        return None


# -- Customer linkage normalisation --------------------------------------------
def is_linked(value) -> bool:
    """Return True if the value represents a known customer (not unknown/empty)."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return s not in {"", "unknown", "none", "nan", "0", "no", "n/a"}


def norm_customer(value) -> str:
    """Normalise customer name to canonical form."""
    if not is_linked(value):
        return "unknown"
    s = str(value).strip().lower()
    for key, canonical in KNOWN_CUSTOMERS.items():
        if key in s:
            return canonical
    # numeric ID â†’ linked but unnamed
    if re.match(r"^\d+$", s):
        return "customer_id:" + s
    return str(value).strip()


# -- Read golden labels XLSX (new template or archived Baseline) ----------------
def _get_evaluation_sheet(wb):
    """Return the data worksheet: prefer 'evaluation', fall back to first sheet."""
    if "evaluation" in wb.sheetnames:
        return wb["evaluation"]
    return wb[wb.sheetnames[0]]


def _norm_header(raw) -> str:
    """Normalise header cell to lowercase with underscores (handles both old and new templates)."""
    s = str(raw).strip().lower() if raw else ""
    return s.replace(" ", "_")


def _find_header_row_idx(rows) -> int:
    """Find the row whose first cell is 'File Name' (handles a title/banner row above it)."""
    for i, row in enumerate(rows):
        if row and _norm_header(row[0]) == "file_name":
            return i
    return 0


def read_baseline(path: Path) -> tuple[pd.DataFrame, float | None]:
    """Return (golden_df, expert_time_seconds).

    Handles both:
      - New template columns: data_class_category, security_classification, linked_to_customer
      - Old Baseline columns:  data class category, security classification, linked to customer
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = _get_evaluation_sheet(wb)
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]

    hdr_idx = _find_header_row_idx(rows)
    header = [_norm_header(c) for c in rows[hdr_idx]]
    data_rows, time_sec = [], None

    for row in rows[hdr_idx + 1:]:
        d = dict(zip(header, row))
        fn = str(d.get("file_name", "") or "").strip()
        if not fn or fn.lower().startswith("time"):
            for val in row:
                t = parse_seconds(val)
                if t and t > 10:
                    time_sec = t
                    break
            continue
        # support both old (space) and new (underscore) column names - header is already normalised
        cust_val = d.get("linked_to_customer") or d.get("linked_to_customer_id")
        data_rows.append({
            "file_name":       fn,
            "golden_category": str(d.get("data_class_category", "") or "").strip().lower(),
            "golden_security": str(d.get("security_classification", "") or "").strip().lower(),
            "golden_linked":   is_linked(cust_val),
            "golden_customer": norm_customer(cust_val),
        })

    return pd.DataFrame(data_rows), time_sec


# -- Read one participant XLSX -------------------------------------------------
def read_participant(path: Path) -> tuple[pd.DataFrame, float | None]:
    """Return (answers_df, time_seconds).

    Reads from the 'evaluation' sheet if present; otherwise falls back to first sheet.
    Column names are normalised (lower-case with underscores) so both old-format
    ('data class category') and new-format ('data_class_category') templates work.
    Time is read from the 'total_time_taken_seconds' column when present.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = _get_evaluation_sheet(wb)
    rows = [row for row in ws.iter_rows(values_only=True) if any(v is not None for v in row)]

    if not rows:
        return pd.DataFrame(), None

    hdr_idx = _find_header_row_idx(rows)
    header = [_norm_header(c) for c in rows[hdr_idx]]
    data_rows, time_sec = [], None

    for row in rows[hdr_idx + 1:]:
        d = dict(zip(header, row))
        fn = str(d.get("file_name", "") or "").strip()
        TIME_LABELS = ("time", "elapsed", "timing", "duration", "total")
        is_time_row = (
            not fn
            or any(fn.lower().startswith(lbl) for lbl in TIME_LABELS)
            or any(isinstance(v, str) and v.strip().lower().startswith("time") for v in row if v)
        )
        if is_time_row:
            for val in row:
                t = parse_seconds(val)
                if t and t > 10:
                    time_sec = t
                    break
            continue

        # capture per-row time from new-format 'total_time_taken_seconds' column
        if time_sec is None:
            row_time = parse_seconds(d.get("total_time_taken_seconds"))
            if row_time and row_time > 10:
                time_sec = row_time

        cust_key = next((k for k in header if "customer" in k), None)
        cust_val = d.get(cust_key) if cust_key else None
        data_rows.append({
            "file_name": fn,
            "category":  str(d.get("data_class_category", "") or "").strip().lower(),
            "security":  str(d.get("security_classification", "") or "").strip().lower(),
            "linked":    is_linked(cust_val),
            "customer":  norm_customer(cust_val),
        })

    return pd.DataFrame(data_rows), time_sec


def read_pipeline_csv(path: Path, display_name: str) -> tuple[pd.DataFrame, float | None]:
    """Normalise pipeline CSV to standard evaluation columns."""
    df = pd.read_csv(path)
    # customer_id non-empty â†’ linked
    df["linked"] = df.get("customer_id", pd.Series([""] * len(df))).apply(
        lambda v: is_linked(v)
    )
    df["customer"] = df.get("customer_id", pd.Series([""] * len(df))).apply(norm_customer)
    df["category"] = df["info_type"].str.strip().str.lower()
    df["security"] = df["security_class"].str.strip().str.lower()
    df = df[["file_name", "category", "security", "linked", "customer"]].copy()

    avg_sec = None
    if "processing_sec" in pd.read_csv(path).columns:
        avg_sec = pd.read_csv(path)["processing_sec"].sum()

    return df, avg_sec


# -- Score one method against golden ------------------------------------------
def score_method(golden: pd.DataFrame, pred: pd.DataFrame, name: str) -> dict:
    merged = golden.merge(pred, on="file_name", how="left")
    n = len(merged)

    cat_acc  = (merged["category"] == merged["golden_category"]).sum() / n
    sec_acc  = (merged["security"] == merged["golden_security"]).sum() / n
    link_acc = (merged["linked"]   == merged["golden_linked"]).sum()   / n

    return {
        "method":              name,
        "n_files":             n,
        "category_accuracy":   round(cat_acc,  3),
        "security_accuracy":   round(sec_acc,  3),
        "linkage_accuracy":    round(link_acc, 3),
        "overall_accuracy":    round((cat_acc + sec_acc + link_acc) / 3, 3),
    }


# -- Main ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="RO1 evaluation of all methods against the golden labels")
    for key, display_name, prefix in PIPELINE_METHODS:
        ap.add_argument(f"--{key}", metavar="CSV", help=f"{display_name} metadata CSV ({prefix}*.csv)")
    ap.add_argument("--methods", nargs="+", choices=[k for k, _, _ in PIPELINE_METHODS],
                    default=[k for k, _, _ in PIPELINE_METHODS], help="pipelines to score (default: all)")
    ap.add_argument("--out-dir", type=Path, default=EVAL_OUT)
    args = ap.parse_args()

    eval_out = args.out_dir
    eval_out.mkdir(parents=True, exist_ok=True)
    log_path     = eval_out / f"log_evaluation_{TIMESTAMP}.txt"
    summary_path = eval_out / f"evaluation_summary_{TIMESTAMP}.csv"
    perfile_path = eval_out / f"evaluation_per_file_{TIMESTAMP}.csv"
    human_path   = eval_out / f"evaluation_human_detail_{TIMESTAMP}.csv"
    kappa_path   = eval_out / f"evaluation_kappa_{TIMESTAMP}.csv"
    prf_path     = eval_out / f"evaluation_category_prf_{TIMESTAMP}.csv"
    sec_prf_path = eval_out / f"evaluation_security_prf_{TIMESTAMP}.csv"

    logger = setup_logging(log_path)
    logger.info("=" * 70)
    logger.info("RO1 COMPREHENSIVE EVALUATION  -  All Methods vs Baseline")
    logger.info(f"Timestamp : {TIMESTAMP}")
    logger.info("=" * 70)

    # -- 1. Golden labels - prefer new researcher-filled template, fall back to archive --
    _new_golden  = EVAL_DIR / "evaluation_template_golden.xlsx"
    _old_golden  = EVAL_DIR / "archive_8_participants" / "Baseline.xlsx"
    if _new_golden.exists():
        baseline_path = _new_golden
    elif _old_golden.exists():
        baseline_path = _old_golden
        logger.warning(
            "New golden template not found - falling back to archived Baseline.xlsx (7 files only). "
            "Fill evaluation/evaluation_template_golden.xlsx for full 55-file evaluation."
        )
    else:
        logger.error("No golden labels found. Expected evaluation/evaluation_template_golden.xlsx")
        return
    golden_df, expert_time = read_baseline(baseline_path)
    logger.info(f"\nGolden labels loaded : {len(golden_df)} files from {baseline_path.name}")
    logger.info(f"Expert time          : {expert_time}s")
    logger.info("\nGolden label summary:")
    for _, row in golden_df.iterrows():
        logger.info(
            f"  {row['file_name']:<35} "
            f"cat={row['golden_category']:<22} "
            f"sec={row['golden_security']:<14} "
            f"linked={row['golden_linked']}"
        )

    # -- 2. Participant XLSX files ----------------------------------------------
    # Deduplicate using resolved paths (Windows glob is case-insensitive).
    # Exclude template stubs and the golden file - only real completed participant files.
    EXCLUDED_NAMES = {"Baseline.xlsx", "evaluation_template_participant.xlsx", "evaluation_template_golden.xlsx"}
    seen = set()
    participant_paths = []
    # Only participant*.xlsx: a bare *.xlsx glob also picked up tab_golden.xlsx as a "participant".
    for p in sorted(EVAL_DIR.glob("participant*.xlsx")):
        key = p.resolve()
        if key not in seen and not p.name.startswith("~") and p.name not in EXCLUDED_NAMES:
            seen.add(key)
            participant_paths.append(p)
    logger.info(f"\nParticipant files    : {len(participant_paths)} found")

    human_rows   = []
    human_preds  = []  # list of (name, df) for kappa
    human_times  = []

    for p in participant_paths:
        if p.name.startswith("~"):
            continue
        pred_df, time_sec = read_participant(p)
        if pred_df.empty:
            logger.warning(f"  {p.name}: empty - skipped")
            continue
        scores = score_method(golden_df, pred_df, p.stem)
        scores["time_seconds"] = time_sec
        human_rows.append(scores)
        human_preds.append((p.stem, pred_df))
        if time_sec:
            human_times.append(time_sec)
        t_str = f"{time_sec:.0f}s" if time_sec is not None else "N/A"
        logger.info(
            f"  {p.stem:<30}  cat={scores['category_accuracy']:.3f}  "
            f"sec={scores['security_accuracy']:.3f}  "
            f"link={scores['linkage_accuracy']:.3f}  "
            f"time={t_str}"
        )

    human_detail_df = pd.DataFrame(human_rows)
    human_detail_df.to_csv(human_path, index=False)
    logger.info(f"\nHuman detail CSV     : {human_path}")

    # Human average row
    if human_rows:
        avg_cat  = round(pd.DataFrame(human_rows)["category_accuracy"].mean(), 3)
        avg_sec  = round(pd.DataFrame(human_rows)["security_accuracy"].mean(), 3)
        avg_link = round(pd.DataFrame(human_rows)["linkage_accuracy"].mean(), 3)
        avg_time = round(sum(human_times) / len(human_times), 1) if human_times else None
        human_avg_scores = {
            "method":            "Human (avg)",
            "n_files":           len(golden_df),
            "category_accuracy": avg_cat,
            "security_accuracy": avg_sec,
            "linkage_accuracy":  avg_link,
            "overall_accuracy":  round((avg_cat + avg_sec + avg_link) / 3, 3),
            "time_seconds":      avg_time,
        }
        logger.info(
            f"\nHuman average        :  cat={avg_cat:.3f}  "
            f"sec={avg_sec:.3f}  link={avg_link:.3f}  "
            f"time={avg_time}s"
        )

    # -- 3. Expert Baseline (researcher) ---------------------------------------
    expert_scores = score_method(golden_df, golden_df.rename(columns={
        "golden_category": "category",
        "golden_security":  "security",
        "golden_linked":    "linked",
        "golden_customer":  "customer",
    }), "Expert Baseline")
    expert_scores["time_seconds"] = expert_time

    # -- 4. Automated pipeline methods -----------------------------------------
    logger.info("\nAutomated pipeline results:")
    pipeline_rows = []
    pipeline_preds: list[tuple[str, pd.DataFrame]] = []

    on_golden = file_set_check(golden_df["file_name"])
    for key, display_name, prefix in PIPELINE_METHODS:
        if key not in args.methods:
            continue
        csv_path = select_artefact(OUTPUT_DIR, f"{prefix}*.csv", explicit=getattr(args, key.replace("-", "_")),
                                   accept=on_golden, what=display_name, option=f"--{key}")
        if not csv_path:
            logger.warning(f"  {display_name:<14}: no CSV found - skipping")
            continue
        pred_df, total_sec = read_pipeline_csv(csv_path, display_name)
        scores = score_method(golden_df, pred_df, display_name)
        scores["time_seconds"] = total_sec
        pipeline_rows.append(scores)
        pipeline_preds.append((display_name, pred_df))
        logger.info(
            f"  {display_name:<14}: cat={scores['category_accuracy']:.3f}  "
            f"sec={scores['security_accuracy']:.3f}  "
            f"link={scores['linkage_accuracy']:.3f}  "
            f"time={total_sec:.1f}s  ({csv_path.name})"
        )

    # -- 5. Summary table ------------------------------------------------------
    summary_rows = []
    if human_rows:
        summary_rows.append(human_avg_scores)
    summary_rows.append(expert_scores)
    summary_rows.extend(pipeline_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY - 4 Factors")
    logger.info("=" * 70)
    hdr = f"{'Method':<20} {'Category':>10} {'Security':>10} {'Linkage':>10} {'Overall':>10} {'Time(s)':>10}"
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for _, row in summary_df.iterrows():
        t = f"{row['time_seconds']:.0f}" if pd.notna(row.get("time_seconds")) else "N/A"
        logger.info(
            f"{row['method']:<20} "
            f"{row['category_accuracy']:>10.3f} "
            f"{row['security_accuracy']:>10.3f} "
            f"{row['linkage_accuracy']:>10.3f} "
            f"{row['overall_accuracy']:>10.3f} "
            f"{t:>10}"
        )
    logger.info("-" * len(hdr))
    logger.info(f"\nSummary CSV          : {summary_path}")

    # -- 6. Per-file breakdown -------------------------------------------------
    perfile_df = golden_df.copy()
    all_method_preds = (
        (human_preds if human_rows else []) +
        [("Expert Baseline", golden_df.rename(columns={
            "golden_category": "category",
            "golden_security":  "security",
            "golden_linked":    "linked",
            "golden_customer":  "customer",
        }))] +
        pipeline_preds
    )
    for name, pred_df in all_method_preds:
        key = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        merged = golden_df[["file_name"]].merge(pred_df[["file_name", "category", "security", "linked"]], on="file_name", how="left")
        perfile_df[f"{key}_category"] = merged["category"].values
        perfile_df[f"{key}_security"]  = merged["security"].values
        perfile_df[f"{key}_linked"]    = merged["linked"].values
        perfile_df[f"{key}_cat_ok"]    = (merged["category"].values == golden_df["golden_category"].values)
        perfile_df[f"{key}_sec_ok"]    = (merged["security"].values  == golden_df["golden_security"].values)
        perfile_df[f"{key}_link_ok"]   = (merged["linked"].values    == golden_df["golden_linked"].values)

    perfile_df.to_csv(perfile_path, index=False)
    logger.info(f"Per-file CSV         : {perfile_path}")

    # Per-file readable table for category
    logger.info("\n-- Per-file: Data Class Category --")
    pipe_names = [n for n, _ in pipeline_preds]
    col_w = 22
    header_line = f"{'File':<35} {'Golden':<22}" + "".join(f"{n:<{col_w}}" for n in pipe_names)
    logger.info(header_line)
    logger.info("-" * len(header_line))
    for _, row in golden_df.iterrows():
        fn = row["file_name"]
        line = f"{fn:<35} {row['golden_category']:<22}"
        for name, pred_df in pipeline_preds:
            m = pred_df[pred_df["file_name"] == fn]
            val  = m["category"].values[0] if not m.empty else "?"
            mark = "✓" if val == row["golden_category"] else "✗"
            line += f"{str(val):<{col_w-2}}{mark}  "
        logger.info(line)
    logger.info("-" * len(header_line))

    # Per-file security
    logger.info("\n-- Per-file: Security Classification --")
    logger.info(header_line.replace("Category", "Security "))
    logger.info("-" * len(header_line))
    for _, row in golden_df.iterrows():
        fn = row["file_name"]
        line = f"{fn:<35} {row['golden_security']:<22}"
        for name, pred_df in pipeline_preds:
            m = pred_df[pred_df["file_name"] == fn]
            val  = m["security"].values[0] if not m.empty else "?"
            mark = "✓" if val == row["golden_security"] else "✗"
            line += f"{str(val):<{col_w-2}}{mark}  "
        logger.info(line)
    logger.info("-" * len(header_line))

    # Per-file linkage
    logger.info("\n-- Per-file: Customer Linkage --")
    logger.info(header_line.replace("Category", "Linked? "))
    logger.info("-" * len(header_line))
    for _, row in golden_df.iterrows():
        fn = row["file_name"]
        line = f"{fn:<35} {str(row['golden_linked']):<22}"
        for name, pred_df in pipeline_preds:
            m = pred_df[pred_df["file_name"] == fn]
            val  = m["linked"].values[0] if not m.empty else False
            mark = "✓" if val == row["golden_linked"] else "✗"
            line += f"{str(val):<{col_w-2}}{mark}  "
        logger.info(line)
    logger.info("-" * len(header_line))

    # -- 7. Inter-method Cohen's Kappa -----------------------------------------
    # Compare automated pipelines against human average for each factor
    kappa_rows = []
    all_named_preds = [("Human (avg)", None)] if human_rows else []
    all_named_preds += pipeline_preds

    # Build prediction arrays aligned to golden file order
    file_order = golden_df["file_name"].tolist()

    def get_aligned(name, pred_df, col, default):
        if pred_df is None:
            # Human average: majority vote per file
            votes = []
            for fn in file_order:
                vals = []
                for hname, hdf in human_preds:
                    m = hdf[hdf["file_name"] == fn]
                    if not m.empty:
                        vals.append(m[col].values[0])
                # majority vote
                if vals:
                    from collections import Counter
                    votes.append(Counter(vals).most_common(1)[0][0])
                else:
                    votes.append(default)
            return votes
        result = []
        for fn in file_order:
            m = pred_df[pred_df["file_name"] == fn]
            result.append(m[col].values[0] if not m.empty else default)
        return result

    gold_cat  = golden_df.set_index("file_name").loc[file_order, "golden_category"].tolist()
    gold_sec  = golden_df.set_index("file_name").loc[file_order, "golden_security"].tolist()
    gold_link = golden_df.set_index("file_name").loc[file_order, "golden_linked"].astype(str).tolist()

    logger.info("\nCohen's Kappa (each method vs Golden Labels):")
    logger.info(f"{'Method':<20} {'Kappa-Category':>16} {'Kappa-Security':>16} {'Kappa-Linkage':>14}")
    logger.info("-" * 70)
    for name, pred_df in pipeline_preds:
        pred_cat  = get_aligned(name, pred_df, "category", "unknown")
        pred_sec  = get_aligned(name, pred_df, "security",  "general")
        pred_link = [str(v) for v in get_aligned(name, pred_df, "linked", False)]
        try:
            kc = cohen_kappa_score(gold_cat,  pred_cat,  labels=INFO_TYPE_CLASSES)
            ks = cohen_kappa_score(gold_sec,  pred_sec,  labels=SECURITY_CLASSES)
            kl = cohen_kappa_score(gold_link, pred_link)
            kappa_rows.append({"method_a": "Golden", "method_b": name,
                                "kappa_category": round(kc, 3),
                                "kappa_security":  round(ks, 3),
                                "kappa_linkage":   round(kl, 3)})
            logger.info(f"{name:<20} {kc:>16.3f} {ks:>16.3f} {kl:>14.3f}")
        except Exception as e:
            logger.warning(f"  {name}: Kappa failed - {e}")

    # Pairwise kappa between pipeline methods
    logger.info("\nPairwise Cohen's Kappa (between automated methods):")
    logger.info(f"{'Method A':<16} {'Method B':<16} {'Category':>10} {'Security':>10} {'Linkage':>10}")
    logger.info("-" * 65)
    for (na, dfa), (nb, dfb) in itertools.combinations(pipeline_preds, 2):
        pa_cat  = get_aligned(na, dfa, "category", "unknown")
        pb_cat  = get_aligned(nb, dfb, "category", "unknown")
        pa_sec  = get_aligned(na, dfa, "security",  "general")
        pb_sec  = get_aligned(nb, dfb, "security",  "general")
        pa_link = [str(v) for v in get_aligned(na, dfa, "linked", False)]
        pb_link = [str(v) for v in get_aligned(nb, dfb, "linked", False)]
        try:
            kc = cohen_kappa_score(pa_cat,  pb_cat)
            ks = cohen_kappa_score(pa_sec,  pb_sec)
            kl = cohen_kappa_score(pa_link, pb_link)
            kappa_rows.append({"method_a": na, "method_b": nb,
                                "kappa_category": round(kc, 3),
                                "kappa_security":  round(ks, 3),
                                "kappa_linkage":   round(kl, 3)})
            logger.info(f"{na:<16} {nb:<16} {kc:>10.3f} {ks:>10.3f} {kl:>10.3f}")
        except Exception as e:
            logger.warning(f"  {na} vs {nb}: Kappa failed - {e}")

    pd.DataFrame(kappa_rows).to_csv(kappa_path, index=False)
    logger.info(f"\nKappa CSV            : {kappa_path}")

    # -- 8. Per-class precision / recall / F1 for data class category ---------
    # Kappa and accuracy summarise agreement in a single number and can mask
    # imbalanced-class effects (e.g. a method that never predicts 'unknown').
    # Per-class precision/recall/F1 makes that visible for the four-way
    # category task, which is the primary classification target of this study.
    logger.info("\n" + "=" * 70)
    logger.info("Per-Class Precision / Recall / F1 - Data Class Category")
    logger.info("=" * 70)
    prf_rows = []
    for name, pred_df in all_named_preds:
        pred_cat = get_aligned(name, pred_df, "category", "unknown")
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_cat, pred_cat, labels=INFO_TYPE_CLASSES, zero_division=0
        )
        logger.info(f"\n{name}:")
        logger.info(f"  {'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        for cls, p, r, f, s in zip(INFO_TYPE_CLASSES, precision, recall, f1, support):
            logger.info(f"  {cls:<24} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>8d}")
            prf_rows.append({
                "method": name, "class": cls,
                "precision": round(float(p), 3),
                "recall": round(float(r), 3),
                "f1": round(float(f), 3),
                "support": int(s),
            })
        # Macro average: unweighted mean across the four classes, so rare
        # classes (e.g. 'identities') count as much as common ones - standard
        # practice for imbalanced multi-class reporting.
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_cat, pred_cat, labels=INFO_TYPE_CLASSES, average="macro", zero_division=0
        )
        logger.info(f"  {'macro avg':<24} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f} {len(gold_cat):>8d}")
        prf_rows.append({
            "method": name, "class": "macro avg",
            "precision": round(float(macro_p), 3),
            "recall": round(float(macro_r), 3),
            "f1": round(float(macro_f1), 3),
            "support": len(gold_cat),
        })

    pd.DataFrame(prf_rows).to_csv(prf_path, index=False)
    logger.info(f"\nCategory precision/recall/F1 CSV : {prf_path}")

    # -- 9. Per-class precision / recall / F1 for security classification -----
    # Security is binary (confidential/general), but accuracy alone can hide
    # a method that just defaults to the majority class (as already seen for
    # some methods via a near-zero security Kappa in step 7 above).
    logger.info("\n" + "=" * 70)
    logger.info("Per-Class Precision / Recall / F1 - Security Classification")
    logger.info("=" * 70)
    sec_prf_rows = []
    for name, pred_df in all_named_preds:
        pred_sec = get_aligned(name, pred_df, "security", "general")
        precision, recall, f1, support = precision_recall_fscore_support(
            gold_sec, pred_sec, labels=SECURITY_CLASSES, zero_division=0
        )
        logger.info(f"\n{name}:")
        logger.info(f"  {'Class':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        for cls, p, r, f, s in zip(SECURITY_CLASSES, precision, recall, f1, support):
            logger.info(f"  {cls:<24} {p:>10.3f} {r:>10.3f} {f:>10.3f} {s:>8d}")
            sec_prf_rows.append({
                "method": name, "class": cls,
                "precision": round(float(p), 3),
                "recall": round(float(r), 3),
                "f1": round(float(f), 3),
                "support": int(s),
            })
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            gold_sec, pred_sec, labels=SECURITY_CLASSES, average="macro", zero_division=0
        )
        logger.info(f"  {'macro avg':<24} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f} {len(gold_sec):>8d}")
        sec_prf_rows.append({
            "method": name, "class": "macro avg",
            "precision": round(float(macro_p), 3),
            "recall": round(float(macro_r), 3),
            "f1": round(float(macro_f1), 3),
            "support": len(gold_sec),
        })

    pd.DataFrame(sec_prf_rows).to_csv(sec_prf_path, index=False)
    logger.info(f"\nSecurity precision/recall/F1 CSV : {sec_prf_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete. All outputs written to:")
    logger.info(f"  {eval_out}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

