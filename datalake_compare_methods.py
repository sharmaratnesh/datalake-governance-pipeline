"""
Data Lake Methods Comparison  —  Experiment 1, RO1

Compares all pipeline outputs (rule-based, NLP, LLM-llama3.2-1b, LLM-phi3-mini)
against expert golden labels.  GOLDEN_LABELS covers the original 7 manually
created objects; accuracy is therefore computed on those 7 files only.
For full 55-file comparison, complete evaluation/evaluation_template_golden.xlsx
and run datalake_evaluate_all.py instead.

Computes per-method:
  • Accuracy, precision, recall, F1 for info-type classification
  • Accuracy, precision, recall, F1 for security classification (binary)
  • Processing time (avg seconds per file)

Across all method pairs:
  • Inter-method Cohen's Kappa (info-type and security)

Outputs written to  output/ :
  comparison_<timestamp>.csv       — per-file comparison table
  metrics_<timestamp>.csv          — classification performance metrics
  distribution_<timestamp>.csv     — category distribution per method
  kappa_<timestamp>.csv            — pairwise Cohen's Kappa
  log_compare_<timestamp>.txt      — execution log

Run AFTER all pipelines have produced output CSVs:
  python datalake_compare_methods.py [--csv key=path ...]

A method's CSV is auto-selected only if it is the ONLY data_swamp run in output/;
otherwise pin it, key = prefix without "metadata_", e.g.
  --csv llm_phi3_mini=output/metadata_llm_phi3_mini_20261021_223725.csv
"""

import argparse
import io
import sys
import logging
import itertools
from pathlib import Path
from datetime import datetime

import pandas as pd

from datalake_artefacts import select_artefact, file_set_check, parse_pins, prefix_key
from sklearn.metrics import (
    precision_recall_fscore_support,
    cohen_kappa_score,
    accuracy_score,
)

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Golden labels — expert annotation for the 7 data swamp objects ──────────
# info_type : identities / service_requests / financial_transactions / unknown
# security  : confidential / general
GOLDEN_LABELS = {
    "DL.jpg": {
        "info_type": "identities",
        "security_class": "confidential",
        "notes": "Driver licence image — contains name, DOB, address, licence number (PII)",
    },
    "identity driving license.jpg": {
        "info_type": "identities",
        "security_class": "confidential",
        "notes": "Second driver licence image — contains name, DOB, address, licence number (PII)",
    },
    "jane complaint.wav": {
        "info_type": "service_requests",
        "security_class": "confidential",
        "notes": "Customer complaint audio — mentions name and account details",
    },
    "call2.wav": {
        "info_type": "service_requests",
        "security_class": "confidential",
        "notes": "Short complaint reference audio — caller identifies themselves by name (PII)",
    },
    "Recording.wav": {
        "info_type": "unknown",
        "security_class": "general",
        "notes": "Internal availability / meeting coordination audio — no sensitive content",
    },
    "Transaction 1.txt": {
        "info_type": "financial_transactions",
        "security_class": "confidential",
        "notes": "Purchase record naming John Murray — contains PII (full name linked to transaction)",
    },
    "Transactions.csv": {
        "info_type": "financial_transactions",
        "security_class": "confidential",
        "notes": "Tabular transaction data with customer IDs and amounts",
    },
}

INFO_TYPE_CLASSES = ["identities", "service_requests", "financial_transactions", "unknown"]
SECURITY_CLASSES  = ["confidential", "general"]

# ── Method registry: (display_name, csv_prefix) ───────────────────────────────
# The CSV used for each is chosen explicitly in main(), never by recency.
METHODS = [
    ("Rule-Based",       "metadata_rule_based_"),
    ("NLP",              "metadata_nlp_"),
    ("LLM-llama3.2-1b",  "metadata_llm_llama32_1b_"),
    ("LLM-phi3-mini",    "metadata_llm_phi3_mini_"),
]

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("compare")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ── CSV discovery ─────────────────────────────────────────────────────────────
# ── Metrics computation ───────────────────────────────────────────────────────
def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    method_name: str,
    task: str,
    logger: logging.Logger,
) -> list[dict]:
    rows = []
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"\n{task} — {method_name}")
    logger.info(f"  Accuracy: {acc:.3f}")
    for lbl, p, r, f, s in zip(labels, prec, rec, f1, sup):
        logger.info(f"  {lbl:<30} P={p:.2f}  R={r:.2f}  F1={f:.2f}  n={s}")
        rows.append({
            "task":      task,
            "method":    method_name,
            "class":     lbl,
            "precision": round(float(p), 3),
            "recall":    round(float(r), 3),
            "f1":        round(float(f), 3),
            "support":   int(s),
            "accuracy":  round(acc, 3),
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Compare pipeline outputs on the data_swamp corpus")
    ap.add_argument("--csv", action="append", metavar="KEY=PATH", help="pin a metadata CSV (repeatable)")
    pins = parse_pins(ap.parse_args().csv)
    swamp_files = [p.name for p in (BASE_DIR / "data_swamp").glob("*") if p.is_file()]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path  = OUTPUT_DIR / f"log_compare_{TIMESTAMP}.txt"
    comp_path = OUTPUT_DIR / f"comparison_{TIMESTAMP}.csv"
    met_path  = OUTPUT_DIR / f"metrics_{TIMESTAMP}.csv"
    dist_path = OUTPUT_DIR / f"distribution_{TIMESTAMP}.csv"
    kap_path  = OUTPUT_DIR / f"kappa_{TIMESTAMP}.csv"

    logger = setup_logging(log_path)
    logger.info("=" * 65)
    logger.info("DATA LAKE METHODS COMPARISON")
    logger.info(f"Timestamp: {TIMESTAMP}")
    logger.info("=" * 65)

    # ── Load all available method CSVs ────────────────────────────────────────
    loaded: list[tuple[str, pd.DataFrame]] = []
    for display_name, prefix in METHODS:
        csv_path = select_artefact(OUTPUT_DIR, f"{prefix}*.csv", explicit=pins.get(prefix_key(prefix)),
                                   accept=file_set_check(swamp_files), what=display_name,
                                   option=f"--csv {prefix_key(prefix)}=")
        if csv_path:
            df = pd.read_csv(csv_path)
            loaded.append((display_name, df))
            logger.info(f"{display_name:<14}: {csv_path.name}")
        else:
            logger.warning(f"{display_name:<14}: no CSV found (skipping)")

    if not loaded:
        logger.error("No method output CSVs found. Run the pipeline scripts first.")
        return

    # ── Build golden DataFrame ────────────────────────────────────────────────
    golden_df = pd.DataFrame([
        {"file_name": k,
         "golden_info_type": v["info_type"],
         "golden_security":  v["security_class"],
         "notes":            v["notes"]}
        for k, v in GOLDEN_LABELS.items()
    ])

    # ── Merge all methods into comparison table ────────────────────────────────
    comp_df = golden_df.copy()
    for display_name, df in loaded:
        key = display_name.lower().replace("-", "_").replace(" ", "_")
        slim = df[["file_name", "info_type", "security_class"]].copy()
        if "processing_sec" in df.columns:
            slim["processing_sec"] = df["processing_sec"]
        slim = slim.rename(columns={
            "info_type":      f"{key}_info_type",
            "security_class": f"{key}_security",
            "processing_sec": f"{key}_sec",
        })
        comp_df = comp_df.merge(slim, on="file_name", how="left")
        comp_df[f"{key}_info_correct"] = comp_df[f"{key}_info_type"] == comp_df["golden_info_type"]
        comp_df[f"{key}_sec_correct"]  = comp_df[f"{key}_security"]  == comp_df["golden_security"]

    comp_df.to_csv(comp_path, index=False)
    logger.info(f"\nPer-file comparison → {comp_path}")

    # ── Print per-file info-type comparison table ─────────────────────────────
    method_keys = [n.lower().replace("-", "_").replace(" ", "_") for n, _ in loaded]
    method_names = [n for n, _ in loaded]
    col_w = 22

    header = f"{'File':<32} {'Golden':<24}" + "".join(f"{n:<{col_w}}" for n in method_names)
    logger.info("\n" + "─" * len(header))
    logger.info(header)
    logger.info("─" * len(header))
    for _, row in comp_df.iterrows():
        cols = f"{row['file_name']:<32} {row['golden_info_type']:<24}"
        for key in method_keys:
            val  = str(row.get(f"{key}_info_type", "?"))
            mark = "✓" if row.get(f"{key}_info_correct") else "✗"
            cols += f"{val:<{col_w-2}}{mark}  "
        logger.info(cols)
    logger.info("─" * len(header))

    # ── Print per-file security comparison table ───────────────────────────────
    header2 = f"{'File':<32} {'Golden Sec':<16}" + "".join(f"{n:<16}" for n in method_names)
    logger.info("\n" + "─" * len(header2))
    logger.info(header2)
    logger.info("─" * len(header2))
    for _, row in comp_df.iterrows():
        cols = f"{row['file_name']:<32} {row['golden_security']:<16}"
        for key in method_keys:
            val  = str(row.get(f"{key}_security", "?"))
            mark = "✓" if row.get(f"{key}_sec_correct") else "✗"
            cols += f"{val:<14}{mark}  "
        logger.info(cols)
    logger.info("─" * len(header2))

    # ── Metrics per method ────────────────────────────────────────────────────
    valid = comp_df.dropna(subset=[f"{method_keys[0]}_info_type"])
    y_golden_it = valid["golden_info_type"].tolist()
    y_golden_s  = valid["golden_security"].tolist()

    metrics_rows = []
    pred_it: dict[str, list] = {}
    pred_s:  dict[str, list] = {}

    for key, name in zip(method_keys, method_names):
        if f"{key}_info_type" not in valid.columns:
            continue
        valid_m = comp_df.dropna(subset=[f"{key}_info_type"])
        yt_it = valid_m["golden_info_type"].tolist()
        yt_s  = valid_m["golden_security"].tolist()
        yp_it = valid_m[f"{key}_info_type"].tolist()
        yp_s  = valid_m[f"{key}_security"].tolist()
        pred_it[name] = yp_it
        pred_s[name]  = yp_s
        metrics_rows += compute_metrics(yt_it, yp_it, INFO_TYPE_CLASSES, name, "info_type", logger)
        metrics_rows += compute_metrics(yt_s,  yp_s,  SECURITY_CLASSES,  name, "security",  logger)

    pd.DataFrame(metrics_rows).to_csv(met_path, index=False)
    logger.info(f"\nMetrics CSV → {met_path}")

    # ── Pairwise Cohen's Kappa ─────────────────────────────────────────────────
    kappa_rows = []
    names_avail = list(pred_it.keys())
    logger.info("\nInter-method Cohen's Kappa:")
    for a, b in itertools.combinations(names_avail, 2):
        try:
            k_it = cohen_kappa_score(pred_it[a], pred_it[b])
            k_s  = cohen_kappa_score(pred_s[a],  pred_s[b])
            logger.info(f"  {a} vs {b}  —  info_type: {k_it:.3f}  security: {k_s:.3f}")
            kappa_rows.append({"method_a": a, "method_b": b,
                                "kappa_info_type": round(k_it, 3),
                                "kappa_security":  round(k_s, 3)})
        except Exception as e:
            logger.warning(f"  {a} vs {b}: Kappa failed — {e}")
    pd.DataFrame(kappa_rows).to_csv(kap_path, index=False)
    logger.info(f"Kappa CSV → {kap_path}")

    # ── Category distribution ──────────────────────────────────────────────────
    dist_rows = []
    for cat in INFO_TYPE_CLASSES:
        row = {"category": cat, "golden": y_golden_it.count(cat)}
        for key, name in zip(method_keys, method_names):
            row[name] = pred_it.get(name, []).count(cat)
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(dist_path, index=False)
    logger.info(f"\nDistribution CSV → {dist_path}")

    hdr = f"{'Category':<30} {'Golden':>7}" + "".join(f"{n:>12}" for n in method_names)
    logger.info("\n" + "─" * len(hdr))
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for _, row in dist_df.iterrows():
        line = f"{row['category']:<30} {row['golden']:>7}" + "".join(f"{row[n]:>12}" for n in method_names)
        logger.info(line)
    logger.info("─" * len(hdr))

    # ── Processing time comparison ─────────────────────────────────────────────
    logger.info("\nProcessing time (avg seconds per file):")
    for key, name in zip(method_keys, method_names):
        sec_col = f"{key}_sec"
        if sec_col in comp_df.columns:
            avg = comp_df[sec_col].mean()
            logger.info(f"  {name:<14}: {avg:.2f}s")

    # ── Overall accuracy summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("OVERALL ACCURACY AGAINST GOLDEN LABELS")
    logger.info("=" * 65)
    logger.info(f"{'Metric':<30}" + "".join(f"{n:>12}" for n in method_names))
    logger.info("─" * (30 + 12 * len(method_names)))

    info_accs = {}
    sec_accs  = {}
    for key, name in zip(method_keys, method_names):
        ic = f"{key}_info_correct"
        sc = f"{key}_sec_correct"
        info_accs[name] = comp_df[ic].mean() if ic in comp_df.columns else float("nan")
        sec_accs[name]  = comp_df[sc].mean() if sc in comp_df.columns else float("nan")

    logger.info(f"{'Info-type accuracy':<30}" + "".join(f"{info_accs[n]:>12.3f}" for n in method_names))
    logger.info(f"{'Security accuracy':<30}" + "".join(f"{sec_accs[n]:>12.3f}" for n in method_names))
    logger.info("=" * 65)
    logger.info(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
