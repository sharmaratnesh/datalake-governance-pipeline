# -*- coding: utf-8 -*-
"""
Step 4 — Conventional supervised baselines (TF-IDF features; LogisticRegression, LinearSVC,
DummyClassifier(strategy="most_frequent") as the majority-class floor).

Stratified 5-fold cross-validation, random_state=42, evaluated with out-of-fold predictions
(cross_val_predict) so every sample contributes one held-out prediction. The TF-IDF vectorizer
is refit inside each fold (via a Pipeline) to avoid train/test leakage.

Run on both corpora:
  - Ours (data_swamp, 55 files): info_type (4-class) and security_classification (binary).
    Text is re-extracted from the raw files via datalake_nlp_pipeline.extract_text (imported,
    not modified) — OCR for images, Whisper for audio, as the pipelines themselves do.
  - TAB (tab_corpus): security_confidential_only, across all three Step 2b derivations —
    adjudicated (primary, scored only on the 582-doc quality_checked subset where the label
    exists), union, and intersection (both over the full 1268).

Reports, per (dataset, task, model): accuracy, Wilson 95% CI on accuracy, macro precision/
recall/F1, Cohen's kappa, and per-class precision/recall/F1/support.
Output: evaluation/output/classical_baselines_<ts>.csv
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from statsmodels.stats.proportion import proportion_confint

BASE_DIR = Path(__file__).parent
RANDOM_STATE = 42
N_FOLDS = 5


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("classical_baselines")
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


def make_models():
    return [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("LinearSVC", lambda: LinearSVC(random_state=RANDOM_STATE)),
        ("DummyClassifier", lambda: DummyClassifier(strategy="most_frequent")),
    ]


def min_class_count(y) -> int:
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min())


def run_cv_task(texts: list, labels: list, dataset: str, task: str, logger: logging.Logger) -> list:
    y = np.asarray(labels)
    n = len(y)
    classes = sorted(set(labels))
    n_folds = min(N_FOLDS, min_class_count(y))
    if n_folds < 2:
        logger.warning(f"{dataset} | {task}: smallest class has <2 members — skipping (n={n}, classes={classes})")
        return []
    if n_folds < N_FOLDS:
        logger.warning(f"{dataset} | {task}: smallest class has only {n_folds} members — "
                        f"using {n_folds}-fold instead of {N_FOLDS}-fold")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for model_name, model_fn in make_models():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", model_fn()),
        ])
        preds = cross_val_predict(pipe, texts, y, cv=skf)

        acc = accuracy_score(y, preds)
        lo, hi = wilson_ci(int(round(acc * n)), n)
        prec, rec, f1, sup = precision_recall_fscore_support(y, preds, labels=classes, zero_division=0)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y, preds, labels=classes, average="macro", zero_division=0)
        try:
            kappa = round(float(cohen_kappa_score(y, preds, labels=classes)), 3)
        except Exception:
            kappa = None

        rows.append({
            "dataset": dataset, "task": task, "model": model_name, "class": "__overall__",
            "n": n, "n_folds": n_folds,
            "accuracy": round(acc, 3), "accuracy_wilson_lo": lo, "accuracy_wilson_hi": hi,
            "macro_precision": round(macro_p, 3), "macro_recall": round(macro_r, 3),
            "macro_f1": round(macro_f1, 3), "cohens_kappa": kappa,
            "precision": None, "recall": None, "f1": None, "support": n,
        })
        for cls, p, r, f, s in zip(classes, prec, rec, f1, sup):
            rows.append({
                "dataset": dataset, "task": task, "model": model_name, "class": cls,
                "n": None, "n_folds": n_folds,
                "accuracy": None, "accuracy_wilson_lo": None, "accuracy_wilson_hi": None,
                "macro_precision": None, "macro_recall": None, "macro_f1": None, "cohens_kappa": None,
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": int(s),
            })
        logger.info(f"{dataset:<12} {task:<28} {model_name:<20} "
                    f"acc={acc:.3f} [{lo},{hi}]  macro_f1={macro_f1:.3f}  kappa={kappa}  n={n} folds={n_folds}")
    return rows


def read_golden_sheet(xlsx_path: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["evaluation"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    header, data = rows[0], rows[1:]
    df = pd.DataFrame(data, columns=header)
    return df[df["File Name"].notna()]


def extract_our_corpus_texts(logger: logging.Logger) -> dict:
    """file_name -> extracted text, via the NLP pipeline's own extract_text (imported, not modified)."""
    import datalake_nlp_pipeline as nlp
    swamp_dir = BASE_DIR / "data_swamp"
    texts = {}
    for f in sorted(swamp_dir.glob("*")):
        if not f.is_file():
            continue
        content, _ = nlp.extract_text(f, logger)
        texts[f.name] = content or ""
    return texts


def build_our_corpus_rows(logger: logging.Logger) -> pd.DataFrame:
    golden = read_golden_sheet(BASE_DIR / "evaluation" / "evaluation_template_golden.xlsx")
    texts = extract_our_corpus_texts(logger)
    golden["text"] = golden["File Name"].map(texts)
    before = len(golden)
    golden = golden[golden["text"].notna() & (golden["text"].str.strip() != "")]
    if len(golden) < before:
        logger.warning(f"Our corpus: dropped {before - len(golden)} file(s) with no extractable text")
    return golden


def build_tab_rows(logger: logging.Logger) -> pd.DataFrame:
    golden = read_golden_sheet(BASE_DIR / "evaluation" / "tab_golden.xlsx")
    corpus_dir = BASE_DIR / "tab_corpus"

    def read_text(file_name):
        p = corpus_dir / file_name
        return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None

    golden["text"] = golden["File Name"].map(read_text)
    before = len(golden)
    golden = golden[golden["text"].notna() & (golden["text"].str.strip() != "")]
    if len(golden) < before:
        logger.warning(f"TAB: dropped {before - len(golden)} file(s) with no text in tab_corpus/ "
                        "(adapter likely run with --limit; re-run without --limit for the full run)")
    return golden


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_classical_baselines_{ts}.txt")

    logger.info("=" * 70)
    logger.info("CLASSICAL BASELINES (Step 4)")
    logger.info("=" * 70)

    all_rows = []

    # ---- Our corpus: info_type (4-class) and security (binary) ----
    logger.info("Building our-corpus text + labels (OCR/Whisper extraction)...")
    ours = build_our_corpus_rows(logger)
    logger.info(f"Our corpus: {len(ours)} files with text")
    all_rows += run_cv_task(ours["text"].tolist(), ours["Data Class Category"].tolist(),
                             "ours", "info_type", logger)
    all_rows += run_cv_task(ours["text"].tolist(), ours["Security Classification"].tolist(),
                             "ours", "security_classification", logger)

    # ---- TAB: security_confidential_only, three derivations ----
    logger.info("Loading TAB text + labels...")
    tab = build_tab_rows(logger)
    logger.info(f"TAB: {len(tab)} files with text")

    adj_col = "Security Confidential Only (Adjudicated)"
    adj = tab[tab[adj_col].notna() & (tab[adj_col].astype(str).str.strip() != "")]
    logger.info(f"TAB adjudicated subset (primary): {len(adj)} files")
    all_rows += run_cv_task(adj["text"].tolist(), adj[adj_col].tolist(),
                             "tab", "security_confidential_only_adjudicated", logger)

    all_rows += run_cv_task(tab["text"].tolist(), tab["Security Confidential Only (Union)"].tolist(),
                             "tab", "security_confidential_only_union", logger)
    all_rows += run_cv_task(tab["text"].tolist(), tab["Security Confidential Only (Intersection)"].tolist(),
                             "tab", "security_confidential_only_intersection", logger)

    out_path = out_dir / f"classical_baselines_{ts}.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    logger.info(f"\nOutput -> {out_path}  ({len(all_rows)} rows)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
