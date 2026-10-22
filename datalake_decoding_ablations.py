# -*- coding: utf-8 -*-
"""
Step 7 — Decoding ablations for LLaMA 3.2 1B on TAB.

Separates model failure from harness failure across five conditions:
  a: baseline   — current prompt (verbatim from datalake_llm_pipeline.analyse_with_llm),
                   no retry, single parse attempt.
  b: + retry     — baseline prompt, up to 3 retries (4 attempts total) on parse failure.
  c: + json_mode — baseline prompt + Ollama's schema-constrained `format: "json"` decoding.
  d: + threshold — baseline prompt + an explicit sensitivity-threshold sentence.
  e: c + d combined.

Corpus: a fixed, seeded (42) 200-document subsample of TAB by default — sampled with the
SAME method as external/tab_adapter.py (sort all docs by doc_id, then Random(42).shuffle,
take the first N), so it's reproducible and consistent with the rest of the brief. Pass
--full to use the complete 1268 (cluster, walltime permitting).

Accuracy is scored against tab_golden's primary metric — security_confidential_only
(adjudicated) — restricted to subsample documents that actually have an adjudicated label
(reported explicitly, since not all do). Valid-output rate and accuracy are reported
separately per condition so a parse failure isn't conflated with a wrong classification.

Usage:
    python datalake_decoding_ablations.py --model llama3.2:1b
    python datalake_decoding_ablations.py --model llama3.2:1b --full --ollama-url http://localhost:11434
"""
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "external"))
import tab_adapter as ta  # noqa: E402

SUBSAMPLE_SEED = 42
SUBSAMPLE_N = 200
CONDITIONS = ["a", "b", "c", "d", "e"]

THRESHOLD_SENTENCE = (
    "\nIMPORTANT: set security_class=confidential if the content touches health, ethnicity, "
    "political opinion, religious/philosophical belief, or sex life/sexual orientation of an "
    "identifiable person — not only generic PII such as names or ID numbers. Use general only "
    "when none of these sensitive-category topics are present."
)


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("decoding_ablations")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def build_subsample(full: bool) -> list:
    docs = []
    for split in ta.SPLITS:
        d = ta.load_split(split)
        d.sort(key=lambda x: x["doc_id"])
        docs.extend(d)
    if full:
        return docs
    rng = random.Random(SUBSAMPLE_SEED)
    rng.shuffle(docs)
    return docs[:SUBSAMPLE_N]


def gold_label(doc: dict):
    return ta.conf_only_adjudicated(doc)  # None if no quality_checked -> no adjudicated label


def build_prompt(content: str, file_name: str, with_threshold: bool) -> str:
    """Verbatim copy of datalake_llm_pipeline.analyse_with_llm's prompt template (same prompt
    requirement across conditions), optionally appending the threshold sentence for d/e."""
    prompt = f"""You are a data governance analyst. Analyse the file content below and return ONLY a JSON object.

File name: {file_name}
File content (first 1200 chars):
{content[:1200]}

Return this JSON structure (no other text):
{{
  "info_type": "<identities|service_requests|financial_transactions|unknown>",
  "security_class": "<confidential|general>",
  "first_name": "<first name of person if found, else empty string>",
  "last_name": "<last name if found, else empty string>",
  "customer_id": "<any ID number found such as MANUFACLIC or account number, else empty>",
  "reason": "<one sentence explaining both classifications>"
}}

Classification rules:
- info_type=identities       : file contains a driver licence, passport, or other personal identity document
- info_type=service_requests : file contains a customer complaint, inquiry, or support interaction
- info_type=financial_transactions : file records payments, transactions, deposits, or withdrawals
- info_type=unknown          : does not clearly fit any above category
- security_class=confidential: file contains PII (names+IDs, DOB, address, account numbers)
- security_class=general     : no clearly personally identifiable or sensitive information

Respond with ONLY the JSON object."""
    if with_threshold:
        prompt += THRESHOLD_SENTENCE
    return prompt


def parse_json_from_response(raw: str) -> dict:
    """Verbatim copy of datalake_llm_pipeline.parse_json_from_response."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    return {}


def call_ollama(prompt: str, model: str, url: str, timeout: int, json_mode: bool, logger) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "num_predict": 500}}
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        logger.error(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"Ollama call error: {e!r}")
    return ""


def run_condition(cond: str, docs: list, model: str, url: str, timeout: int, logger) -> dict:
    json_mode = cond in ("c", "e")
    with_threshold = cond in ("d", "e")
    max_attempts = 4 if cond == "b" else 1  # b: 1 initial + up to 3 retries

    n_valid = 0
    n_scored = 0
    n_correct = 0
    per_doc_rows = []

    for i, doc in enumerate(docs):
        file_name = f"{doc['doc_id']}.txt"
        content = doc["text"]
        result = None
        attempts_used = 0
        for attempt in range(max_attempts):
            attempts_used += 1
            prompt = build_prompt(content, file_name, with_threshold)
            raw = call_ollama(prompt, model, url, timeout, json_mode, logger)
            parsed = parse_json_from_response(raw)
            if parsed and "security_class" in parsed:
                result = parsed
                break

        is_valid = result is not None
        pred = result.get("security_class") if is_valid else None
        gold = gold_label(doc)

        if is_valid:
            n_valid += 1
        scored = is_valid and gold is not None
        correct = scored and pred == gold
        if scored:
            n_scored += 1
            if correct:
                n_correct += 1

        per_doc_rows.append({
            "condition": cond, "doc_id": doc["doc_id"], "attempts_used": attempts_used,
            "valid_output": is_valid, "pred": pred, "gold_adjudicated": gold,
            "scored": scored, "correct": correct,
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(docs):
            logger.info(f"  condition {cond}: {i+1}/{len(docs)}  "
                        f"valid_rate_so_far={n_valid/(i+1):.3f}")

    valid_rate = round(n_valid / len(docs), 3) if docs else None
    accuracy = round(n_correct / n_scored, 3) if n_scored else None
    return {
        "condition": cond, "n_docs": len(docs), "n_valid_output": n_valid,
        "valid_output_rate": valid_rate, "n_scored_vs_adjudicated": n_scored,
        "n_correct": n_correct, "accuracy_vs_adjudicated": accuracy,
    }, per_doc_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--full", action="store_true", help="use the full 1268 docs instead of the 200-doc subsample")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    a = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = BASE_DIR / "evaluation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / f"log_decoding_ablations_{ts}.txt")

    logger.info("=" * 70)
    logger.info(f"DECODING ABLATIONS (Step 7)  model={a.model}  full={a.full}")
    logger.info("=" * 70)

    docs = build_subsample(a.full)
    n_with_label = sum(1 for d in docs if gold_label(d) is not None)
    logger.info(f"Corpus: {len(docs)} documents ({'full 1268' if a.full else f'seeded {SUBSAMPLE_N}-doc subsample, seed={SUBSAMPLE_SEED}'})")
    logger.info(f"Documents with an adjudicated label (scoreable): {n_with_label} / {len(docs)}")

    summary_rows = []
    all_per_doc = []
    conditions = a.conditions.split(",")
    for cond in conditions:
        t0 = time.time()
        summary, per_doc = run_condition(cond, docs, a.model, a.ollama_url, a.timeout, logger)
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        summary_rows.append(summary)
        all_per_doc.extend(per_doc)
        logger.info(f"CONDITION {cond} DONE: valid_rate={summary['valid_output_rate']}  "
                    f"accuracy={summary['accuracy_vs_adjudicated']}  "
                    f"n_scored={summary['n_scored_vs_adjudicated']}  "
                    f"elapsed={summary['elapsed_sec']}s")

    import pandas as pd
    summary_path = out_dir / f"decoding_ablations_{ts}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    detail_path = out_dir / f"decoding_ablations_detail_{ts}.csv"
    pd.DataFrame(all_per_doc).to_csv(detail_path, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for row in summary_rows:
        logger.info(f"  {row['condition']}: valid_rate={row['valid_output_rate']}  "
                    f"accuracy={row['accuracy_vs_adjudicated']}  n_scored={row['n_scored_vs_adjudicated']}")
    logger.info(f"Summary -> {summary_path}")
    logger.info(f"Detail  -> {detail_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
