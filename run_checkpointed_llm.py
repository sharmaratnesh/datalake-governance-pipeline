# -*- coding: utf-8 -*-
"""
Checkpointed driver for datalake_llm_pipeline.py (Step 5a: overnight full-corpus run).

datalake_llm_pipeline.py itself has no resume capability — it accumulates all records in
memory and writes the metadata CSV once at the end, so a crash partway through a ~6h run
loses everything. The brief forbids editing that file beyond the SWAMP_DIR line, so this
script does NOT modify it: it imports its module-level functions (extract_text,
analyse_with_llm, init_db, insert_file_record, build_graph, generate_owl, etc. are all
already free functions, not nested inside main()) and drives the same per-file logic
itself, appending each completed record to a stable JSONL checkpoint file immediately.

On restart, any file_name already present in the checkpoint is skipped, so the run resumes
instead of restarting. Each record also gets a wall-clock start/end ISO timestamp and a
platform tag, to support Step 5a/5c's per-record latency distribution and platform-specific
throughput reporting.

Usage:
    python run_checkpointed_llm.py --model phi3:mini --swamp-dir tab_corpus --tag tab_full
    python run_checkpointed_llm.py --model llama3.2:1b --swamp-dir tab_corpus --tag tab_full
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--swamp-dir", required=True)
ap.add_argument("--tag", required=True, help="stable identifier for checkpoint/output filenames")
ap.add_argument("--platform", default="laptop-cpu")
ap.add_argument("--limit", type=int, default=None, help="process at most N files this invocation")
a = ap.parse_args()

# Must set these env vars BEFORE importing the pipeline module — it reads them at import time.
os.environ["OLLAMA_MODEL"] = a.model
os.environ["SWAMP_DIR"] = a.swamp_dir

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
import datalake_llm_pipeline as llm  # noqa: E402  (must follow the env var assignment above)

MODEL_SLUG = a.model.replace(":", "_").replace(".", "")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stable (non-timestamped) paths so a resumed run reuses the same checkpoint/db.
checkpoint_path = OUTPUT_DIR / f"checkpoint_llm_{MODEL_SLUG}_{a.tag}.jsonl"
llm.DB_PATH = OUTPUT_DIR / f"db_llm_{MODEL_SLUG}_{a.tag}.db"

run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = BASE_DIR / "evaluation" / "output"
log_dir.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(f"checkpointed_llm_{MODEL_SLUG}_{a.tag}")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh = logging.FileHandler(log_dir / f"log_llm_overnight_{MODEL_SLUG}_{a.tag}_{run_ts}.txt", encoding="utf-8")
fh.setFormatter(fmt)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)


def load_done_file_names() -> set:
    if not checkpoint_path.exists():
        return set()
    done = set()
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["file_name"])
            except Exception:
                continue
    return done


def append_checkpoint(rec: dict) -> None:
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_all_checkpointed() -> list:
    rows = []
    with open(checkpoint_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    logger.info("=" * 70)
    logger.info(f"CHECKPOINTED LLM RUN  model={a.model}  swamp_dir={a.swamp_dir}  tag={a.tag}")
    logger.info(f"Checkpoint file: {checkpoint_path}")
    logger.info(f"DB file        : {llm.DB_PATH}")
    logger.info("=" * 70)

    if not llm.DB_PATH.exists():
        llm.init_db(llm.DB_PATH)
        logger.info("Initialised new DB")
    else:
        logger.info("Reusing existing DB (resume)")

    ollama_proc = llm.start_ollama(logger)
    if not llm.ensure_model(logger):
        sys.exit(2)
    llm.write_run_info(llm.DB_PATH, {**llm.run_info(), "tag": a.tag})

    swamp_files = sorted(f for f in Path(a.swamp_dir).glob("*") if f.is_file())
    if not swamp_files:
        # SWAMP_DIR may be relative; llm.SWAMP_DIR is the resolved absolute path.
        swamp_files = sorted(f for f in llm.SWAMP_DIR.glob("*") if f.is_file())
    logger.info(f"Files found: {len(swamp_files)}")

    done = load_done_file_names()
    logger.info(f"Already checkpointed (resume): {len(done)}")

    # (global_idx, file_path) pairs — global_idx is the file's stable position in the FULL
    # swamp_files list, so file_id stays unique across resumed invocations (a resume-local
    # index would collide with a prior invocation's ids and silently corrupt its records —
    # found and fixed during the smoke test).
    todo = [(i, f) for i, f in enumerate(swamp_files) if f.name not in done]
    if a.limit is not None:
        todo = todo[: a.limit]
    logger.info(f"Remaining this invocation: {len(todo)}")

    t_run_start = time.time()
    n_done_this_run = 0

    for pos, (idx, file_path) in enumerate(todo):
        t0 = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()
        try:
            content, file_type = llm.extract_text(file_path, logger)
            if file_type == "unsupported" or not content.strip():
                logger.info(f"  [{pos+1}/{len(todo)}] {file_path.name}: skipped (unsupported/empty)")
                continue

            analysis = llm.analyse_with_llm(content, file_path.name, logger)
            info_type = analysis.get("info_type", "unknown")
            security = analysis.get("security_class", "general")
            first_name = analysis.get("first_name", "").strip()
            last_name = analysis.get("last_name", "").strip()
            reason = analysis.get("reason", "")

            customer_id = None
            complaint_id = None
            if info_type == "identities" and first_name:
                full_name = f"{first_name} {last_name}".strip()
                cid = first_name.strip().capitalize()
                llm.upsert_customer(llm.DB_PATH, cid, full_name)
                customer_id = cid
            if info_type == "service_requests" and first_name:
                customer_id = llm.lookup_customer_by_firstname(llm.DB_PATH, first_name)
                if customer_id:
                    complaint_id = llm.insert_complaint(llm.DB_PATH, customer_id, content, str(file_path))

            elapsed = round(time.time() - t0, 3)
            ended_iso = datetime.now(timezone.utc).isoformat()
            rec = {
                "file_id": f"FILE_{a.tag}_{MODEL_SLUG}_{idx:04d}",
                "file_name": file_path.name,
                "file_type": file_type,
                "file_size_bytes": file_path.stat().st_size,
                "info_type": info_type,
                "security_class": security,
                "customer_id": customer_id or "",
                "complaint_id": complaint_id or "",
                "llm_reason": reason,
                "method": f"llm_{MODEL_SLUG}",
                "model": a.model,
                "ollama_version": llm.PROVENANCE["ollama_version"],
                "model_digest": llm.PROVENANCE["model_digest"],
                "processing_sec": elapsed,
                "content_excerpt": content[:120].replace("\n", " ").strip(),
                "started_utc": started_iso,
                "ended_utc": ended_iso,
                "platform": a.platform,
            }
            llm.insert_file_record(llm.DB_PATH, rec)
            append_checkpoint(rec)
            n_done_this_run += 1

            if (pos + 1) % 25 == 0 or (pos + 1) == len(todo):
                rate = n_done_this_run / max(time.time() - t_run_start, 1e-9)
                logger.info(f"  Progress: {pos+1}/{len(todo)} this invocation "
                            f"({len(done) + n_done_this_run}/{len(swamp_files)} total)  "
                            f"rate={rate*60:.1f} files/min")
        except Exception as e:
            logger.error(f"  [{pos+1}/{len(todo)}] {file_path.name}: UNHANDLED ERROR {e!r} — "
                         "recording as parse-failure default and continuing")
            elapsed = round(time.time() - t0, 3)
            rec = {
                "file_id": f"FILE_{a.tag}_{MODEL_SLUG}_{idx:04d}",
                "file_name": file_path.name,
                "file_type": "error",
                "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "info_type": "unknown",
                "security_class": "general",
                "customer_id": "",
                "complaint_id": "",
                "llm_reason": f"pipeline error: {e!r}",
                "method": f"llm_{MODEL_SLUG}",
                "model": a.model,
                "ollama_version": llm.PROVENANCE["ollama_version"],
                "model_digest": llm.PROVENANCE["model_digest"],
                "processing_sec": elapsed,
                "content_excerpt": "",
                "started_utc": started_iso,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "platform": a.platform,
            }
            append_checkpoint(rec)
            n_done_this_run += 1

    total_sec = round(time.time() - t_run_start, 2)
    logger.info(f"Invocation complete: {n_done_this_run} files processed in {total_sec}s")

    all_rows = load_all_checkpointed()
    logger.info(f"Total checkpointed so far: {len(all_rows)} / {len(swamp_files)}")

    if len(all_rows) >= len(swamp_files):
        import pandas as pd
        final_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"metadata_llm_{MODEL_SLUG}_{final_ts}.csv"
        df = pd.DataFrame(all_rows)
        df.drop(columns=["file_id", "content_excerpt"], errors="ignore").to_csv(csv_path, index=False)
        logger.info(f"ALL FILES DONE. Final metadata CSV -> {csv_path}")

        graph_path = OUTPUT_DIR / f"graph_llm_{MODEL_SLUG}_{final_ts}.png"
        owl_path = OUTPUT_DIR / f"ontology_llm_{MODEL_SLUG}_{final_ts}.owl"
        try:
            G = llm.build_graph(llm.DB_PATH)
            llm.save_graph(G, graph_path, f"Entity Graph — LLM ({a.model}, {a.tag})", logger)
            cats = list({r["info_type"] for r in all_rows})
            llm.generate_owl(owl_path, cats)
            logger.info(f"Graph -> {graph_path}")
            logger.info(f"Ontology -> {owl_path}")
        except Exception as e:
            logger.error(f"Graph/ontology generation failed (non-fatal): {e!r}")
    else:
        logger.info(f"NOT YET COMPLETE — {len(swamp_files) - len(all_rows)} files remaining. "
                     "Re-run this command to resume.")

    if ollama_proc:
        ollama_proc.terminate()

    print(f"\nCheckpointed run complete for this invocation. "
          f"{len(all_rows)}/{len(swamp_files)} total files done. Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
