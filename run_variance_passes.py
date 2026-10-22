"""
Run-to-run variance passes for the LLM pipeline (Part A of the Phi-3 variance brief).

Runs datalake_llm_pipeline.py unmodified, once per pass, in a child process. The child
wraps requests.post so that:
  * /api/generate gets options.temperature and options.seed set for this pass
    (prompt, num_predict, parser and fallback are untouched);
  * /api/pull is skipped, so the model cannot be updated mid-experiment.
The resolved weights blob digest is recorded before and after every pass.

Writes evaluation/output/variance_runs_<ts>.csv (one row per pass) which
score_variance_passes.py later joins with the scores.

Run:
  python run_variance_passes.py                  # full plan
  python run_variance_passes.py --models phi3:mini --passes 1
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
EVAL_OUT   = BASE_DIR / "evaluation" / "output"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

CHILD = r'''
import os, sys
sys.path.insert(0, os.environ["VARIANCE_REPO"])
import datalake_llm_pipeline as p
_orig_post = p.requests.post
T = float(os.environ["VARIANCE_TEMPERATURE"])
S = int(os.environ["VARIANCE_SEED"])
class _Skipped:
    status_code = 200
    text = "pull skipped by run_variance_passes"
def _post(url, json=None, **kw):
    if url.endswith("/api/pull"):
        return _Skipped()
    if url.endswith("/api/generate") and json is not None:
        json = dict(json)
        json["options"] = {**json.get("options", {}), "temperature": T, "seed": S}
    return _orig_post(url, json=json, **kw)
p.requests.post = _post
print(f"VARIANCE_TS={p.TIMESTAMP}", flush=True)
p.main()
'''


def blob_digest(model: str) -> str:
    # "name" for older servers, "model" for newer ones
    r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model, "name": model}, timeout=30)
    m = re.search(r"sha256-[0-9a-f]{64}", r.json().get("modelfile", ""))
    return m.group(0) if m else ""


def ollama_version() -> str:
    return requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json().get("version", "")


def total_sec_from_log(model: str, ts: str) -> str:
    slug = model.replace(":", "_").replace(".", "")
    log = OUTPUT_DIR / f"log_llm_{slug}_{ts}.txt"
    if log.exists():
        m = re.search(r"Total processing time\s*:\s*([\d.]+)s", log.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["phi3:mini", "llama3.2:1b"])
    ap.add_argument("--passes", type=int, default=5, help="passes at temperature 0.1")
    ap.add_argument("--t0-passes", type=int, default=2, help="control passes at temperature 0")
    ap.add_argument("--pass-prefix", default="", help="prefix for pass ids, e.g. v0342_")
    args = ap.parse_args()

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = EVAL_OUT / f"variance_runs_{run_ts}.csv"
    fields   = ["model", "pass", "temperature", "seed", "ollama_version", "run_ts", "blob_digest_before",
                "blob_digest_after", "pipeline_total_sec", "process_wallclock_sec", "returncode"]
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    plan = []
    for model in args.models:
        plan += [(model, f"{args.pass_prefix}p{i}", 0.1, i) for i in range(1, args.passes + 1)]
        plan += [(model, f"{args.pass_prefix}t0_{i}", 0.0, 0) for i in range(1, args.t0_passes + 1)]

    for model, pass_id, temp, seed in plan:
        env = {**os.environ, "OLLAMA_MODEL": model, "VARIANCE_REPO": str(BASE_DIR),
               "VARIANCE_TEMPERATURE": str(temp), "VARIANCE_SEED": str(seed),
               "PYTHONIOENCODING": "utf-8"}
        before = blob_digest(model)
        print(f"[{datetime.now():%H:%M:%S}] {model} {pass_id} T={temp} seed={seed} blob={before[:19]}", flush=True)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-c", CHILD], env=env, cwd=BASE_DIR,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        wall = round(time.time() - t0, 1)
        m = re.search(r"VARIANCE_TS=(\d{8}_\d{6})", proc.stdout)
        ts = m.group(1) if m else ""
        after = blob_digest(model)
        row = {"model": model, "pass": pass_id, "temperature": temp, "seed": seed,
               "ollama_version": ollama_version(), "run_ts": ts,
               "blob_digest_before": before, "blob_digest_after": after,
               "pipeline_total_sec": total_sec_from_log(model, ts),
               "process_wallclock_sec": wall, "returncode": proc.returncode}
        with manifest.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)
        print(f"    -> run_ts={ts} rc={proc.returncode} {wall}s", flush=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], flush=True)

    print(f"Manifest -> {manifest}")


if __name__ == "__main__":
    main()
