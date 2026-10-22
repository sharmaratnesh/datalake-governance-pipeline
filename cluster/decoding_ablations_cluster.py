# -*- coding: utf-8 -*-
"""
Step 7 — Decoding ablations, cluster-side (100% stdlib, mirrors cluster/ollama_tab_runner.py's
constraints). Runs all five conditions over the full TAB corpus on the GPU and writes RAW
per-document results only (valid_output, pred, attempts_used) — NOT accuracy, since the gold
adjudicated labels live in the TAB JSON annotations which are not staged on the cluster.
Accuracy is scored back on the laptop by datalake_decoding_ablations.py's scoring logic,
joining these raw predictions against tab_golden.xlsx.

Conditions (same prompt as datalake_llm_pipeline.analyse_with_llm, copied verbatim):
  a: baseline — no retry, single attempt
  b: + up to 3 retries (4 attempts total) on parse failure
  c: + Ollama schema-constrained `format: "json"`
  d: + explicit sensitivity-threshold sentence appended to the prompt
  e: c + d combined

Usage:
    python decoding_ablations_cluster.py --model llama3.2:1b --corpus-dir /scratch/corpus \
        --out-jsonl /scratch/results/ablations.jsonl --conditions a,b,c,d,e
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--corpus-dir", required=True)
ap.add_argument("--out-jsonl", required=True)
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--timeout", type=int, default=180)
ap.add_argument("--require-digest", default="", metavar="SHA256",
                help="abort unless the pulled model resolves to this weights-blob digest")
ap.add_argument("--conditions", default="a,b,c,d,e")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

THRESHOLD_SENTENCE = (
    "\nIMPORTANT: set security_class=confidential if the content touches health, ethnicity, "
    "political opinion, religious/philosophical belief, or sex life/sexual orientation of an "
    "identifiable person — not only generic PII such as names or ID numbers. Use general only "
    "when none of these sensitive-category topics are present."
)


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} {msg}", flush=True)


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body) if body else {}


def _get(url: str, timeout: int):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _pull_model() -> bool:
    log(f"Pulling/checking model {a.model} ...")
    try:
        _post_json(f"{a.ollama_url}/api/pull", {"name": a.model, "stream": False}, timeout=1800)
        log("Model ready")
        return True
    except Exception as e:
        log(f"Model pull failed: {e!r}")
        return False


PROVENANCE = {"ollama_version": "", "model_digest": ""}


def ensure_model() -> bool:
    """Pull the model (a fresh container has none), then record and check what it resolved to."""
    if not _pull_model():
        return False
    try:
        PROVENANCE["ollama_version"] = json.loads(_get(f"{a.ollama_url}/api/version", 10)[1]).get("version", "")
        show = _post_json(f"{a.ollama_url}/api/show", {"model": a.model, "name": a.model}, timeout=60)
        m = re.search(r"sha256-[0-9a-f]{64}", show.get("modelfile", ""))
        PROVENANCE["model_digest"] = m.group(0) if m else ""
    except Exception as e:
        log(f"Could not resolve model provenance: {e!r}")
    log(f"Ollama version {PROVENANCE['ollama_version'] or 'unknown'}, model digest {PROVENANCE['model_digest'] or 'unknown'}")
    want = a.require_digest.strip().replace("sha256:", "sha256-")
    if want and PROVENANCE["model_digest"] != want:
        log(f"FATAL: model digest {PROVENANCE['model_digest']} != --require-digest {want}")
        return False
    return True


def call_ollama(prompt: str, json_mode: bool) -> str:
    payload = {"model": a.model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "num_predict": 500}}
    if json_mode:
        payload["format"] = "json"
    try:
        result = _post_json(f"{a.ollama_url}/api/generate", payload, timeout=a.timeout)
        return result.get("response", "").strip()
    except Exception as e:
        log(f"Ollama call error: {e!r}")
        return ""


def parse_json_from_response(raw: str) -> dict:
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    return {}


def build_prompt(content: str, file_name: str, with_threshold: bool) -> str:
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


def load_done(out_jsonl: Path) -> set:
    if not out_jsonl.exists():
        return set()
    done = set()
    with open(out_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add((r["condition"], r["doc_id"]))
                except Exception:
                    continue
    return done


def run_condition(cond: str, files: list, done: set, out_jsonl: Path):
    json_mode = cond in ("c", "e")
    with_threshold = cond in ("d", "e")
    max_attempts = 4 if cond == "b" else 1

    todo = [f for f in files if (cond, f.stem) not in done]
    log(f"Condition {cond}: {len(todo)} remaining of {len(files)}")
    t0 = time.time()
    for i, fp in enumerate(todo):
        content = fp.read_text(encoding="utf-8", errors="ignore")
        result = None
        attempts_used = 0
        for _ in range(max_attempts):
            attempts_used += 1
            prompt = build_prompt(content, fp.name, with_threshold)
            raw = call_ollama(prompt, json_mode)
            parsed = parse_json_from_response(raw)
            if parsed and "security_class" in parsed:
                result = parsed
                break
        rec = {
            **PROVENANCE,
            "condition": cond, "doc_id": fp.stem, "attempts_used": attempts_used,
            "valid_output": result is not None,
            "pred": (result or {}).get("security_class"),
        }
        with open(out_jsonl, "a", encoding="utf-8") as out:
            out.write(json.dumps(rec) + "\n")
        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            log(f"  condition {cond}: {i+1}/{len(todo)}  rate={rate*60:.1f}/min")


def main():
    out_jsonl = Path(a.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(30):
        try:
            status, _ = _get(f"{a.ollama_url}/api/tags", timeout=5)
            if status == 200:
                log("Ollama server reachable")
                break
        except Exception:
            pass
        log(f"Waiting for Ollama server... ({attempt+1}/30)")
        time.sleep(5)
    else:
        log("FATAL: Ollama server never became reachable")
        sys.exit(1)

    if not ensure_model():
        sys.exit(1)

    files = sorted(p for p in Path(a.corpus_dir).glob("*.txt") if p.is_file())
    if a.limit is not None:
        files = files[: a.limit]
    log(f"Corpus: {len(files)} files")

    done = load_done(out_jsonl)
    log(f"Already done (resume): {len(done)}")

    for cond in a.conditions.split(","):
        run_condition(cond, files, done, out_jsonl)

    log("ALL CONDITIONS COMPLETE")


if __name__ == "__main__":
    main()
