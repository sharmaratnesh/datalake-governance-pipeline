# -*- coding: utf-8 -*-
"""
Slim, 100%-stdlib TAB corpus runner for the PBS/GPU cluster (Step 5b).

datalake_llm_pipeline.py is not used here because it imports pandas/networkx/matplotlib/
PIL/pytesseract/faster_whisper — a heavy, unverified dependency set on an unfamiliar HPC
Python environment (observed: python3.6.8 on the login node). TAB documents are plain .txt
(no OCR/audio branches needed), so this script is intentionally self-contained: stdlib only
(urllib instead of requests), to maximise the chance it just runs as-is on the compute node.

The prompt template and JSON-parsing logic below are copied VERBATIM from
datalake_llm_pipeline.py's analyse_with_llm/parse_json_from_response (not imported, since
that would pull in the heavy deps above) — this is required for Step 5b's "same prompt,
same parser" comparability between the laptop and cluster/scale-probe runs. If the laptop
pipeline's prompt ever changes, this copy must be updated to match.

Checkpointing: appends one JSON line per processed file to --out-jsonl immediately, and
skips file_names already present on restart — same resume design as run_checkpointed_llm.py.

Usage:
    python ollama_tab_runner.py --model phi3:mini --corpus-dir /scratch/tab_corpus \
        --out-jsonl /scratch/results/phi3_mini.jsonl --platform hpc-gpu-phi3-mini
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
ap.add_argument("--platform", required=True)
ap.add_argument("--ollama-url", default="http://localhost:11434")
ap.add_argument("--timeout", type=int, default=180)
ap.add_argument("--require-digest", default="", metavar="SHA256",
                help="abort unless the pulled model resolves to this weights-blob digest")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

OLLAMA_URL = a.ollama_url
TIMEOUT_SEC = a.timeout
OLLAMA_MODEL = a.model


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
    log(f"Pulling/checking model {OLLAMA_MODEL} ...")
    try:
        _post_json(f"{OLLAMA_URL}/api/pull", {"name": OLLAMA_MODEL, "stream": False}, timeout=1800)
        log(f"Model {OLLAMA_MODEL} ready")
        return True
    except urllib.error.HTTPError as e:
        log(f"Model pull HTTP {e.code}: {e.read()[:200]}")
    except Exception as e:
        log(f"Model pull failed: {e!r}")
    return False


PROVENANCE = {"ollama_version": "", "model_digest": ""}


def ensure_model() -> bool:
    """Pull the model (a fresh container has none), then record and check what it resolved to."""
    if not _pull_model():
        return False
    try:
        PROVENANCE["ollama_version"] = json.loads(_get(f"{OLLAMA_URL}/api/version", 10)[1]).get("version", "")
        show = _post_json(f"{OLLAMA_URL}/api/show", {"model": OLLAMA_MODEL, "name": OLLAMA_MODEL}, timeout=60)
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


def call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.1, "num_predict": 500}}
    try:
        result = _post_json(f"{OLLAMA_URL}/api/generate", payload, timeout=TIMEOUT_SEC)
        return result.get("response", "").strip()
    except urllib.error.HTTPError as e:
        log(f"Ollama HTTP {e.code}: {e.read()[:200]}")
    except Exception as e:
        log(f"Ollama call error: {e!r}")
    return ""


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


def analyse_with_llm(content: str, file_name: str) -> dict:
    """Verbatim copy of datalake_llm_pipeline.analyse_with_llm's prompt (same prompt,
    same parser requirement, Step 5b). content[:1200] matches the laptop pipeline exactly."""
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

    raw = call_ollama(prompt)
    result = parse_json_from_response(raw)
    if not result:
        log(f"  JSON parse failed. Raw output: {raw[:300]}")
        result = {
            "info_type": "unknown", "security_class": "general",
            "first_name": "", "last_name": "", "customer_id": "",
            "reason": "LLM parse failure — defaulted to unknown/general",
        }
    return result


def load_done(out_jsonl: Path) -> set:
    if not out_jsonl.exists():
        return set()
    done = set()
    with open(out_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["file_name"])
                except Exception:
                    continue
    return done


def main():
    out_jsonl = Path(a.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    log(f"RUN model={a.model} corpus={a.corpus_dir} platform={a.platform}")

    for attempt in range(30):
        try:
            status, _ = _get(f"{OLLAMA_URL}/api/tags", timeout=5)
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
        log("FATAL: could not ensure model availability")
        sys.exit(1)

    files = sorted(p for p in Path(a.corpus_dir).glob("*.txt") if p.is_file())
    log(f"Files found: {len(files)}")

    done = load_done(out_jsonl)
    log(f"Already done (resume): {len(done)}")

    todo = [(i, f) for i, f in enumerate(files) if f.name not in done]
    if a.limit is not None:
        todo = todo[: a.limit]
    log(f"Remaining this invocation: {len(todo)}")

    t_start = time.time()
    for pos, (idx, fp) in enumerate(todo):
        t0 = time.time()
        started = datetime.now(timezone.utc).isoformat()
        content = fp.read_text(encoding="utf-8", errors="ignore")
        analysis = analyse_with_llm(content, fp.name)
        elapsed = round(time.time() - t0, 3)
        rec = {
            **PROVENANCE,
            "file_id": f"CLUSTER_{idx:04d}",
            "file_name": fp.name,
            "file_type": "text",
            "file_size_bytes": fp.stat().st_size,
            "info_type": analysis.get("info_type", "unknown"),
            "security_class": analysis.get("security_class", "general"),
            "first_name": analysis.get("first_name", ""),
            "last_name": analysis.get("last_name", ""),
            "llm_customer_id": analysis.get("customer_id", ""),
            "llm_reason": analysis.get("reason", ""),
            "method": f"llm_{a.model.replace(':', '_').replace('.', '')}",
            "model": a.model,
            "processing_sec": elapsed,
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "platform": a.platform,
        }
        with open(out_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if (pos + 1) % 25 == 0 or (pos + 1) == len(todo):
            rate = (pos + 1) / max(time.time() - t_start, 1e-9)
            log(f"Progress: {pos+1}/{len(todo)} this invocation "
                f"({len(done)+pos+1}/{len(files)} total)  rate={rate*60:.1f} files/min")

    log(f"Invocation complete. {len(done)+len(todo)}/{len(files)} total done.")


if __name__ == "__main__":
    main()
