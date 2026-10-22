"""
Data Lake LLM Pipeline  —  Experiment 1, RO1

Processes all files in data_swamp/ using a locally-running LLM (Ollama).
No paid API calls are made — all inference is local.

Processing steps per file:
  • OCR (pytesseract) for images
  • Google Speech Recognition for audio (free tier)
  • Ollama LLM for thematic categorisation, security classification,
    entity extraction, and classification rationale
  • SQLite relational store
  • NetworkX + Matplotlib entity-relationship graph
  • OWL RDF/XML ontology generation

Configuration via environment variables:
  OLLAMA_MODEL          default: llama3.2:1b
  OLLAMA_URL            default: http://localhost:11434
  OLLAMA_TIMEOUT        default: 180   (seconds per LLM call)
  OLLAMA_ALLOW_PULL     default: 0     (1 = let the pipeline pull/update the model tag)
  OLLAMA_REQUIRE_DIGEST default: ""    (abort unless the model's weights blob has this digest)

The pipeline never pulls a model unless asked to: a pull silently re-resolves the tag to
whatever the registry serves that day. It logs the Ollama server version and the resolved
weights-blob digest at start-up and writes both into every metadata row and into the
run_info table of the run's DB. The paper's Phi-3 Mini runs used
  sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf

Outputs written to  output/  :
  metadata_llm_<model>_<timestamp>.csv
  graph_llm_<model>_<timestamp>.png
  ontology_llm_<model>_<timestamp>.owl
  log_llm_<model>_<timestamp>.txt

Run:
  python datalake_llm_pipeline.py
  OLLAMA_MODEL=phi3:mini python datalake_llm_pipeline.py \
      --require-digest sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf
  python datalake_llm_pipeline.py --allow-pull      # opt in to pulling the tag
"""

import argparse
import os
import sys
import json
import logging
import re
import sqlite3
import subprocess
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False
    _WHISPER_MODEL = None

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
SWAMP_DIR     = BASE_DIR / os.environ.get("SWAMP_DIR", "data_swamp")
OUTPUT_DIR    = BASE_DIR / "output"
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434")
TIMEOUT_SEC   = int(os.environ.get("OLLAMA_TIMEOUT", "180"))
ALLOW_PULL    = os.environ.get("OLLAMA_ALLOW_PULL", "0") == "1"
REQUIRE_DIGEST = os.environ.get("OLLAMA_REQUIRE_DIGEST", "")
PROVENANCE    = {"ollama_version": "", "model_digest": ""}   # filled by ensure_model()
TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_SLUG    = OLLAMA_MODEL.replace(":", "_").replace(".", "")
METHOD        = f"llm_{MODEL_SLUG}"
DB_PATH       = OUTPUT_DIR / f"datalake_llm_{MODEL_SLUG}_{TIMESTAMP}.db"

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("llm_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    import io
    sh = logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace"))
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ── Ollama helpers ────────────────────────────────────────────────────────────
def start_ollama(logger: logging.Logger) -> subprocess.Popen | None:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            logger.info("Ollama server already running")
            return None
    except Exception:
        pass
    logger.info("Starting Ollama server...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    return proc

def resolve_model() -> dict:
    """Ollama server version and the weights-blob digest the model tag resolves to locally
    ("" if the model is not present)."""
    out = {"ollama_version": "", "model_digest": ""}
    try:
        out["ollama_version"] = requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json().get("version", "")
        # "name" for older servers, "model" for newer ones
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": OLLAMA_MODEL, "name": OLLAMA_MODEL}, timeout=30)
        m = re.search(r"sha256-[0-9a-f]{64}", r.json().get("modelfile", "")) if r.status_code == 200 else None
        out["model_digest"] = m.group(0) if m else ""
    except Exception:
        pass
    return out

def ensure_model(logger: logging.Logger) -> bool:
    """Resolve (and only if ALLOW_PULL, pull) the model; record and check its provenance."""
    if ALLOW_PULL:
        logger.info(f"Pulling model {OLLAMA_MODEL} (--allow-pull)")
        try:
            requests.post(f"{OLLAMA_URL}/api/pull", json={"name": OLLAMA_MODEL, "stream": False}, timeout=600)
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
    PROVENANCE.update(resolve_model())
    logger.info(f"Ollama version  : {PROVENANCE['ollama_version'] or 'unknown'}")
    logger.info(f"Model digest    : {PROVENANCE['model_digest'] or 'not found'}")
    if not PROVENANCE["model_digest"]:
        logger.error(f"Model {OLLAMA_MODEL} is not available locally and pulling is disabled. "
                     f"Run `ollama pull {OLLAMA_MODEL}` yourself and check its digest, or pass --allow-pull.")
        return False
    want = REQUIRE_DIGEST.strip().replace("sha256:", "sha256-")
    if want and PROVENANCE["model_digest"] != want:
        logger.error(f"Model digest mismatch: {OLLAMA_MODEL} resolves to {PROVENANCE['model_digest']}, "
                     f"--require-digest is {want}. Aborting.")
        return False
    logger.info(f"Model {OLLAMA_MODEL} ready")
    return True

def write_run_info(db_path: Path, info: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS run_info (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT OR REPLACE INTO run_info (key, value) VALUES (?,?)",
                         [(k, str(v)) for k, v in info.items()])
        conn.commit()

def run_info() -> dict:
    return {"model": OLLAMA_MODEL, **PROVENANCE, "temperature": 0.1, "num_predict": 500}

def call_ollama(prompt: str, logger: logging.Logger) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 500},
    }
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate", json=payload, timeout=TIMEOUT_SEC
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        logger.error(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"Ollama call error: {e}")
    return ""

def parse_json_from_response(raw: str) -> dict:
    """Extract first valid JSON object from LLM output."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                return obj
            except json.JSONDecodeError:
                continue
    return {}

def analyse_with_llm(content: str, file_name: str, logger: logging.Logger) -> dict:
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

    raw = call_ollama(prompt, logger)
    result = parse_json_from_response(raw)
    if not result:
        logger.warning(f"  JSON parse failed. Raw output: {raw[:300]}")
        result = {
            "info_type": "unknown",
            "security_class": "general",
            "first_name": "",
            "last_name": "",
            "customer_id": "",
            "reason": "LLM parse failure — defaulted to unknown/general",
        }
    return result

# ── Content extraction (identical logic to NLP pipeline) ─────────────────────
def extract_text(file_path: Path, logger: logging.Logger) -> tuple[str, str]:
    ext = file_path.suffix.lower()

    if ext == ".txt":
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            logger.info(f"  [TEXT] {len(content)} chars")
            return content, "text"
        except Exception as e:
            logger.error(f"  [TEXT] error: {e}")
            return "", "text"

    elif ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            content = df.to_string()
            logger.info(f"  [CSV] {len(df)} rows, cols={list(df.columns)}")
            return content, "csv"
        except Exception as e:
            logger.error(f"  [CSV] error: {e}")
            return "", "csv"

    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        if not OCR_AVAILABLE:
            logger.warning("  [IMAGE] pytesseract unavailable")
            return "", "image"
        try:
            with Image.open(file_path) as img:
                content = pytesseract.image_to_string(img)
            logger.info(f"  [IMAGE/OCR] {len(content)} chars extracted")
            return content, "image"
        except Exception as e:
            logger.error(f"  [IMAGE] OCR error: {e}")
            return "", "image"

    elif ext in {".wav", ".mp3", ".m4a", ".flac"}:
        if not WHISPER_AVAILABLE:
            logger.warning("  [AUDIO] faster-whisper unavailable")
            return "", "audio"
        try:
            segments, _ = _WHISPER_MODEL.transcribe(str(file_path))
            content = " ".join(seg.text for seg in segments).strip()
            logger.info(f"  [AUDIO/Whisper] '{content[:80]}{'...' if len(content)>80 else ''}'")
            return content, "audio"
        except Exception as e:
            logger.warning(f"  [AUDIO] transcription failed: {e}")
            return "", "audio"

    elif ext == ".json":
        try:
            data = json.loads(file_path.read_text(encoding="utf-8", errors="ignore"))
            content = json.dumps(data, indent=2)
            logger.info(f"  [JSON] {len(content)} chars")
            return content, "json"
        except Exception as e:
            logger.error(f"  [JSON] {e}")
            return "", "json"

    logger.info(f"  [SKIP] unsupported: {ext}")
    return "", "unsupported"

# ── SQLite database ───────────────────────────────────────────────────────────
def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE customers (
                customer_id   TEXT PRIMARY KEY,
                name          TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE customer_complaints (
                complaint_id  TEXT PRIMARY KEY,
                customer_id   TEXT REFERENCES customers(customer_id),
                complaint_text TEXT,
                file_source   TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE data_files (
                file_id         TEXT PRIMARY KEY,
                file_name       TEXT,
                file_type       TEXT,
                info_type       TEXT,
                security_class  TEXT,
                customer_id     TEXT REFERENCES customers(customer_id),
                complaint_id    TEXT,
                llm_reason      TEXT,
                processing_sec  REAL,
                content_excerpt TEXT,
                created_at      TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

def upsert_customer(db_path: Path, cid: str, name: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO customers (customer_id, name) VALUES (?,?)", (cid, name)
        )
        conn.commit()

def insert_complaint(db_path: Path, cid: str, text: str, source: str) -> str:
    comp_id = f"COMP{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO customer_complaints (complaint_id, customer_id, complaint_text, file_source) VALUES (?,?,?,?)",
            (comp_id, cid, text[:500], source),
        )
        conn.commit()
    return comp_id

def lookup_customer_by_firstname(db_path: Path, first_name: str) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT customer_id FROM customers WHERE name LIKE ?", (f"{first_name}%",)
        ).fetchone()
    return row[0] if row else None

def insert_file_record(db_path: Path, rec: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO data_files
               (file_id, file_name, file_type, info_type, security_class,
                customer_id, complaint_id, llm_reason, processing_sec, content_excerpt)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                rec["file_id"], rec["file_name"], rec["file_type"],
                rec["info_type"], rec["security_class"],
                rec["customer_id"] or None, rec["complaint_id"] or None,
                rec["llm_reason"], rec["processing_sec"], rec["content_excerpt"],
            ),
        )
        conn.commit()

# ── Knowledge graph (shared design with NLP pipeline) ────────────────────────
NODE_COLORS = {
    "Customer":                        "#4472C4",
    "Complaint":                       "#ED7D31",
    "DataFile_identities":             "#70AD47",
    "DataFile_service_requests":       "#FF0000",
    "DataFile_financial_transactions": "#FFD966",
    "DataFile_unknown":                "#BFBFBF",
}

def build_graph(db_path: Path) -> nx.DiGraph:
    G = nx.DiGraph()
    with sqlite3.connect(db_path) as conn:
        for cid, name in conn.execute("SELECT customer_id, name FROM customers"):
            if name:
                label = name
            elif "<" in cid or ">" in cid or "|" in cid or "," in cid or cid.count(" ") > 2 or len(cid) > 24:
                label = "Unresolved ID"
            else:
                label = cid
            G.add_node(f"cust:{cid}", label=label, node_type="Customer")

        for comp_id, cid in conn.execute(
            "SELECT complaint_id, customer_id FROM customer_complaints"
        ):
            G.add_node(f"comp:{comp_id}", label=comp_id[:14], node_type="Complaint")
            if cid:
                G.add_edge(f"cust:{cid}", f"comp:{comp_id}", relation="HAS_COMPLAINT")

        for fid, fname, cat, sec, cid in conn.execute(
            "SELECT file_id, file_name, info_type, security_class, customer_id FROM data_files"
        ):
            G.add_node(f"file:{fid}", label=fname[:22], node_type=f"DataFile_{cat}", security=sec)
            if cid:
                G.add_edge(f"cust:{cid}", f"file:{fid}", relation="RELATED_TO")
    return G

def save_graph(G: nx.DiGraph, graph_path: Path, title: str, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    pos = nx.spring_layout(G, seed=42, k=2.8)
    color_map = [NODE_COLORS.get(G.nodes[n].get("node_type", ""), "#BFBFBF") for n in G.nodes]
    labels    = {n: G.nodes[n].get("label", n) for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax, font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowsize=22,
        edge_color="#444444", width=1.8, connectionstyle="arc3,rad=0.12",
    )
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    patches = [
        mpatches.Patch(
            color=c, label=lbl.replace("DataFile_", "File — ").replace("_", " ")
        )
        for lbl, c in NODE_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Graph saved → {graph_path}")

# ── OWL ontology ──────────────────────────────────────────────────────────────
def generate_owl(owl_path: Path, categories: list[str]) -> None:
    body = ""
    for cat in categories:
        if cat != "unknown":
            cls = "".join(w.capitalize() for w in cat.split("_"))
            body += (
                f'    <owl:Class rdf:about="#{cls}">'
                f'<rdfs:subClassOf rdf:resource="#DataFile"/>'
                f'<rdfs:label>{cat.replace("_"," ").title()}</rdfs:label>'
                f"</owl:Class>\n"
            )
    owl = f"""<?xml version="1.0"?>
<rdf:RDF xmlns="http://www.datalake.org/ontology#"
     xml:base="http://www.datalake.org/ontology"
     xmlns:owl="http://www.w3.org/2002/07/owl#"
     xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
     xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
     xmlns:xsd="http://www.w3.org/2001/XMLSchema#">
    <owl:Ontology rdf:about="http://www.datalake.org/ontology"/>
    <owl:Class rdf:about="#Customer"><rdfs:label>Customer</rdfs:label></owl:Class>
    <owl:Class rdf:about="#DataFile"><rdfs:label>Data File</rdfs:label></owl:Class>
    <owl:Class rdf:about="#Complaint"><rdfs:label>Complaint</rdfs:label></owl:Class>
{body}
    <owl:ObjectProperty rdf:about="#hasComplaint">
        <rdfs:domain rdf:resource="#Customer"/>
        <rdfs:range rdf:resource="#Complaint"/>
    </owl:ObjectProperty>
    <owl:ObjectProperty rdf:about="#relatedToCustomer">
        <rdfs:domain rdf:resource="#DataFile"/>
        <rdfs:range rdf:resource="#Customer"/>
    </owl:ObjectProperty>
    <owl:DatatypeProperty rdf:about="#customerName">
        <rdfs:domain rdf:resource="#Customer"/>
        <rdfs:range rdf:resource="xsd:string"/>
    </owl:DatatypeProperty>
    <owl:DatatypeProperty rdf:about="#securityClassification">
        <rdfs:domain rdf:resource="#DataFile"/>
        <rdfs:range rdf:resource="xsd:string"/>
    </owl:DatatypeProperty>
</rdf:RDF>"""
    owl_path.write_text(owl, encoding="utf-8")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path   = OUTPUT_DIR / f"log_llm_{MODEL_SLUG}_{TIMESTAMP}.txt"
    csv_path   = OUTPUT_DIR / f"metadata_llm_{MODEL_SLUG}_{TIMESTAMP}.csv"
    graph_path = OUTPUT_DIR / f"graph_llm_{MODEL_SLUG}_{TIMESTAMP}.png"
    owl_path   = OUTPUT_DIR / f"ontology_llm_{MODEL_SLUG}_{TIMESTAMP}.owl"

    logger = setup_logging(log_path)
    logger.info("=" * 65)
    logger.info(f"DATA LAKE LLM PIPELINE  —  START  (model: {OLLAMA_MODEL})")
    logger.info(f"Method          : {METHOD}")
    logger.info(f"Swamp directory : {SWAMP_DIR}")
    logger.info(f"OCR available   : {OCR_AVAILABLE}")
    logger.info(f"Speech recog.   : {WHISPER_AVAILABLE} (faster-whisper/base)")
    logger.info(f"LLM timeout     : {TIMEOUT_SEC}s per call")
    logger.info(f"Timestamp       : {TIMESTAMP}")
    logger.info("=" * 65)

    ollama_proc = start_ollama(logger)
    if not ensure_model(logger):
        if ollama_proc:
            ollama_proc.terminate()
        sys.exit(2)
    init_db(DB_PATH)
    write_run_info(DB_PATH, {**run_info(), "timestamp": TIMESTAMP})
    t_start = time.time()

    swamp_files = sorted(f for f in SWAMP_DIR.glob("*") if f.is_file())
    logger.info(f"Files found in data swamp: {len(swamp_files)}")

    metadata = []
    file_idx = 0

    for file_path in swamp_files:
        logger.info(f"\n{'─'*55}")
        logger.info(f"File {file_idx+1}/{len(swamp_files)}: {file_path.name}")
        t_file = time.time()

        content, file_type = extract_text(file_path, logger)
        if file_type == "unsupported":
            logger.info("  Skipped (unsupported format)")
            continue
        if not content.strip():
            logger.warning("  No content extracted — LLM call skipped")
            continue

        logger.info(f"  Calling LLM ({OLLAMA_MODEL}, {len(content)} chars input)...")
        analysis = analyse_with_llm(content, file_path.name, logger)
        logger.info(f"  LLM result: {analysis}")

        info_type   = analysis.get("info_type", "unknown")
        security    = analysis.get("security_class", "general")
        first_name  = analysis.get("first_name", "").strip()
        last_name   = analysis.get("last_name", "").strip()
        llm_cust_id = analysis.get("customer_id", "").strip()
        reason      = analysis.get("reason", "")

        customer_id  = None
        complaint_id = None

        # Register identity entity from LLM extraction, keyed on the first name
        # so that service requests can be linked to this customer by first-name match.
        if info_type == "identities" and first_name:
            full_name = f"{first_name} {last_name}".strip()
            cid = first_name.strip().capitalize()   # match key = first name
            upsert_customer(DB_PATH, cid, full_name)
            customer_id = cid
            logger.info(f"  Customer registered: {cid} ({full_name})")

        # Link service request to known customer
        if info_type == "service_requests" and first_name:
            customer_id = lookup_customer_by_firstname(DB_PATH, first_name)
            if customer_id:
                complaint_id = insert_complaint(DB_PATH, customer_id, content, str(file_path))
                logger.info(f"  Complaint {complaint_id} → customer {customer_id}")
            else:
                logger.info(f"  Speaker '{first_name}' not matched to registered customer")

        elapsed   = round(time.time() - t_file, 3)
        file_id   = f"FILE_{TIMESTAMP}_{file_idx:02d}"
        rec = {
            "file_id":        file_id,
            "file_name":      file_path.name,
            "file_type":      file_type,
            "file_size_bytes": file_path.stat().st_size,
            "info_type":      info_type,
            "security_class": security,
            "customer_id":    customer_id or "",
            "complaint_id":   complaint_id or "",
            "llm_reason":     reason,
            "method":         METHOD,
            "model":          OLLAMA_MODEL,
            "ollama_version": PROVENANCE["ollama_version"],
            "model_digest":   PROVENANCE["model_digest"],
            "processing_sec": elapsed,
            "content_excerpt": content[:120].replace("\n", " ").strip(),
        }
        insert_file_record(DB_PATH, rec)
        metadata.append(rec)
        logger.info(
            f"  RESULT → info_type={info_type}  security={security}  "
            f"customer={customer_id}  time={elapsed}s"
        )
        file_idx += 1

    total_sec = round(time.time() - t_start, 2)

    # ── Outputs ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(metadata)
    df.drop(columns=["file_id", "content_excerpt"], errors="ignore").to_csv(csv_path, index=False)
    logger.info(f"\nMetadata CSV → {csv_path}")

    G = build_graph(DB_PATH)
    save_graph(
        G, graph_path,
        f"Data Lake Entity Relationship Graph — LLM Pipeline ({OLLAMA_MODEL})",
        logger,
    )

    cats = list({r["info_type"] for r in metadata})
    generate_owl(owl_path, cats)
    logger.info(f"OWL ontology   → {owl_path}")

    # ── Summary (for paper results table) ─────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        n_cust = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_comp = conn.execute("SELECT COUNT(*) FROM customer_complaints").fetchone()[0]

    logger.info("\n" + "=" * 65)
    logger.info(f"RESULTS SUMMARY — LLM Pipeline ({OLLAMA_MODEL})")
    logger.info("=" * 65)
    logger.info(f"Files processed         : {len(metadata)}")
    logger.info(f"Total processing time   : {total_sec}s")
    logger.info(f"Avg time per file       : {total_sec / max(len(metadata), 1):.2f}s")
    logger.info("Info-type distribution  :")
    for cat, cnt in df["info_type"].value_counts().items():
        logger.info(f"  {cat:<30}: {cnt} ({100*cnt/len(df):.0f}%)")
    logger.info("Security distribution   :")
    for cls, cnt in df["security_class"].value_counts().items():
        logger.info(f"  {cls:<30}: {cnt} ({100*cnt/len(df):.0f}%)")
    logger.info(f"Customers identified    : {n_cust}")
    logger.info(f"Complaints linked       : {n_comp}")
    logger.info(f"Graph nodes             : {G.number_of_nodes()}")
    logger.info(f"Graph edges             : {G.number_of_edges()}")
    logger.info("=" * 65)
    logger.info("Output files:")
    logger.info(f"  Metadata  : {csv_path}")
    logger.info(f"  Graph PNG : {graph_path}")
    logger.info(f"  Ontology  : {owl_path}")
    logger.info(f"  Log       : {log_path}")
    logger.info("=" * 65)

    if ollama_proc:
        ollama_proc.terminate()

    print(f"\nLLM pipeline ({OLLAMA_MODEL}) complete in {total_sec}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM pipeline over SWAMP_DIR (model etc. via environment variables)")
    ap.add_argument("--allow-pull", action="store_true", help="pull the model tag before running (off by default)")
    ap.add_argument("--require-digest", metavar="SHA256", help="abort unless the model resolves to this weights-blob digest")
    args = ap.parse_args()
    ALLOW_PULL = ALLOW_PULL or args.allow_pull
    REQUIRE_DIGEST = args.require_digest or REQUIRE_DIGEST
    main()
