"""
Data Lake Rule-Based Pipeline  —  Experiment 1, RO1

Processes all files in data_swamp/ using pure rule-based methods only:
  • No machine learning, no embeddings, no external API calls
  • OCR (pytesseract) for image text extraction
  • Google Speech Recognition for audio transcription (free tier)
  • Keyword priority rules for thematic categorisation
  • Regex patterns for entity extraction from identity documents
  • Deterministic security classification via keyword lists
  • SQLite relational store
  • NetworkX + Matplotlib entity-relationship graph
  • OWL RDF/XML ontology generation

This script provides the simplest and most transparent baseline for comparison
against the NLP (keyword+regex) and LLM pipelines. It does NOT use any:
  - Machine learning models
  - Word embeddings
  - External API calls
  - Probabilistic scoring

Outputs written to  output/  :
  metadata_rule_based_<timestamp>.csv
  graph_rule_based_<timestamp>.png
  ontology_rule_based_<timestamp>.owl
  log_rule_based_<timestamp>.txt

Run:
  python datalake_rule_based_pipeline.py
"""

import json
import os
import sys
import sqlite3
import re
import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
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

# ── Paths & constants ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
SWAMP_DIR  = BASE_DIR / os.environ.get("SWAMP_DIR", "data_swamp")
OUTPUT_DIR = BASE_DIR / "output"
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
METHOD     = "rule_based"
DB_PATH    = OUTPUT_DIR / f"datalake_rule_based_{TIMESTAMP}.db"

# ── Classification rules ──────────────────────────────────────────────────────
# Priority order: identities > service_requests > financial_transactions > unknown
# Rules are evaluated in this order; first match wins.

RULES = {
    "identities": {
        "keywords": [
            "driver licence", "driving licence", "driver license", "driving license",
            "licence no", "license no", "manufaclic", "passport", "identity card",
            "dob", "date of birth", "date_of_birth",
        ],
        "security": "confidential",   # identity documents are always confidential
    },
    "service_requests": {
        "keywords": [
            "complaint", "issue", "problem", "dispute", "grievance",
            "request", "inquiry", "enquiry", "support", "service call",
            "unexpected transaction", "not my transaction",
        ],
        "security": None,   # determined separately
    },
    "financial_transactions": {
        "keywords": [
            "transaction", "payment", "deposit", "withdrawal", "transfer",
            "receipt", "invoice", "purchase", "shopping", "spent", "debit",
            "credit", "amount", "balance",
        ],
        "security": None,
    },
    "unknown": {
        "keywords": [],     # fallback
        "security": None,
    },
}

# Keywords that flag content as confidential regardless of category
CONFIDENTIAL_INDICATORS = [
    "account number", "account no", "manufaclic",
    "license number", "licence number", "id number",
    "date of birth", "dob", "address", "phone number", "mobile number",
    "ssn", "social security", "credit card", "bank account",
    "tax file number", "tfn", "medicare", "passport number",
    "personal", "private", "confidential", "driver licence", "driving licence",
]

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("rule_pipeline")
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

# ── Rule engine ───────────────────────────────────────────────────────────────
def apply_category_rules(content: str) -> str:
    """Return the first matching category (priority order preserved)."""
    if not content:
        return "unknown"
    cl = content.lower()
    for cat in ("identities", "service_requests", "financial_transactions"):
        if any(kw in cl for kw in RULES[cat]["keywords"]):
            return cat
    return "unknown"

def apply_security_rules(content: str, info_type: str) -> str:
    """Return 'confidential' or 'general' based on deterministic rules."""
    # Identity documents are always confidential
    if info_type == "identities":
        return "confidential"
    if not content:
        return "general"
    cl = content.lower()
    return "confidential" if any(kw in cl for kw in CONFIDENTIAL_INDICATORS) else "general"

def count_keyword_hits(content: str, category: str) -> int:
    """Count how many rule keywords matched (for reporting transparency)."""
    if not content or category not in RULES:
        return 0
    cl = content.lower()
    return sum(1 for kw in RULES[category]["keywords"] if kw in cl)

# ── Regex entity extraction from identity documents ───────────────────────────
def extract_identity_entity(content: str) -> dict:
    info = {"name": None, "id_number": None, "date_of_birth": None, "address": None}
    cu = content.upper()
    if "DRIVER LICENCE" not in cu and "DRIVING LICENCE" not in cu:
        return info

    # Licence number
    m = re.search(r"MANUFACLIC(\w+)", content, re.IGNORECASE)
    if m:
        info["id_number"] = f"MANUFACLIC{m.group(1)}"

    # Name from line after LICENCE NO.
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    for i, line in enumerate(lines):
        if "LICENCE NO" in line.upper() and i + 1 < len(lines):
            nm = re.search(r"([A-Z][a-z]+)\s+([A-Z][a-z]+)", lines[i + 1])
            if nm:
                info["name"] = f"{nm.group(1)} {nm.group(2)}"
            break

    # Date of birth
    m = re.search(r"DOB[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})", content, re.IGNORECASE)
    if m:
        info["date_of_birth"] = m.group(1)
    else:
        dates = re.findall(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b", content)
        if dates:
            info["date_of_birth"] = dates[0]

    # Address
    m = re.search(
        r"(\d+\s+[A-Za-z\s]+(st|street|rd|road|ave|avenue)[^\n]+\d{4})",
        content, re.IGNORECASE,
    )
    if m:
        info["address"] = m.group(1).strip()

    return info

def extract_speaker_firstname(content: str) -> str | None:
    m = re.search(r"my\s+name\s+is\s+([A-Za-z]+)", content, re.IGNORECASE)
    return m.group(1).strip() if m else None

# ── Content extraction ────────────────────────────────────────────────────────
def extract_text(file_path: Path, logger: logging.Logger) -> tuple[str, str]:
    ext = file_path.suffix.lower()

    if ext == ".txt":
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            logger.info(f"  [TEXT] {len(content)} chars")
            return content, "text"
        except Exception as e:
            logger.error(f"  [TEXT] {e}")
            return "", "text"

    elif ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            content = df.to_string()
            logger.info(f"  [CSV] {len(df)} rows, cols={list(df.columns)}")
            return content, "csv"
        except Exception as e:
            logger.error(f"  [CSV] {e}")
            return "", "csv"

    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        if not OCR_AVAILABLE:
            logger.warning("  [IMAGE] pytesseract unavailable")
            return "", "image"
        try:
            with Image.open(file_path) as img:
                content = pytesseract.image_to_string(img)
            logger.info(f"  [IMAGE/OCR] {len(content)} chars")
            return content, "image"
        except Exception as e:
            logger.error(f"  [IMAGE] {e}")
            return "", "image"

    elif ext in {".wav", ".mp3", ".m4a", ".flac"}:
        if not WHISPER_AVAILABLE:
            logger.warning("  [AUDIO] faster-whisper unavailable")
            return "", "audio"
        try:
            segments, _ = _WHISPER_MODEL.transcribe(str(file_path))
            content = " ".join(seg.text for seg in segments).strip()
            logger.info(f"  [AUDIO/Whisper] '{content[:80]}{'...' if len(content) > 80 else ''}'")
            return content, "audio"
        except Exception as e:
            logger.warning(f"  [AUDIO] {e}")
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
                date_of_birth TEXT,
                address       TEXT,
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
                file_id             TEXT PRIMARY KEY,
                file_name           TEXT,
                file_type           TEXT,
                info_type           TEXT,
                security_class      TEXT,
                keyword_hits        INTEGER,
                customer_id         TEXT REFERENCES customers(customer_id),
                complaint_id        TEXT,
                processing_sec      REAL,
                content_excerpt     TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

def upsert_customer(db_path: Path, info: dict) -> str | None:
    if not info.get("name"):
        return None
    # Match key is the customer's first name so records can be linked by first name.
    first_name = info["name"].split()[0].capitalize()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO customers (customer_id, name, date_of_birth, address) VALUES (?,?,?,?)",
            (first_name, info["name"], info.get("date_of_birth"), info.get("address")),
        )
        conn.commit()
    return first_name

def insert_complaint(db_path: Path, customer_id: str, text: str, source: str) -> str:
    cid = f"COMP{datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO customer_complaints (complaint_id, customer_id, complaint_text, file_source) VALUES (?,?,?,?)",
            (cid, customer_id, text[:500], source),
        )
        conn.commit()
    return cid

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
                keyword_hits, customer_id, complaint_id, processing_sec, content_excerpt)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                rec["file_id"], rec["file_name"], rec["file_type"],
                rec["info_type"], rec["security_class"], rec["keyword_hits"],
                rec["customer_id"] or None, rec["complaint_id"] or None,
                rec["processing_sec"], rec["content_excerpt"],
            ),
        )
        conn.commit()

# ── Knowledge graph ───────────────────────────────────────────────────────────
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

def save_graph(G: nx.DiGraph, graph_path: Path, logger: logging.Logger) -> None:
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
        mpatches.Patch(color=c, label=lbl.replace("DataFile_", "File — ").replace("_", " "))
        for lbl, c in NODE_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("Data Lake Entity Relationship Graph — Rule-Based Pipeline", fontsize=13, pad=20)
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
    log_path   = OUTPUT_DIR / f"log_rule_based_{TIMESTAMP}.txt"
    csv_path   = OUTPUT_DIR / f"metadata_rule_based_{TIMESTAMP}.csv"
    graph_path = OUTPUT_DIR / f"graph_rule_based_{TIMESTAMP}.png"
    owl_path   = OUTPUT_DIR / f"ontology_rule_based_{TIMESTAMP}.owl"

    logger = setup_logging(log_path)
    logger.info("=" * 65)
    logger.info("DATA LAKE RULE-BASED PIPELINE  —  START")
    logger.info(f"Method          : {METHOD}")
    logger.info(f"Swamp directory : {SWAMP_DIR}")
    logger.info(f"OCR available   : {OCR_AVAILABLE}")
    logger.info(f"Speech recog.   : {WHISPER_AVAILABLE} (faster-whisper/base)")
    logger.info(f"Timestamp       : {TIMESTAMP}")
    logger.info("Category rules  :")
    for cat, cfg in RULES.items():
        logger.info(f"  {cat:<28}: {len(cfg['keywords'])} keywords")
    logger.info(f"Confidential kw : {len(CONFIDENTIAL_INDICATORS)} indicators")
    logger.info("=" * 65)

    init_db(DB_PATH)
    t_start = time.time()

    swamp_files = sorted(f for f in SWAMP_DIR.glob("*") if f.is_file())
    logger.info(f"Files found in data swamp: {len(swamp_files)}")

    metadata  = []
    file_idx  = 0

    for file_path in swamp_files:
        logger.info(f"\n{'─'*55}")
        logger.info(f"File {file_idx+1}/{len(swamp_files)}: {file_path.name}")
        t_file = time.time()

        content, file_type = extract_text(file_path, logger)
        if file_type == "unsupported":
            logger.info("  Skipped (unsupported format)")
            continue

        # Apply deterministic rules
        info_type    = apply_category_rules(content)
        security     = apply_security_rules(content, info_type)
        kw_hits      = count_keyword_hits(content, info_type)

        customer_id  = None
        complaint_id = None

        # Identity documents → regex entity extraction
        if info_type == "identities" and content:
            entity = extract_identity_entity(content)
            logger.info(f"  Regex extraction: {entity}")
            customer_id = upsert_customer(DB_PATH, entity)
            if customer_id:
                logger.info(f"  Customer registered: {customer_id} ({entity.get('name')})")

        # Service requests → link complaint to known customer
        if info_type == "service_requests" and content:
            first_name = extract_speaker_firstname(content)
            if first_name:
                customer_id = lookup_customer_by_firstname(DB_PATH, first_name)
                if customer_id:
                    complaint_id = insert_complaint(DB_PATH, customer_id, content, str(file_path))
                    logger.info(f"  Complaint {complaint_id} → customer {customer_id}")
                else:
                    logger.info(f"  Speaker '{first_name}' not matched to a registered customer")

        elapsed = round(time.time() - t_file, 3)
        file_id = f"FILE_{TIMESTAMP}_{file_idx:02d}"
        rec = {
            "file_id":        file_id,
            "file_name":      file_path.name,
            "file_type":      file_type,
            "file_size_bytes": file_path.stat().st_size,
            "info_type":      info_type,
            "security_class": security,
            "keyword_hits":   kw_hits,
            "customer_id":    customer_id or "",
            "complaint_id":   complaint_id or "",
            "method":         METHOD,
            "processing_sec": elapsed,
            "content_excerpt": content[:120].replace("\n", " ").strip() if content else "",
        }
        insert_file_record(DB_PATH, rec)
        metadata.append(rec)
        logger.info(
            f"  RESULT → info_type={info_type}  security={security}  "
            f"kw_hits={kw_hits}  customer={customer_id}  time={elapsed}s"
        )
        file_idx += 1

    total_sec = round(time.time() - t_start, 2)

    # ── Outputs ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(metadata)
    df.drop(columns=["file_id", "content_excerpt"], errors="ignore").to_csv(csv_path, index=False)
    logger.info(f"\nMetadata CSV → {csv_path}")

    G = build_graph(DB_PATH)
    save_graph(G, graph_path, logger)

    cats = list({r["info_type"] for r in metadata})
    generate_owl(owl_path, cats)
    logger.info(f"OWL ontology   → {owl_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        n_cust = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_comp = conn.execute("SELECT COUNT(*) FROM customer_complaints").fetchone()[0]

    logger.info("\n" + "=" * 65)
    logger.info("RESULTS SUMMARY — Rule-Based Pipeline")
    logger.info("=" * 65)
    logger.info(f"Files processed         : {len(metadata)}")
    logger.info(f"Total processing time   : {total_sec}s")
    logger.info(f"Avg time per file       : {total_sec / max(len(metadata), 1):.3f}s")
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

    print(f"\nRule-based pipeline complete in {total_sec}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
