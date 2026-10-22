"""
Data Lake NLP Pipeline  —  Experiment 1, RO1

Processes all files in data_swamp/ using a statistical NLP method (no external API):
  • OCR (pytesseract) for image text extraction
  • Local Whisper (faster-whisper) for audio transcription
  • TF-IDF vectorisation + cosine similarity against short reference
    descriptions, for BOTH thematic categorisation (identities /
    service_requests / financial_transactions / unknown) AND security
    classification (confidential / general) — see CATEGORY_SEED_TEXTS and
    SECURITY_SEED_TEXTS below. This is a genuinely different mechanism from
    the substring keyword matching used in datalake_rule_based_pipeline.py:
    similarity is a continuous, corpus-weighted score, not a boolean
    keyword-present check, and there is no fixed priority order.
  • Regex entity extraction from identity documents
  • SQLite relational store
  • NetworkX + Matplotlib entity-relationship graph
  • OWL RDF/XML ontology generation

Outputs written to  output/  :
  metadata_nlp_<timestamp>.csv
  graph_nlp_<timestamp>.png
  ontology_nlp_<timestamp>.owl
  log_nlp_<timestamp>.txt

Run:
  python datalake_nlp_pipeline.py
"""

import os
import json
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
METHOD     = "nlp_tfidf"
DB_PATH    = OUTPUT_DIR / f"datalake_nlp_{TIMESTAMP}.db"

# ── Reference descriptions for TF-IDF / cosine-similarity classification ──────
# Each category/security class is represented by one short natural-language
# description of what that class typically contains. A TfidfVectorizer is
# fit on these descriptions together with every file's extracted content, so
# term weights reflect this corpus (rare, distinctive terms score higher than
# common ones). Each file is then classified by cosine similarity to the
# description vector it is closest to. This replaces keyword substring
# matching entirely for the category and security decisions below.
CATEGORY_SEED_TEXTS = {
    "identities": (
        "This document is a government issued identity document such as a driver "
        "licence, driving license, or passport. It shows a person's full legal "
        "name, date of birth, residential address, and a unique licence or "
        "identification number issued by a government authority."
    ),
    "service_requests": (
        "This is a customer service interaction. A customer contacts support to "
        "raise a complaint, dispute, inquiry, or request about a problem or issue, "
        "such as an unexpected transaction on their account, seeking help or "
        "resolution from a service representative."
    ),
    "financial_transactions": (
        "This is a financial transaction record describing a payment, deposit, "
        "withdrawal, transfer, purchase, invoice, or receipt. It includes monetary "
        "amounts, account balances, debit and credit entries, and details of a "
        "purchase or shopping transaction."
    ),
}
CATEGORY_LABELS = list(CATEGORY_SEED_TEXTS.keys())
CATEGORY_SIMILARITY_THRESHOLD = 0.01  # below this, no category is a confident match

SECURITY_SEED_TEXTS = {
    "confidential": (
        "This content is confidential and sensitive. It includes personal "
        "identifying information such as a full name, date of birth, home "
        "address, account number, licence number, phone number, email address, "
        "government identification number, or other private financial or "
        "personal details that must be protected."
    ),
    "general": (
        "This content is general, non-sensitive, routine information. It does "
        "not contain personal identifying details, account numbers, or private "
        "financial data. It is ordinary correspondence, an internal note, or "
        "publicly shareable information."
    ),
}
SECURITY_LABELS = list(SECURITY_SEED_TEXTS.keys())

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("nlp_pipeline")
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
                complaint_type TEXT,
                file_source   TEXT,
                created_at    TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE data_files (
                file_id               TEXT PRIMARY KEY,
                file_name             TEXT,
                file_type             TEXT,
                info_type             TEXT,
                security_class        TEXT,
                category_similarity   REAL,
                security_similarity   REAL,
                customer_id           TEXT REFERENCES customers(customer_id),
                complaint_id          TEXT,
                processing_sec        REAL,
                content_excerpt       TEXT,
                created_at            TEXT DEFAULT (datetime('now'))
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
            "INSERT INTO customer_complaints (complaint_id, customer_id, complaint_text, complaint_type, file_source) VALUES (?,?,?,?,?)",
            (cid, customer_id, text[:500], "complaint", source),
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
                category_similarity, security_similarity,
                customer_id, complaint_id, processing_sec, content_excerpt)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec["file_id"], rec["file_name"], rec["file_type"],
                rec["info_type"], rec["security_class"],
                rec["category_similarity"], rec["security_similarity"],
                rec["customer_id"] or None, rec["complaint_id"] or None,
                rec["processing_sec"], rec["content_excerpt"],
            ),
        )
        conn.commit()

# ── Content extraction ────────────────────────────────────────────────────────
def extract_text(file_path: Path, logger: logging.Logger) -> tuple[str, str]:
    """Return (content, file_type_label). file_type_label is used in metadata."""
    ext = file_path.suffix.lower()

    if ext == ".txt":
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            logger.info(f"  [TEXT] {len(content)} chars read")
            return content, "text"
        except Exception as e:
            logger.error(f"  [TEXT] read error: {e}")
            return "", "text"

    elif ext == ".csv":
        try:
            df = pd.read_csv(file_path)
            content = df.to_string()
            logger.info(f"  [CSV] {len(df)} rows, columns={list(df.columns)}")
            return content, "csv"
        except Exception as e:
            logger.error(f"  [CSV] read error: {e}")
            return "", "csv"

    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        if not OCR_AVAILABLE:
            logger.warning("  [IMAGE] pytesseract not available — skipping OCR")
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
            logger.warning("  [AUDIO] faster-whisper not available — skipping")
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

    else:
        logger.info(f"  [SKIP] unsupported extension: {ext}")
        return "", "unsupported"

# ── NLP analysis: TF-IDF + cosine similarity ──────────────────────────────────
def build_tfidf_space(seed_texts: dict, documents: list[str]) -> tuple[TfidfVectorizer, "scipy.sparse.spmatrix"]:
    """Fit a TF-IDF vectorizer on the seed reference descriptions plus every
    file's extracted content, so term weights reflect this corpus rather than
    a fixed vocabulary. Returns the fitted vectorizer and the seed texts'
    TF-IDF vectors, in the same order as seed_texts.keys()."""
    labels = list(seed_texts.keys())
    corpus = [seed_texts[lbl] for lbl in labels] + list(documents)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    seed_vectors = tfidf_matrix[: len(labels)]
    return vectorizer, seed_vectors

def categorise_tfidf(vectorizer: TfidfVectorizer, seed_vectors, content: str) -> tuple[str, float]:
    """Return (category, top_cosine_similarity) against CATEGORY_SEED_TEXTS.
    No fixed priority order: the category is whichever seed description the
    file's TF-IDF vector is most similar to. Falls back to 'unknown' when the
    best similarity is below CATEGORY_SIMILARITY_THRESHOLD."""
    if not content:
        return "unknown", 0.0
    file_vec = vectorizer.transform([content])
    sims = cosine_similarity(file_vec, seed_vectors)[0]
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    if best_sim < CATEGORY_SIMILARITY_THRESHOLD:
        return "unknown", best_sim
    return CATEGORY_LABELS[best_idx], best_sim

def classify_security_tfidf(vectorizer: TfidfVectorizer, seed_vectors, content: str, info_type: str) -> tuple[str, float]:
    """Return (security_class, confidential_similarity) against SECURITY_SEED_TEXTS.
    Identity documents are always assigned 'confidential' regardless of
    similarity score, the same governance override used by the Rule-Based
    pipeline: an identity document is inherently sensitive by policy, not
    because of its wording."""
    if info_type == "identities":
        return "confidential", 1.0
    if not content:
        return "general", 0.0
    file_vec = vectorizer.transform([content])
    sims = cosine_similarity(file_vec, seed_vectors)[0]  # order: [confidential, general]
    security = SECURITY_LABELS[int(sims.argmax())]
    return security, float(sims[0])

def extract_identity_entity(content: str) -> dict:
    """Regex extraction from driver licence text (as produced by OCR)."""
    info = {"name": None, "id_number": None, "date_of_birth": None, "address": None}
    cu = content.upper()
    if "DRIVER LICENCE" not in cu and "DRIVING LICENCE" not in cu:
        return info

    # License number (MANUFACLIC + digits)
    m = re.search(r"MANUFACLIC(\w+)", content, re.IGNORECASE)
    if m:
        info["id_number"] = f"MANUFACLIC{m.group(1)}"

    # Name: typically on the line after "LICENCE NO."
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
        for cid, name in conn.execute("SELECT customer_id, name FROM customers").fetchall():
            if name:
                label = name
            elif "<" in cid or ">" in cid or "|" in cid or "," in cid or cid.count(" ") > 2 or len(cid) > 24:
                label = "Unresolved ID"
            else:
                label = cid
            G.add_node(f"cust:{cid}", label=label, node_type="Customer")

        for comp_id, cid in conn.execute(
            "SELECT complaint_id, customer_id FROM customer_complaints"
        ).fetchall():
            G.add_node(f"comp:{comp_id}", label=comp_id[:14], node_type="Complaint")
            if cid:
                G.add_edge(f"cust:{cid}", f"comp:{comp_id}", relation="HAS_COMPLAINT")

        for fid, fname, cat, sec, cid in conn.execute(
            "SELECT file_id, file_name, info_type, security_class, customer_id FROM data_files"
        ).fetchall():
            G.add_node(
                f"file:{fid}",
                label=fname[:22],
                node_type=f"DataFile_{cat}",
                security=sec,
            )
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
        mpatches.Patch(color=c, label=lbl.replace("DataFile_", "File — ").replace("_", " "))
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
    log_path   = OUTPUT_DIR / f"log_nlp_{TIMESTAMP}.txt"
    csv_path   = OUTPUT_DIR / f"metadata_nlp_{TIMESTAMP}.csv"
    graph_path = OUTPUT_DIR / f"graph_nlp_{TIMESTAMP}.png"
    owl_path   = OUTPUT_DIR / f"ontology_nlp_{TIMESTAMP}.owl"

    logger = setup_logging(log_path)
    logger.info("=" * 65)
    logger.info("DATA LAKE NLP PIPELINE  —  START")
    logger.info(f"Method          : {METHOD}")
    logger.info(f"Swamp directory : {SWAMP_DIR}")
    logger.info(f"OCR available   : {OCR_AVAILABLE}")
    logger.info(f"Speech recog.   : {WHISPER_AVAILABLE} (faster-whisper/base)")
    logger.info(f"Timestamp       : {TIMESTAMP}")
    logger.info("=" * 65)

    init_db(DB_PATH)
    t_start = time.time()

    swamp_files = sorted(f for f in SWAMP_DIR.glob("*") if f.is_file())
    logger.info(f"Files found in data swamp: {len(swamp_files)}")

    # ── Pass 1: content extraction (OCR / Whisper / parsing) ──────────────────
    logger.info("\nPass 1/2: extracting content from all files...")
    extracted = []  # list of dicts: file_path, content, file_type, extract_sec
    for file_path in swamp_files:
        t_file = time.time()
        content, file_type = extract_text(file_path, logger)
        if file_type == "unsupported":
            logger.info(f"  [SKIP] {file_path.name}")
            continue
        extracted.append({
            "file_path":   file_path,
            "content":     content,
            "file_type":   file_type,
            "extract_sec": time.time() - t_file,
        })

    # ── Fit TF-IDF spaces once on the whole corpus + reference descriptions ──
    all_contents = [e["content"] for e in extracted]
    cat_vectorizer, cat_seed_vecs = build_tfidf_space(CATEGORY_SEED_TEXTS, all_contents)
    sec_vectorizer, sec_seed_vecs = build_tfidf_space(SECURITY_SEED_TEXTS, all_contents)
    logger.info(
        f"\nTF-IDF spaces fit: category vocab={len(cat_vectorizer.vocabulary_)} terms, "
        f"security vocab={len(sec_vectorizer.vocabulary_)} terms "
        f"(over {len(all_contents)} documents + reference descriptions)"
    )

    # ── Pass 2: classification + entity extraction + persistence ─────────────
    logger.info("\nPass 2/2: classifying and linking entities...")
    metadata = []
    file_idx = 0

    for item in extracted:
        file_path = item["file_path"]
        content   = item["content"]
        file_type = item["file_type"]
        logger.info(f"\n{'─'*55}")
        logger.info(f"File {file_idx+1}/{len(extracted)}: {file_path.name}")
        t_classify = time.time()

        info_type, cat_sim = categorise_tfidf(cat_vectorizer, cat_seed_vecs, content)
        security, sec_sim   = classify_security_tfidf(sec_vectorizer, sec_seed_vecs, content, info_type)

        customer_id  = None
        complaint_id = None

        # Identity documents → extract and register entity
        if info_type == "identities" and content:
            entity = extract_identity_entity(content)
            logger.info(f"  Entity extracted: {entity}")
            customer_id = upsert_customer(DB_PATH, entity)
            if customer_id:
                logger.info(f"  Customer registered: {customer_id} ({entity.get('name')})")

        # Service requests / complaints → link to known customer
        if info_type == "service_requests" and content:
            first_name = extract_speaker_firstname(content)
            if first_name:
                customer_id = lookup_customer_by_firstname(DB_PATH, first_name)
                if customer_id:
                    complaint_id = insert_complaint(DB_PATH, customer_id, content, str(file_path))
                    logger.info(f"  Complaint {complaint_id} linked → customer {customer_id}")
                else:
                    logger.info(f"  Speaker '{first_name}' not matched to any customer")

        elapsed = round(item["extract_sec"] + (time.time() - t_classify), 3)
        file_id = f"FILE_{TIMESTAMP}_{file_idx:02d}"
        rec = {
            "file_id":        file_id,
            "file_name":      file_path.name,
            "file_type":      file_type,
            "file_size_bytes": file_path.stat().st_size,
            "info_type":      info_type,
            "security_class": security,
            "category_similarity": round(cat_sim, 4),
            "security_similarity": round(sec_sim, 4),
            "customer_id":    customer_id or "",
            "complaint_id":   complaint_id or "",
            "method":         METHOD,
            "processing_sec": elapsed,
            "content_excerpt": content[:120].replace("\n", " ").strip() if content else "",
        }
        insert_file_record(DB_PATH, rec)
        metadata.append(rec)
        logger.info(
            f"  RESULT → info_type={info_type} (sim={cat_sim:.3f})  "
            f"security={security} (sim={sec_sim:.3f})  "
            f"customer={customer_id}  time={elapsed}s"
        )
        file_idx += 1

    total_sec = round(time.time() - t_start, 2)

    # ── Outputs ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(metadata)
    df.drop(columns=["file_id", "content_excerpt"], errors="ignore").to_csv(csv_path, index=False)
    logger.info(f"\nMetadata CSV → {csv_path}")

    G = build_graph(DB_PATH)
    save_graph(G, graph_path, "Data Lake Entity Relationship Graph — NLP Pipeline", logger)

    cats = list({r["info_type"] for r in metadata})
    generate_owl(owl_path, cats)
    logger.info(f"OWL ontology   → {owl_path}")

    # ── Summary (for paper results table) ─────────────────────────────────────
    with sqlite3.connect(DB_PATH) as conn:
        n_cust  = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        n_comp  = conn.execute("SELECT COUNT(*) FROM customer_complaints").fetchone()[0]
        n_nodes = conn.execute("SELECT COUNT(*) FROM data_files").fetchone()[0]
        n_rel   = conn.execute(
            "SELECT COUNT(*) FROM data_files WHERE customer_id IS NOT NULL"
        ).fetchone()[0]

    logger.info("\n" + "=" * 65)
    logger.info("RESULTS SUMMARY — NLP Pipeline")
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

    print(f"\nNLP pipeline complete in {total_sec}s. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
