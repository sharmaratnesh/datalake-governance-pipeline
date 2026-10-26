"""Populate the generated OWL ontology with individuals from the pipeline's SQLite
store, then run SPARQL competency queries against it. Demonstrates that the
generated artefacts support governance queries (Table 7: queryable knowledge graph).

The DB/OWL pair is selected explicitly (datalake_artefacts.py): it is auto-selected only
if exactly one Phi-3 Mini run over the 55-file data_swamp/ exists; otherwise pass
--run-ts <timestamp>, or --db and --tbox. The DB and OWL must come from the same run.

  python datalake_query_knowledge_graph.py --run-ts 20261022_143804 \\
      --results-csv evaluation/output/competency_queries_<ts>.csv
"""
import argparse
import csv
import html
import re
import sqlite3
from pathlib import Path

from rdflib import Graph

from datalake_artefacts import SelectionError, select_artefact

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"


def primary_corpus_db(path: Path) -> str | None:
    """accept() for DBs: every record must be a data_swamp/ file, and all 55 present."""
    swamp = {p.name for p in (BASE_DIR / "data_swamp").glob("*") if p.is_file()}
    with sqlite3.connect(path) as c:
        files = {r[0] for r in c.execute("SELECT file_name FROM data_files")}
    if not swamp:
        return None  # corpus not fetched; cannot check
    if files != swamp:
        return f"{len(files & swamp)} of {len(files)} records are data_swamp files ({len(swamp)} expected)"
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--run-ts", help="timestamp of the Phi-3 Mini run, e.g. 20261022_143804")
ap.add_argument("--db", default=None, help="path to a pipeline SQLite DB")
ap.add_argument("--tbox", default=None, help="path to the matching OWL ontology")
ap.add_argument("--out", default=None, help="path to write the populated ontology (default: output/ontology_populated_<run>.owl)")
ap.add_argument("--results-csv", default=None, help="also write the query results to this CSV")
args = ap.parse_args()

if args.run_ts:
    args.db = args.db or str(OUTPUT_DIR / f"datalake_llm_phi3_mini_{args.run_ts}.db")
    args.tbox = args.tbox or str(OUTPUT_DIR / f"ontology_llm_phi3_mini_{args.run_ts}.owl")
DB = select_artefact(OUTPUT_DIR, "datalake_llm_phi3_mini_*.db", explicit=args.db,
                     accept=primary_corpus_db, what="Phi-3 Mini DB", option="--run-ts")
if DB is None:
    raise SelectionError(f"No Phi-3 Mini data_swamp DB in {OUTPUT_DIR} - run datalake_llm_pipeline.py first.")
run = re.search(r"(\d{8}_\d{6})", DB.name).group(1)
TBOX = Path(args.tbox) if args.tbox else OUTPUT_DIR / f"ontology_llm_phi3_mini_{run}.owl"
if not TBOX.exists():
    raise SelectionError(f"{TBOX} does not exist")
if run not in TBOX.name:
    raise SelectionError(f"DB ({DB.name}) and ontology ({TBOX.name}) come from different runs")
OUT = args.out or str(OUTPUT_DIR / f"ontology_populated_{run}.owl")
print(f"DB   : {DB.name}\nTBox : {TBOX.name}")

CLASS_MAP = {"identities": "Identities", "service_requests": "ServiceRequests",
             "financial_transactions": "FinancialTransactions", "unknown": "DataFile"}

def esc(s):
    return html.escape(str(s), quote=True) if s is not None else ""

def safe_id(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))

c = sqlite3.connect(DB)
files = c.execute("SELECT file_id,file_name,file_type,info_type,security_class,customer_id FROM data_files").fetchall()
custs = c.execute("SELECT customer_id,name FROM customers").fetchall()

tbox = open(TBOX, encoding="utf-8").read().replace("\r\n", "\n")
body = tbox.rsplit("</rdf:RDF>", 1)[0]

parts = [body, "\n    <!-- ABox: individuals asserted from the pipeline's entity store -->\n"]
for cid, nm in custs:
    parts.append(f'    <owl:NamedIndividual rdf:about="#customer_{safe_id(cid)}">'
                 f'<rdf:type rdf:resource="#Customer"/>'
                 f'<customerName>{esc(nm)}</customerName></owl:NamedIndividual>\n')
for fid, fname, ftype, info, sec, cid in files:
    cls = CLASS_MAP.get((info or "").strip(), "DataFile")
    s = [f'    <owl:NamedIndividual rdf:about="#file_{safe_id(fid)}">',
         f'<rdf:type rdf:resource="#{cls}"/>',
         f'<rdfs:label>{esc(fname)}</rdfs:label>',
         f'<securityClassification>{esc(sec)}</securityClassification>']
    if cid and str(cid).strip().lower() not in ("", "none", "unknown"):
        s.append(f'<relatedToCustomer rdf:resource="#customer_{safe_id(cid)}"/>')
    s.append('</owl:NamedIndividual>\n')
    parts.append("".join(s))
parts.append("</rdf:RDF>\n")
open(OUT, "w", encoding="utf-8").write("".join(parts))

g = Graph(); g.parse(OUT, format="xml")
print(f"populated ontology: {len(g)} triples\n")

QUERIES = {
 "CQ1: Which files are classified confidential?":
   """SELECT (COUNT(?f) AS ?n) WHERE { ?f <http://www.datalake.org/ontology#securityClassification> "confidential" }""",
 "CQ2: Which identity documents are confidential (policy check)?":
   """SELECT (COUNT(?f) AS ?n) WHERE { ?f a <http://www.datalake.org/ontology#Identities> ;
        <http://www.datalake.org/ontology#securityClassification> "confidential" }""",
 "CQ3: Which files are linked to a known customer?":
   """SELECT (COUNT(?f) AS ?n) WHERE { ?f <http://www.datalake.org/ontology#relatedToCustomer> ?c }""",
 "CQ4: Which confidential files have NO customer link (orphaned sensitive data)?":
   """SELECT (COUNT(?f) AS ?n) WHERE { ?f <http://www.datalake.org/ontology#securityClassification> "confidential" .
        FILTER NOT EXISTS { ?f <http://www.datalake.org/ontology#relatedToCustomer> ?c } }""",
 "CQ5: Count of files per information-type subclass":
   """SELECT ?cls (COUNT(?f) AS ?n) WHERE { ?f a ?cls .
        ?cls <http://www.w3.org/2000/01/rdf-schema#subClassOf> <http://www.datalake.org/ontology#DataFile> }
        GROUP BY ?cls ORDER BY DESC(?n)""",
}
results = [{"run": run, "query": "triples", "key": "", "value": len(g)}]
for label, q in QUERIES.items():
    print(label)
    for row in g.query(q):
        vals = [str(v).split("#")[-1] for v in row]
        print("   ", " | ".join(vals))
        results.append({"run": run, "query": label.split(":")[0], "key": vals[0] if len(vals) > 1 else "",
                        "value": vals[-1]})
    print()

if args.results_csv:
    with open(args.results_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "query", "key", "value"])
        w.writeheader(); w.writerows(results)
    print(f"-> {args.results_csv}")
