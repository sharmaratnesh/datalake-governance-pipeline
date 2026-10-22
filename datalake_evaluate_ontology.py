# -*- coding: utf-8 -*-
"""Ontology and entity-graph evaluation (Reviewer 1, point 5).

Evaluates, for each pipeline, three things the competency queries did not:
  (1) structural validity of the emitted ontology  - does it parse, what does it declare
  (2) entity extraction quality                    - customers recovered vs golden
  (3) relation extraction quality                  - file -> customer links vs golden

Artefacts are selected EXPLICITLY. There is no "latest file" default: output/ holds
runs over more than one corpus (55-file primary swamp, TAB), and picking the newest
file silently scored TAB ontologies against the 55-file golden table.

Select either with a run tag that matches exactly one DB and one OWL per method
    python datalake_evaluate_ontology.py --label primary --run-tag 20261021_22
or per method (any subset of methods; omitted methods are skipped)
    python datalake_evaluate_ontology.py --label primary \
        --rule-based-db output/datalake_rule_based_20261021_222646.db \
        --rule-based-owl output/ontology_rule_based_20261021_222646.owl  ...
If a selection is missing or ambiguous the script exits non-zero and lists candidates.

Entity/relation scoring is refused (exit 2) when a DB's file set is not drawn from the
golden table's file set. For corpora with no customer golden (TAB) pass
--structural-only: OWL validity is reported and DBs are only counted.

Output: evaluation/output/ontology_eval_<timestamp>_<label>.csv, one row per method.
Run from the repo root (or pass --repo).
"""
import argparse, glob, os, re, sqlite3, sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from rdflib import Graph, RDF, OWL, RDFS

OWL_NS = "http://www.datalake.org/ontology#"

METHODS = {  # key -> (display name, DB glob, OWL glob)
    "rule-based": ("Rule-Based",   "datalake_rule_based_*.db",       "ontology_rule_based_*.owl"),
    "nlp":        ("NLP",          "datalake_nlp_*.db",              "ontology_nlp_*.owl"),
    "phi3":       ("Phi-3 Mini",   "datalake_llm_phi3_mini_*.db",    "ontology_llm_phi3_mini_*.owl"),
    "llama":      ("LLaMA 3.2 1B", "datalake_llm_llama32_1b_*.db",   "ontology_llm_llama32_1b_*.owl"),
}

def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def norm_name(s):
    """Loose person-name normalisation: lowercase, strip titles/suffixes/punctuation."""
    if s is None: return ""
    s = str(s).strip().lower()
    if s in ("", "nan", "none", "unknown", "n/a", "multiple"): return ""
    s = re.sub(r"\b(mr|mrs|ms|miss|dr|prof)\.?\b", " ", s)
    s = re.sub(r"\b(dds|md|phd|jr|sr|iii|ii)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def first_name(s):
    n = norm_name(s)
    return n.split(" ")[0] if n else ""

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f = 2*p*r/(p+r) if p+r else 0.0
    return round(p,3), round(r,3), round(f,3)

def structural(owl_path):
    out = {"parses": False, "triples": 0, "classes": 0, "object_properties": 0,
           "datatype_properties": 0, "individuals": 0, "subclasses_of_DataFile": 0,
           "malformed_iri": 0, "parse_error": "", "offending_iri": ""}
    # IRIs containing characters that signal a label leaked straight into the IRI.
    # Read from the raw text so they are reported even when the file does not parse.
    raw = Path(owl_path).read_text(encoding="utf-8", errors="replace")
    bad = [m.group(1) for m in re.finditer(r'rdf:(?:about|resource)="([^"]*)"', raw)
           if re.search(r"[<>|\s]", m.group(1))]
    out["malformed_iri"] = len(bad)
    out["offending_iri"] = " ; ".join(bad)
    try:
        g = Graph(); g.parse(owl_path, format="xml")
    except Exception as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
        return out
    out["parses"] = True
    out["triples"] = len(g)
    out["classes"] = len(set(g.subjects(RDF.type, OWL.Class)))
    out["object_properties"] = len(set(g.subjects(RDF.type, OWL.ObjectProperty)))
    out["datatype_properties"] = len(set(g.subjects(RDF.type, OWL.DatatypeProperty)))
    out["individuals"] = len(set(g.subjects(RDF.type, OWL.NamedIndividual)))
    out["subclasses_of_DataFile"] = len(list(g.subjects(RDFS.subClassOf, None)))
    return out

def resolve(outdir, pattern, explicit, run_tag, what, method):
    """Return exactly one path, or exit listing the candidates. Never picks 'latest'."""
    candidates = sorted(glob.glob(str(outdir/pattern)))
    if explicit:
        p = Path(explicit)
        if not p.exists(): die(f"{method} {what}: {p} does not exist")
        return p
    if run_tag is None:
        return None
    hits = [c for c in candidates if run_tag in os.path.basename(c)]
    if len(hits) != 1:
        listing = "\n  ".join(os.path.basename(c) for c in candidates) or "(none)"
        die(f"{method} {what}: run tag '{run_tag}' matched {len(hits)} files "
            f"(need exactly 1). Candidates:\n  {listing}\n"
            f"Use a more specific --run-tag or pass --{method}-{what} explicitly.")
    return Path(hits[0])

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="evaluation/output")
    ap.add_argument("--label", required=True, help="suffix for the output file, e.g. primary or tab")
    ap.add_argument("--run-tag", help="substring that must match exactly one DB and one OWL per method")
    ap.add_argument("--structural-only", action="store_true",
                    help="score OWL validity only (for corpora without a customer golden, e.g. TAB)")
    ap.add_argument("--golden", default="evaluation/evaluation_template_golden.xlsx")
    for key in METHODS:
        ap.add_argument(f"--{key}-db"); ap.add_argument(f"--{key}-owl")
    a = ap.parse_args()
    repo = Path(a.repo); outdir_art = repo/"output"

    selected = {}
    for key, (name, db_glob, owl_glob) in METHODS.items():
        db  = resolve(outdir_art, db_glob,  getattr(a, f"{key.replace('-','_')}_db"),  a.run_tag, "db",  key)
        owl = resolve(outdir_art, owl_glob, getattr(a, f"{key.replace('-','_')}_owl"), a.run_tag, "owl", key)
        if db or owl: selected[key] = (name, db, owl)
    if not selected:
        cands = "\n  ".join(os.path.basename(c) for k in METHODS for g in METHODS[k][1:]
                            for c in sorted(glob.glob(str(outdir_art/g))))
        die("no artefacts selected. Pass --run-tag or per-method --<method>-db/--<method>-owl. "
            f"Candidates in {outdir_art}:\n  {cands}")

    if not a.structural_only:
        golden = pd.read_excel(repo/a.golden, sheet_name="evaluation", header=1).dropna(subset=["File Name"])
        gold_link = {str(r["File Name"]).strip(): str(r["Linked To Customer"]) for _, r in golden.iterrows()}
        gold_customers = {norm_name(v) for v in gold_link.values() if norm_name(v)}
        gold_first = {first_name(c) for c in gold_customers}

    rows = []
    for key, (method, db, owl) in selected.items():
        row = {"method": method, "owl_file": owl.name if owl else "", "db_file": db.name if db else ""}
        if owl: row.update(structural(owl))
        if db is None:
            rows.append(row); continue
        con = sqlite3.connect(db)
        db_files = {str(f).strip() for (f,) in con.execute("SELECT file_name FROM data_files")}
        row["db_records"] = len(db_files)
        if a.structural_only:
            con.close(); rows.append(row); continue

        # ---- corpus guard: the DB must be scored against its own golden ----
        overlap = db_files & set(gold_link)
        foreign = db_files - set(gold_link)
        if not overlap or foreign:
            die(f"{method}: {db.name} is not a run over the golden corpus "
                f"({len(overlap)} of {len(db_files)} files in golden, {len(foreign)} foreign, "
                f"e.g. {sorted(foreign)[:3]}). Refusing to score entity/relation metrics. "
                f"Use --structural-only for corpora without a customer golden.")
        missing = set(gold_link) - db_files
        if missing:
            print(f"WARNING: {method}: {len(missing)} golden files absent from {db.name}", file=sys.stderr)

        # ---- entity level: customers recovered -------------------------------
        pred_cust = {norm_name(n) for (n,) in con.execute("SELECT name FROM customers") if norm_name(n)}
        # a predicted customer counts as correct if its first name matches a golden customer's
        tp = len({c for c in pred_cust if first_name(c) in gold_first})
        fp = len(pred_cust) - tp
        fn = len(gold_first - {first_name(c) for c in pred_cust})
        p, r, f = prf(tp, fp, fn)
        row.update({"predicted_customers": len(pred_cust), "golden_customers": len(gold_first),
                    "entity_tp": tp, "entity_fp": fp, "entity_fn": fn,
                    "entity_precision": p, "entity_recall": r, "entity_f1": f})

        # ---- relation level: file -> customer links --------------------------
        rtp = rfp = rfn = 0
        for fn_, cid in con.execute("SELECT file_name, customer_id FROM data_files"):
            pred = cid is not None and str(cid).strip().lower() not in ("", "none", "unknown", "nan")
            gold = bool(norm_name(gold_link.get(str(fn_).strip(), "")))
            if pred and gold:
                # correct only if it points at the right person
                hit = first_name(str(cid)) == first_name(gold_link[str(fn_).strip()])
                rtp += 1 if hit else 0
                rfp += 0 if hit else 1
            elif pred and not gold: rfp += 1
            elif gold and not pred: rfn += 1
        p, r, f = prf(rtp, rfp, rfn)
        row.update({"relation_tp": rtp, "relation_fp": rfp, "relation_fn": rfn,
                    "relation_precision": p, "relation_recall": r, "relation_f1": f,
                    "linkable_files": sum(1 for v in gold_link.values() if norm_name(v))})
        con.close()
        rows.append(row)

    outdir = repo/a.out; outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir/f"ontology_eval_{datetime.now():%Y%m%d_%H%M%S}_{a.label}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print("=== SELECTED ARTEFACTS ===")
    print(df[["method", "db_file", "owl_file"]].to_string(index=False))
    print("\n=== STRUCTURAL VALIDITY ===")
    cols = [c for c in ["method","parses","triples","classes","object_properties","datatype_properties",
                        "individuals","malformed_iri","parse_error","offending_iri"] if c in df]
    print(df[cols].to_string(index=False))
    if not a.structural_only:
        print("\n=== ENTITY EXTRACTION (customers) ===")
        print(df[["method","predicted_customers","golden_customers","entity_tp","entity_fp","entity_fn",
                  "entity_precision","entity_recall","entity_f1"]].to_string(index=False))
        print("\n=== RELATION EXTRACTION (file -> customer) ===")
        print(df[["method","relation_tp","relation_fp","relation_fn","relation_precision",
                  "relation_recall","relation_f1","linkable_files"]].to_string(index=False))
    print(f"\n-> {out_csv}")

if __name__ == "__main__":
    main()
