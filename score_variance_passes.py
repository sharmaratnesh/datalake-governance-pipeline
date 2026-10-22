"""
Score the run-to-run variance passes produced by run_variance_passes.py.

Each pass is scored by datalake_evaluate_all.py, pinned to that pass's metadata CSV and
written to its own directory, so no pass can be confused with another run. Results are
collected into

  evaluation/output/variance_primary_<ts>.csv           one row per (model, pass)
  evaluation/output/variance_primary_<ts>_pairwise.csv  label/input differences between passes

The July (paper) and October reference runs are appended as rows with pass=ref_<ts>,
read from their original evaluation outputs; they had no seed and their blob digest was
not recorded at run time.

Run:
  python score_variance_passes.py evaluation/output/variance_runs_<ts>.csv [more manifests ...]       --default-ollama-version 0.40.7
(--default-ollama-version fills manifests written before the runner recorded the version.)
"""

import argparse
import itertools
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
EVAL_OUT   = BASE_DIR / "evaluation" / "output"
RESULTS    = BASE_DIR / "evaluation" / "results"   # published copies, in a fresh clone

MODELS = {  # ollama name -> (evaluate_all key, display name, file slug)
    "phi3:mini":   ("phi3",  "LLM-phi3-mini",   "phi3_mini"),
    "llama3.2:1b": ("llama", "LLM-llama3.2-1b", "llama32_1b"),
}
JULY_VERSION = "unrecorded (<=0.32.3, the newest release at run time)"
REFERENCES = [  # (model, run_ts, evaluation_ts, ollama version) - the paper's runs and the October rerun
    ("phi3:mini",   "20260723_135524", "20260723_155341", JULY_VERSION),
    ("phi3:mini",   "20261021_223725", "20261021_225854", "0.40.7"),
    ("llama3.2:1b", "20260723_134633", "20260723_155341", JULY_VERSION),
    ("llama3.2:1b", "20261021_222833", "20261021_225854", "0.40.7"),
]
COLUMNS = ["model", "pass", "temperature", "seed", "ollama_version", "blob_digest", "info_type_accuracy",
           "sensitivity_accuracy", "sensitivity_kappa", "general_precision", "general_recall",
           "general_f1", "n_predicted_general", "linkage_accuracy", "overall_accuracy",
           "wallclock_sec", "run_ts", "n_predicted_other_security_label"]


def scores_from(eval_dir: Path, eval_ts: str, display: str) -> dict:
    summ = pd.read_csv(eval_dir / f"evaluation_summary_{eval_ts}.csv").set_index("method").loc[display]
    kap  = pd.read_csv(eval_dir / f"evaluation_kappa_{eval_ts}.csv")
    kap  = kap[(kap.method_a == "Golden") & (kap.method_b == display)].iloc[0]
    prf  = pd.read_csv(eval_dir / f"evaluation_security_prf_{eval_ts}.csv")
    gen  = prf[(prf.method == display) & (prf["class"] == "general")].iloc[0]
    return {"info_type_accuracy": summ.category_accuracy, "sensitivity_accuracy": summ.security_accuracy,
            "sensitivity_kappa": kap.kappa_security, "general_precision": gen.precision,
            "general_recall": gen.recall, "general_f1": gen.f1,
            "linkage_accuracy": summ.linkage_accuracy, "overall_accuracy": summ.overall_accuracy}


def predictions(slug: str, run_ts: str) -> pd.DataFrame:
    """Per-file labels, read from the run's DB (the July LLaMA CSV no longer exists)."""
    with sqlite3.connect(OUTPUT_DIR / f"datalake_llm_{slug}_{run_ts}.db") as con:
        return pd.read_sql("SELECT file_name, info_type, security_class FROM data_files", con).set_index("file_name")


def llm_inputs(slug: str, run_ts: str) -> dict:
    """file -> number of characters sent to the LLM (from the run log)."""
    d, cur = {}, None
    for line in (OUTPUT_DIR / f"log_llm_{slug}_{run_ts}.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"File \d+/\d+: (.+)$", line)
        if m: cur = m.group(1).strip()
        m = re.search(r"Calling LLM \(.*?, (\d+) chars input\)", line)
        if m and cur: d[cur] = int(m.group(1))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", type=Path, nargs="+")
    ap.add_argument("--default-ollama-version", help="for manifests without an ollama_version column")
    args = ap.parse_args()
    frames = []
    for m in args.manifests:
        f = pd.read_csv(m, dtype={"run_ts": str, "ollama_version": str})
        if "ollama_version" not in f:
            if not args.default_ollama_version:
                sys.exit(f"{m.name} has no ollama_version column: pass --default-ollama-version")
            f["ollama_version"] = args.default_ollama_version
        frames.append(f)
    runs = pd.concat(frames, ignore_index=True).rename(columns={"pass": "pass_id"})  # "pass" is a keyword
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = EVAL_OUT / f"variance_scoring_{ts}"

    rows, labelled = [], []
    for r in runs.itertuples():
        key, display, slug = MODELS[r.model]
        csv = OUTPUT_DIR / f"metadata_llm_{slug}_{r.run_ts}.csv"
        if r.returncode != 0 or not csv.exists():
            print(f"skip {r.model} {r.pass_id}: rc={r.returncode}, csv exists={csv.exists()}")
            continue
        out = work / f"{slug}_{r.pass_id}"
        subprocess.run([sys.executable, str(BASE_DIR / "datalake_evaluate_all.py"), "--methods", key,
                        f"--{key}", str(csv), "--out-dir", str(out)],
                       check=True, capture_output=True, cwd=BASE_DIR)
        eval_ts = next(out.glob("evaluation_summary_*.csv")).stem.removeprefix("evaluation_summary_")
        pred = predictions(slug, r.run_ts)
        digest = r.blob_digest_before if r.blob_digest_before == r.blob_digest_after else \
            f"CHANGED {r.blob_digest_before} -> {r.blob_digest_after}"
        rows.append({"model": r.model, "pass": r.pass_id, "temperature": r.temperature, "seed": r.seed,
                     "ollama_version": r.ollama_version,
                     "blob_digest": digest, **scores_from(out, eval_ts, display),
                     "n_predicted_general": int((pred.security_class == "general").sum()),
                     "wallclock_sec": r.pipeline_total_sec, "run_ts": r.run_ts,
                     "n_predicted_other_security_label": int((~pred.security_class.isin(["general", "confidential"])).sum())})
        labelled.append((r.model, r.pass_id, slug, r.run_ts))

    for model, run_ts, eval_ts, version in REFERENCES:
        key, display, slug = MODELS[model]
        pred = predictions(slug, run_ts)
        log = (OUTPUT_DIR / f"log_llm_{slug}_{run_ts}.txt").read_text(encoding="utf-8")
        m = re.search(r"Total processing time\s*:\s*([\d.]+)s", log)
        ref_dir = EVAL_OUT if (EVAL_OUT / f"evaluation_summary_{eval_ts}.csv").exists() else RESULTS
        rows.append({"model": model, "pass": f"ref_{run_ts}", "temperature": 0.1, "seed": "none",
                     "ollama_version": version,
                     "blob_digest": "not recorded at run time", **scores_from(ref_dir, eval_ts, display),
                     "n_predicted_general": int((pred.security_class == "general").sum()),
                     "wallclock_sec": m.group(1) if m else "", "run_ts": run_ts,
                     "n_predicted_other_security_label": int((~pred.security_class.isin(["general", "confidential"])).sum())})
        labelled.append((model, f"ref_{run_ts}", slug, run_ts))

    out_csv = EVAL_OUT / f"variance_primary_{ts}.csv"
    pd.DataFrame(rows)[COLUMNS].to_csv(out_csv, index=False)

    # Pairwise: how many of the 55 labels differ, and how many LLM inputs differed
    pair_rows = []
    for (ma, pa, sa, ta), (mb, pb, sb, tb) in itertools.combinations(labelled, 2):
        if ma != mb:
            continue
        A, B = predictions(sa, ta), predictions(sb, tb)
        ia, ib = llm_inputs(sa, ta), llm_inputs(sb, tb)
        common = A.index.intersection(B.index)
        pair_rows.append({"model": ma, "pass_a": pa, "pass_b": pb,
                          "security_labels_differing": int((A.loc[common, "security_class"] != B.loc[common, "security_class"]).sum()),
                          "info_type_labels_differing": int((A.loc[common, "info_type"] != B.loc[common, "info_type"]).sum()),
                          "llm_inputs_differing": sum(1 for f in ia if ia[f] != ib.get(f)),
                          "files_with_differing_input": ";".join(f for f in ia if ia[f] != ib.get(f))})
    pd.DataFrame(pair_rows).to_csv(EVAL_OUT / f"variance_primary_{ts}_pairwise.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[COLUMNS].drop(columns=["blob_digest"]).to_string(index=False))
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
