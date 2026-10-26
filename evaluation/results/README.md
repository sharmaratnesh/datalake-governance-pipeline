# evaluation/results

Scored result files that the paper's tables and figures are read from (CC BY 4.0, see
`LICENSE` in this folder). Most files are the direct output of one script run, named
`<analysis>_<timestamp>.csv`; the reproduction map in the top-level README gives the
command for each.

## Assembled file: `evaluation_summary_median_current_runtime.csv`

**This file is not the direct output of a single `datalake_evaluate_all.py` run.** It
combines rows from three runs into the summary that Table 1 and Figures 2–3 report for the
current runtime (Ollama 0.40.7):

| Row(s) | Source |
|---|---|
| Human (avg), Expert Baseline, Rule-Based, NLP | `evaluation_summary_20261021_225854.csv` (October reference run; these methods do not use Ollama) |
| LLM-phi3-mini | median pass **p2**, run `20261022_143804` |
| LLM-llama3.2-1b | median pass **p3**, run `20261022_164754` |

The two LLM rows are each `datalake_evaluate_all.py` scored on that pass alone, e.g.
`python datalake_evaluate_all.py --methods phi3 --phi3 output/metadata_llm_phi3_mini_20261022_143804.csv`
(predictions from the Zenodo deposit, see `scripts/fetch_results_data.sh`). Their accuracy
columns also appear in `variance_primary_20261022_223848.csv`; `time_seconds` is the sum of
per-file processing time, as in every `evaluation_summary_*.csv`. The Human (avg) row needs
the withheld annotator sheets and is copied from the October summary.

`evaluation_summary_20260723_155341.csv` is the July 2026 summary from the superseded
runtime. It is kept as the record of that run, not as the paper's current numbers.
