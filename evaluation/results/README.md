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

## Assembled file: `evaluation_per_file_median_current_runtime.csv`

The per-file counterpart of the summary above, and the input to the current-runtime
robustness outputs. It is `evaluation_per_file_20261021_225854.csv` (October reference run)
with the `llm_phi3_mini_*` columns taken from pass p2's per-file table and the
`llm_llama3.2_1b_*` columns from pass p3's, built with `scripts/merge_per_file.py` (which
refuses to merge unless files, order and golden labels agree). Like every published per-file
table it carries the three-annotator majority, not individual annotators' answers.

Produced from it by
`python datalake_evaluate_robustness.py --per-file evaluation/results/evaluation_per_file_median_current_runtime.csv --label median_current_runtime --no-taxonomy`:

- `evaluation_phase_split_20261026_221836_median_current_runtime.csv`: Table 4.
- `evaluation_modality_split_20261026_221836_median_current_runtime.csv`: file-type split.
- `evaluation_mcnemar_security_20261026_221836_median_current_runtime.csv`: exact McNemar on
  sensitivity correctness between the two strongest sensitivity methods (Phi-3 p2 and TF-IDF).
- `evaluation_confidence_intervals_20261026_221836_median_current_runtime.csv`: Wilson 95% CIs
  for Table 1's current-runtime rows.

The LLaMA failure taxonomy is not regenerated for the current runtime (`--no-taxonomy`): the
paper reports it as a pre-0.34 phenomenon, and its current-runtime counterpart is
`llama_fallbacks_20261023_075855.csv`. The `*_20260727_120827.csv` robustness files are the
superseded-runtime (July) versions.
