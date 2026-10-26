# evaluation/results

Scored result files that the paper's tables and figures are read from (CC BY 4.0, see
`LICENSE` in this folder). Most files are the direct output of one script run, named
`<analysis>_<timestamp>.csv`; the reproduction map in the top-level README gives the
command for each.

## `evaluation_summary_median_current_runtime.csv`

The summary that Table 1 and Figures 2–3 report for the current runtime (Ollama 0.40.7). Its
rows come from three runs: the October reference run for Rule-Based, NLP, Human (avg) and
Expert Baseline (these do not use Ollama), and the median passes for the two LLMs:

| Row(s) | Source |
|---|---|
| Human (avg), Expert Baseline, Rule-Based, NLP | October reference run (`evaluation_summary_20261021_225854.csv`) |
| LLM-phi3-mini | median pass **p2**, run `20261022_143804` |
| LLM-llama3.2-1b | median pass **p3**, run `20261022_164754` |

It is reproduced exactly by one run pinned to those four prediction files:
`python datalake_evaluate_all.py --rule-based output/metadata_rule_based_20261021_222646.csv --nlp output/metadata_nlp_20261021_222739.csv --llama output/metadata_llm_llama32_1b_20261022_164754.csv --phi3 output/metadata_llm_phi3_mini_20261022_143804.csv`.
The same run writes `evaluation_kappa_20261026_223258_median_current_runtime.csv`
(Supplementary Table S6) and
`evaluation_security_prf_20261026_223258_median_current_runtime.csv` (minority-class P/R/F1,
Table 2's current Phi-3 macro F1). Without the withheld annotator sheets, i.e. in a fresh
clone, the Human (avg) rows are absent from its output.

`evaluation_summary_20260723_155341.csv` is the July 2026 summary from the superseded
runtime. It is kept as the record of that run, not as the paper's current numbers.

## `evaluation_per_file_median_current_runtime.csv`

The per-file counterpart of the summary above, and the input to the current-runtime
robustness outputs. It is `evaluation_per_file_20261021_225854.csv` (October reference run)
with the `llm_phi3_mini_*` columns taken from pass p2's per-file table and the
`llm_llama3.2_1b_*` columns from pass p3's, built with `scripts/merge_per_file.py` (which
refuses to merge unless files, order and golden labels agree). Like every published per-file
table it carries the three-annotator majority for information type and linkage only, never
individual annotators' answers (see the top-level README on why the sensitivity majority is
not published).

Produced from it by
`python datalake_evaluate_robustness.py --per-file evaluation/results/evaluation_per_file_median_current_runtime.csv --label median_current_runtime --no-taxonomy`:

- `evaluation_phase_split_20261026_221836_median_current_runtime.csv`: Table 5.
- `evaluation_modality_split_20261026_221836_median_current_runtime.csv`: file-type split.
- `evaluation_mcnemar_security_20261026_221836_median_current_runtime.csv`: exact McNemar on
  sensitivity correctness between the two strongest sensitivity methods (Phi-3 p2 and TF-IDF).
- `evaluation_confidence_intervals_20261026_221836_median_current_runtime.csv`: Wilson 95% CIs
  for Table 1's current-runtime rows.

The LLaMA failure taxonomy is not regenerated for the current runtime (`--no-taxonomy`): the
paper reports it as a pre-0.34 phenomenon, and its current-runtime counterpart is
`llama_fallbacks_20261023_075855.csv`. The `*_20260727_120827.csv` robustness files are the
superseded-runtime (July) versions.


## `human_minority_class_20261027.csv`

Per-participant scores on the minority (non-confidential) sensitivity class: records labelled
non-confidential, precision, recall and F1 for P1–P3, and their mean (Table 4's human row).
Computed by `scripts/human_minority_class.py` from the withheld annotator sheets and
published as aggregates only. Like the per-participant accuracies, they say how many
non-confidential records each annotator found, not which.

## Human sensitivity majority in the robustness outputs

The human sensitivity rows of `evaluation_confidence_intervals_*` and
`evaluation_phase_split_*` were computed from the withheld annotator sheets. The per-record
sensitivity majority they rest on is not published, so re-running
`datalake_evaluate_robustness.py` on the public per-file CSVs leaves those rows blank. All
other rows reproduce exactly.
