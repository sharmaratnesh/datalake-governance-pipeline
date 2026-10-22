# Changelog

## 1.1.0 — results, pinned evaluation, runtime-variance finding

- `evaluation/results/`: the scored CSVs the paper's tables are read from (July 2026 runs
  the paper reports, October 2026 re-run, TAB, robustness, ontology, SPARQL), plus the
  run-to-run variance and Ollama-version test (`variance_primary_*`, `primary_distribution_*`,
  `ollama_runtime_test_*`).
- Explicit artefact selection (`datalake_artefacts.py`): no script picks a file because it is
  the newest. Ambiguous selections exit non-zero and list the candidates; corpus guards
  reject a run scored against the wrong golden table. Fixes the ontology evaluation scoring
  TAB ontologies against the 55-file golden labels, and `datalake_evaluate_all.py` counting
  any `evaluation/*.xlsx` (including `tab_golden.xlsx`) as a participant.
- `datalake_evaluate_ontology.py`: `--run-tag` / per-method `--*-db`/`--*-owl`,
  `--structural-only`, parse-error and offending-IRI reporting.
- `datalake_llm_pipeline.py`: no longer pulls the model on every run (`--allow-pull` to opt
  in); records the Ollama version and weights digest in every metadata row and the DB;
  `--require-digest` aborts on a mismatch. Same digest check and provenance in the cluster
  runners.
- `datalake_query_knowledge_graph.py`: explicit run selection (`--run-ts`), DB/ontology
  same-run check, `--results-csv`.
- `run_variance_passes.py`, `score_variance_passes.py`, `score_primary_distribution.py`:
  the run-to-run variance study. `scripts/fetch_results_data.{sh,ps1}`: fetch the Zenodo v2
  data needed to re-score without inference. `scripts/aggregate_per_file.py`: removes
  per-annotator columns from per-file results.
- `score_nonaudio_subset.py` (sensitivity on the 45 non-audio vs 10 audio records) and
  `score_llama_fallbacks.py` (unparseable LLaMA replies and `general` fallbacks per run).
- Licence split: `evaluation/results/` is CC BY 4.0 (`evaluation/results/LICENSE`); code stays MIT.
- README: executable reproduction map, why the annotator sheets are withheld, known
  run-to-run variance and how to pin the model.

## 1.0.0 — initial public release

Code and evaluation scripts for content-level information-type and sensitivity
classification in ungoverned multimodal data lakes, released alongside the paper.

- Three classification pipelines (rule-based, TF-IDF/NLP, local LLM via Ollama) over a
  55-file synthetic multimodal corpus, scored against a four-expert golden label set and
  three independent human annotators.
- External validation against the Text Anonymization Benchmark (TAB), including
  inter-annotator agreement analysis, span-level evaluation, decoding ablations, and a
  laptop-CPU vs GPU-cluster throughput/break-even comparison.
- Conventional supervised baselines (TF-IDF + Logistic Regression / Linear SVM) for both
  corpora.

See the README's reproduction map for which script produces which result in the paper.
