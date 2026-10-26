# Data Lake Governance Pipeline

Code and evaluation scripts for a paper on content-level information-type and sensitivity
classification for ungoverned multimodal data lakes. Three classification mechanisms
(keyword-based rules, TF-IDF/cosine similarity, and a locally served LLM) are run over a
synthetic 55-file data lake and scored against expert and human baselines. The results are
then externally validated against the Text Anonymization Benchmark (TAB).

Paper: *Content-Level Information-Type and Sensitivity Classification for Ungoverned
Multimodal Data Lakes*. Citation details are in `CITATION.cff`; the venue will be added
once chosen.

**Data.** The 55-file synthetic corpus and all research data (golden labels, aggregate
human scores, per-document predictions, SQLite databases, OWL ontologies) are on Zenodo
(concept DOI [10.5281/zenodo.23551434](https://doi.org/10.5281/zenodo.23551434), all
versions). Version 1 ([10.5281/zenodo.23551435](https://doi.org/10.5281/zenodo.23551435))
holds the corpus; version 2 ([10.5281/zenodo.23580222](https://doi.org/10.5281/zenodo.23580222))
holds everything else.

**Results.** `evaluation/results/` holds the scored CSVs that the paper's tables are read
from, so any number can be checked without running anything. Only the runs the paper
reports are kept there. Superseded and intermediate runs are in the Zenodo deposit
(`superseded_runs.zip`).

> **Read *Known run-to-run variance* below before comparing any LLM number.** The Phi-3
> Mini primary-corpus result changes with the Ollama runtime version, even with identical
> model weights.

---

## Reproduction map

Each row lists what must be fetched first. Every command is pinned to the exact run it
scores. The scoring scripts never pick a file because it is the newest; if a choice is
ambiguous they stop and list the candidates (see *Explicit artefact selection* below).

**Before you start:**
- **(Z)** `bash scripts/fetch_results_data.sh` fetches the golden labels, predictions,
  databases and ontologies from Zenodo v2. It does not run any inference.
- **(C)** `bash scripts/fetch_corpus.sh` fetches the 55-file corpus.
- **(T)** `bash scripts/fetch_tab.sh && python external/tab_adapter.py` fetches TAB at the
  pinned commit and builds `tab_corpus/` and the labels.
- **(G)** needs a GPU host running Ollama in a container (see `cluster/`).
- **(O)** needs a local Ollama install (see *Pinning the model*). It is only needed to
  regenerate predictions; scoring them only needs (Z).

PowerShell twins of every fetch script are in `scripts/`.

Run IDs used below:
- **July 2026 run, which the paper reports:** rule `20260723_134440`, NLP
  `20260723_134532`, LLaMA `20260723_134633`, Phi-3 `20260723_135524`.
- **October 2026 re-run:** rule `20261021_222646`, NLP `20261021_222739`, LLaMA
  `20261021_222833`, Phi-3 `20261021_223725`.
- **TAB laptop runs:** rule `20261021_235159`, NLP `20261021_235257`.

| Paper item | Needs | Command | Published result in `evaluation/results/` |
|---|---|---|---|
| Table 1 and Table 4, current runtime (median passes Phi-3 p2, LLaMA p3: accuracy, Wilson CIs, sensitivity κ, minority-class P/R/F1, 5-pass ranges); Supplementary Table S6 (pairwise κ) | Z | `python datalake_evaluate_all.py --rule-based output/metadata_rule_based_20261021_222646.csv --nlp output/metadata_nlp_20261021_222739.csv --llama output/metadata_llm_llama32_1b_20261022_164754.csv --phi3 output/metadata_llm_phi3_mini_20261022_143804.csv` (October Rule-Based/NLP runs with the two median passes; in a fresh clone the Human (avg) row is absent, see Table 4's human row below). CIs: the current-runtime Table 5 row below. 5-pass ranges: the run-to-run variance row below | `evaluation_summary_median_current_runtime.csv`, `evaluation_kappa_20261026_223258_median_current_runtime.csv` (S6), `evaluation_security_prf_20261026_223258_median_current_runtime.csv`, `evaluation_confidence_intervals_20261026_221836_median_current_runtime.csv`, `variance_primary_20261022_223848.csv`, `primary_distribution_20261022_223948.csv` |
| Table 4, human row (mean of the three annotators on the non-confidential class) | withheld sheets | `python scripts/human_minority_class.py` (runs only where the annotator sheets exist; published as aggregates) | `human_minority_class_20261027.csv` |
| Tables 1, 3 and 4 (accuracy, κ, per-class P/R/F1), July run (superseded runtime) | Z | `python datalake_evaluate_all.py --rule-based output/metadata_rule_based_20260723_134440.csv --nlp output/metadata_nlp_20260723_134532.csv --llama output/metadata_llm_llama32_1b_20260723_134633_FROM_DB.csv --phi3 output/metadata_llm_phi3_mini_20260723_135524.csv` | `evaluation_{summary,kappa,category_prf,security_prf,per_file,human_detail}_20260723_155341.csv` |
| Tables 1, 3 and 4, October reference run (current runtime, single pass; source of the Rule-Based, TF-IDF, human and expert rows) | Z | same command with the October run IDs | `evaluation_*_20261021_225854.csv` |
| Table 5 (phase split), file-type split, McNemar test and Wilson CIs, current runtime (median passes) | Z | `python datalake_evaluate_robustness.py --per-file evaluation/results/evaluation_per_file_median_current_runtime.csv --label median_current_runtime --no-taxonomy`. The per-file input merges the October per-file table with the p2 and p3 per-file tables: `python scripts/merge_per_file.py evaluation/results/evaluation_per_file_20261021_225854.csv <out.csv> --method llm_phi3_mini=<p2 per-file> --method llm_llama3.2_1b=<p3 per-file>`, where the per-file tables come from the two `--methods` commands in the Table 1 row | `evaluation_{phase_split,modality_split,mcnemar_security,confidence_intervals}_20261026_221836_median_current_runtime.csv`, `evaluation_per_file_median_current_runtime.csv` |
| Table 5, McNemar, CIs and LLaMA failure taxonomy, July run (superseded runtime; the paper still quotes this McNemar result and taxonomy) | Z | `python datalake_evaluate_robustness.py --per-file evaluation/results/evaluation_per_file_20260723_155341.csv --llama-log output/log_llm_llama32_1b_20260723_134633.txt` | `evaluation_{confidence_intervals,mcnemar_security,phase_split,modality_split,generator_vs_golden,llama1b_failure_taxonomy}_20260727_120827.csv` |
| LLaMA failure taxonomy, October run | Z | as above, with `--per-file evaluation/results/evaluation_per_file_20261021_225854.csv --llama-log output/log_llm_llama32_1b_20261021_222833.txt` | `evaluation_llama1b_failure_taxonomy_20261022_213729_run_20261021_222833.csv` |
| OCR field accuracy, audio WER | C, Z | `python datalake_evaluate_extraction_quality.py` (needs Tesseract) | `evaluation_ocr_field_accuracy_20260727_224201.csv`, `evaluation_audio_wer_20260727_224201.csv` |
| Table 6 (ontology validity, entity/relation P/R/F1), primary corpus | Z | `python datalake_evaluate_ontology.py --label primary --rule-based-db output/datalake_rule_based_20261021_222646.db --rule-based-owl output/ontology_rule_based_20261021_222646.owl --nlp-db output/datalake_nlp_20261021_222739.db --nlp-owl output/ontology_nlp_20261021_222739.owl --phi3-db output/datalake_llm_phi3_mini_20261021_223725.db --phi3-owl output/ontology_llm_phi3_mini_20261021_223725.owl --llama-db output/datalake_llm_llama32_1b_20261021_222833.db --llama-owl output/ontology_llm_llama32_1b_20261021_222833.owl` | `ontology_eval_20261022_141815_primary.csv` |
| Table 6, TAB (structural; includes the malformed LLaMA ontology) | Z | `python datalake_evaluate_ontology.py --label tab --structural-only --rule-based-db output/datalake_rule_based_20261021_235159.db --rule-based-owl output/ontology_rule_based_20261021_235159.owl --nlp-db output/datalake_nlp_20261021_235257.db --nlp-owl output/ontology_nlp_20261021_235257.owl --phi3-db output/db_llm_phi3_mini_tab_full.db --phi3-owl output/ontology_llm_phi3_mini_20261022_103520.owl --llama-db output/db_llm_llama32_1b_tab_full.db --llama-owl output/ontology_llm_llama32_1b_20261022_023231.owl` | `ontology_eval_20261022_141820_tab.csv` |
| Table 6, primary corpus on Ollama 0.32.1 (malformed LLaMA ontology; Rule-Based and NLP do not use Ollama, so they are the same runs as above) | Z | `python datalake_evaluate_ontology.py --label primary_v0321 --rule-based-db output/datalake_rule_based_20261021_222646.db --rule-based-owl output/ontology_rule_based_20261021_222646.owl --nlp-db output/datalake_nlp_20261021_222739.db --nlp-owl output/ontology_nlp_20261021_222739.owl --phi3-db output/datalake_llm_phi3_mini_20261022_220926.db --phi3-owl output/ontology_llm_phi3_mini_20261022_220926.owl --llama-db output/datalake_llm_llama32_1b_20261022_223019.db --llama-owl output/ontology_llm_llama32_1b_20261022_223019.owl` | `ontology_eval_20261023_075922_primary_v0321.csv` |
| Table 7, Ollama 0.32.1 pass | Z | `python datalake_query_knowledge_graph.py --run-ts 20261022_220926 --results-csv cq_v0321.csv` | `competency_queries_20261022_225900_v0321.csv` |
| Table 7 (SPARQL competency queries), July run | Z | `python datalake_query_knowledge_graph.py --run-ts 20260723_135524 --results-csv cq_july.csv` | `competency_queries_20261022_214213_july_paper.csv` |
| Table 7, current runtime (median-κ pass p2) | Z | `python datalake_query_knowledge_graph.py --run-ts 20261022_143804 --results-csv cq_p2.csv` | `competency_queries_20261022_214213_p2.csv` |
| Table 2 (supervised baselines) | C, T, Z | `python datalake_classical_baselines.py` | `classical_baselines_20261021_233857.csv` |
| Table 2, zero-shot rows (not in `classical_baselines_*.csv`) | Z | read from existing files | **Keyword rules, information type:** accuracy `evaluation_summary_median_current_runtime.csv`, CI `evaluation_confidence_intervals_20261026_221836_median_current_runtime.csv`, macro F1 `evaluation_category_prf_20261021_225854.csv` (macro avg), κ `evaluation_kappa_20261021_225854.csv`. **Phi-3 Mini sensitivity, current runtime (p2):** accuracy `evaluation_summary_median_current_runtime.csv`, CI `evaluation_confidence_intervals_20261026_221836_median_current_runtime.csv`, macro F1 `evaluation_security_prf_20261026_223258_median_current_runtime.csv`, κ `evaluation_kappa_20261026_223258_median_current_runtime.csv`. **Phi-3 Mini sensitivity, superseded runtime:** accuracy `evaluation_summary_20260723_155341.csv`, CI `evaluation_confidence_intervals_20260727_120827.csv`, macro F1 `evaluation_security_prf_20260723_155341.csv`, κ `evaluation_kappa_20260723_155341.csv` |
| Table 8 (TAB external validation) | T, Z | `python external/tab_agreement.py && python datalake_tab_evaluate.py --csv rule_based=output/metadata_rule_based_20261021_235159.csv --csv nlp=output/metadata_nlp_20261021_235257.csv` | `tab_evaluation_20261022_113611.csv`, `external/tab_agreement.json` |
| Table 9 (decoding ablations), scoring | T, Z | `python score_cluster_ablations.py` | `decoding_ablations_20261022_061538.csv` |
| Table 9, re-running the ablations | G, T | `cluster/run_ablations_pbs.sh` (edit the site placeholders first) | — |
| Table 10 (throughput, break-even) | T, Z | `python datalake_tab_throughput.py --csv rule_based=output/metadata_rule_based_20261021_235159.csv --csv nlp=output/metadata_nlp_20261021_235257.csv`, then `python datalake_breakeven_analysis.py --throughput-csv evaluation/output/throughput_<ts>.csv` | `throughput_20261022_113712.csv`, `breakeven_20261022_113718.csv` |
| Span-level agreement (§5.11) | T, Z | `python datalake_tab_span_eval.py --csv rule_based=output/metadata_rule_based_20261021_235159.csv --csv nlp=output/metadata_nlp_20261021_235257.csv` | `tab_span_evaluation_20261022_113637.csv` |
| Table 11 (runtime-version effect) and the run-to-run variance, scoring | Z | `python score_variance_passes.py evaluation/results/variance_runs_20261022_141628.csv evaluation/results/variance_runs_20261022_214018.csv evaluation/results/variance_runs_20261022_220914.csv --default-ollama-version 0.40.7`, then `python score_primary_distribution.py evaluation/output/variance_primary_<ts>.csv` | `variance_primary_20261022_223848.csv` (+ `_pairwise.csv`), `primary_distribution_20261022_223948.csv` |
| Table 3 on the current runtime (per-class information type, median passes: Phi-3 p2, LLaMA p3) | Z | `python datalake_evaluate_all.py --methods phi3 --phi3 output/metadata_llm_phi3_mini_20261022_143804.csv` (LLaMA: `--methods llama --llama output/metadata_llm_llama32_1b_20261022_164754.csv`) | `evaluation_category_prf_20261022_223853_phi3_p2_20261022_143804.csv`, `evaluation_category_prf_20261022_223911_llama_p3_20261022_164754.csv` |
| Table 12 (Phi-3 sensitivity on the 45 non-audio vs 10 audio records) | Z | `python score_nonaudio_subset.py --run july=output/metadata_llm_phi3_mini_20260723_135524.csv --run october=output/metadata_llm_phi3_mini_20261021_223725.csv --run p2_0.40.7=output/metadata_llm_phi3_mini_20261022_143804.csv --run v0321=output/metadata_llm_phi3_mini_20261022_220926.csv` | `nonaudio_subset_20261023_075854.csv` |
| LLaMA unparseable output and `general` fallbacks, per run | Z | `python score_llama_fallbacks.py evaluation/results/variance_runs_20261022_141628.csv evaluation/results/variance_runs_20261022_214018.csv evaluation/results/variance_runs_20261022_220914.csv` | `llama_fallbacks_20261023_075855.csv` |
| Run-to-run variance, regenerating the passes | C, O | `python run_variance_passes.py` (about 3 h on the laptop below) | `variance_runs_*.csv`, `ollama_models_20261022.txt`, `ollama_runtime_test_20261022.txt` |
| Figures 2–3 (current runtime: median passes, as in Table 1) | — | `python generate_comparison_figures.py evaluation/results/evaluation_summary_median_current_runtime.csv` | written to `research_paper/figures/`; input assembled from three runs, see `evaluation/results/README.md`. `evaluation_summary_20260723_155341.csv` is kept only as the superseded-runtime record and no longer drives any figure |
| Figure 1 | — | `python generate_ro1_methodology_figure.py` | written to `research_paper/figures/` |
| Supplementary Table S7 (example Phi-3 rationales, pass p2) | Z | `sqlite3 output/datalake_llm_phi3_mini_20261022_143804.db "SELECT file_name, info_type, security_class, customer_id, llm_reason FROM data_files"` | the DB itself (Zenodo `databases.zip`); `customer_id` is the extracted entity |
| Supplementary Table S8 (file-type split, current runtime) | Z | the current-runtime Table 5 row above | `evaluation_modality_split_20261026_221836_median_current_runtime.csv` |

**Notes on the map:**
- **`output/` and `evaluation/output/`:** the fetch scripts place inputs in `output/`.
  Every scoring script writes a new timestamped file to `evaluation/output/`, which you can
  diff against the published copy in `evaluation/results/`.
- **Human rows:** in a fresh clone, `datalake_evaluate_all.py` produces every row except
  the *Human (avg)* row and `evaluation_human_detail_*.csv`, because those need the withheld
  annotator sheets (next section). The published copies of those two are in
  `evaluation/results/`.
- **The July LLaMA input** is `…_20260723_134633_FROM_DB.csv`. The original CSV was lost.
  This file is a lossless export of the same run's database, and scoring it reproduces the
  July evaluation byte-for-byte.

**Verification status.**
- **Clean-clone check.** Every row marked Z, C or T was re-run from a fresh copy of this
  repository, with data fetched by `scripts/fetch_results_data.sh` from the staged Zenodo v2
  files. Each published CSV listed above was reproduced exactly; the two exceptions are the
  human rows, as noted, and the regeneration timestamp inside `external/tab_agreement.json`.
- **Not re-run from scratch:** the GPU rows (Table 9 re-run, the cluster arm of Table 10).
  Their scoring was verified against the raw cluster predictions.

---

## Why the three annotator sheets are not published

The three practitioners' completed classification sheets are individual human-subject data,
collected under ethics approval ETH25-11255, and are withheld. What is published instead:

- **`evaluation_human_detail_*.csv`:** per-participant aggregate scores (accuracy per task
  and time, as pseudonymous P1–P3).
- **`human_minority_class_20261027.csv`:** per-participant scores on the minority
  (non-confidential) sensitivity class: records labelled non-confidential, precision,
  recall, F1. The mean row is Table 4's human row.
- **`evaluation_per_file_*.csv`:** the three-annotator **majority label** per record for
  information type and linkage, and whether it matches the golden label
  (`human_majority_*`). The 18 per-annotator columns were removed by
  `scripts/aggregate_per_file.py`.

**Two per-record summaries are deliberately not published:**
- **A count of annotators correct.** A unanimous count reveals each annotator's answer
  for that record.
- **The majority label for sensitivity.** Combined with the per-participant accuracies,
  it would identify which records each annotator got wrong.

**What can be recomputed from public data:**
- **Human aggregates:** every per-participant figure, from the aggregate files above.
- **Human majority vote for information type:** recomputed by
  `datalake_evaluate_robustness.py` from the per-file CSVs.
- **Human majority vote for sensitivity** (the human sensitivity rows of the confidence
  intervals and phase split): computed from the withheld sheets and published only as
  results, in the committed `evaluation_confidence_intervals_*` and
  `evaluation_phase_split_*` files. Re-running the script on the public per-file CSVs
  leaves those rows blank.

---

## Known run-to-run variance

**Headline:** the Phi-3 Mini result on the 55-file primary corpus depends on the Ollama
runtime version. The weights are the same in every row below
(`sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf`, 3.8B, Q4_0), as
are the prompt, inputs and decoding settings. Only the Ollama runtime differs:

| Runtime | Passes | Phi-3 sensitivity κ | Records labelled general (of 55; golden: 6) | LLaMA 3.2 1B template echoes (of 55) |
|---|---|---|---|---|
| July 2026 run (paper as first submitted; Ollama version not recorded) | 1, T=0.1 | **0.000** | **0** | 28 |
| Ollama 0.32.1 (newest release the week before the July run) | 1, T=0 | **0.000** | **0** | 30 |
| Ollama 0.34.2 | 1, T=0 | 0.813 | 6 | 0 |
| Ollama 0.40.7 | 2, T=0 (identical) | 0.813 | 6 | 0 |
| Ollama 0.40.7 | 5, T=0.1, seeds 1–5 | 0.813–0.899 (median 0.899) | 5–6 | 0 |
| October 2026 re-run, Ollama 0.40.7 | 1, T=0.1 | 0.899 | 5 | 0 |

**What this shows:**
- **The July collapse did not reproduce on the current runtime.** None of the 9 runs on
  0.34.2 or 0.40.7 reproduced it (8 new passes plus the October re-run).
- **The old runtime reproduces it.** Ollama 0.32.1 reproduces the collapse exactly: Phi-3
  gives the same security label as the July run to all 55 files.
- **Identical inputs, opposite results.** One 0.40.7 pass and the 0.32.1 pass received
  byte-identical model inputs, and they still differ on 5 security labels.
- **The downstream artefacts follow the runtime too.**
  - **Phi-3 graph on 0.32.1:** the SPARQL competency queries return the July values (264
    triples; 55 confidential; 18 of 18; 4 linked; 51 orphaned). The only difference is one
    financial-transactions count, 14 vs 15.
  - **LLaMA ontology on 0.32.1:** malformed in the same way as the TAB one (an unescaped
    `<identities|…>` class IRI).
  - **On 0.34.2 and 0.40.7:** every primary-corpus ontology parses.
- **Temperature 0 is deterministic on each version,** so this is not sampling noise.
- **Where the change lies:** between 0.32.1 and 0.34.2. On a probe prompt, 0.34.2 and 0.40.7
  produce identical text, while 0.32.1 does not.
- **Cause:** the inference runtime version, with weights, prompt, inputs and temperature
  held fixed, inverts the headline sensitivity metric.
- **The July build itself cannot be identified:** no Ollama log from July survives.

**What it does not affect: the TAB external validation.** That collapse (Phi-3 laptop
κ = 0.000 against the adjudicated labels; all six methods between −0.039 and 0.044) was
produced on Ollama 0.40.7 and holds on the current runtime. The result that failed to
replicate is the primary-corpus one, on the easier corpus.

**Other sources of variance:**
- **faster-whisper transcription** is not bit-identical between processes. Up to 6 of the
  10 audio transcripts differ slightly between runs. This does not explain the July result:
  the identical-input comparison above controls for it.
- **On 0.34.2 and 0.40.7, every "general" label LLaMA 3.2 1B produces is the pipeline's
  parse-failure fallback** (checked in all 9 runs). LLaMA never emits "general" itself, so its `general` count measures broken
  JSON, not discrimination.

Full per-pass results: `evaluation/results/variance_primary_20261022_223848.csv`. The spread
per metric is in `primary_distribution_20261022_223948.csv`. How the old builds were obtained
and run is in `ollama_runtime_test_20261022.txt`.

### Pinning the model

A model tag such as `phi3:mini` does not fully specify a result. Pin **both** the weights
digest and the Ollama version:

```bash
ollama --version                       # the paper's revised results use 0.40.7
ollama pull phi3:mini                  # once, yourself; the pipeline no longer pulls
ollama show phi3:mini --modelfile | grep ^FROM   # expect ...sha256-633fc5be925f9a48...
OLLAMA_MODEL=phi3:mini python datalake_llm_pipeline.py \
    --require-digest sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf
```

**How the pipeline handles provenance now:**
- **It never pulls by default.** Earlier versions called `/api/pull` on every run, which can
  silently re-resolve the tag. Opt back in with `--allow-pull`.
- **It records provenance on every run.** The Ollama version and the resolved weights
  digest are logged at start-up and written into every metadata row (`ollama_version`,
  `model_digest`) and into the run DB's `run_info` table.
- **It can enforce the digest.** `--require-digest` aborts if the model resolves to
  anything else.
- **The cluster runners match.** `cluster/ollama_tab_runner.py` and
  `cluster/decoding_ablations_cluster.py` take the same `--require-digest` and record the
  same two fields. They still pull, because a fresh container has no models.

LLaMA 3.2 1B (1.2B, Q8_0):
`sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45`.

**The Ollama desktop app updates itself.** Turn off automatic updates, or record
`ollama --version` with every run.

**On the laptop that produced these results, Ollama is held at 0.40.7.** It had already
downloaded the 0.40.8 installer when this was done (2026-10-23):
- **The staged installer was moved** out of `%LOCALAPPDATA%\Ollama\updates_v2\` into
  `%LOCALAPPDATA%\Ollama\updates_quarantined_v0.40.8\`. The updater neither scans that folder
  nor installs from it.
- **Automatic updates were switched off** by setting `auto_update_enabled = 0` in the
  `settings` table of `%LOCALAPPDATA%\Ollama\db.sqlite`. That is the setting behind the app's
  "automatic updates" toggle. It was changed with the app closed, and the app was then
  restarted.
- **Effect:** the hourly checker still logs that 0.40.8 is available but no longer downloads
  it.
- **To reverse:** set the flag back to 1, or use the app's settings toggle. Optionally move the
  quarantined installer back.

---

## Explicit artefact selection

`output/` normally holds runs over several corpora (55-file, 6-file TAB smoke test,
1,268-document TAB) and several runs per corpus. An earlier version of the ontology
evaluation picked the newest file and silently scored TAB ontologies against the 55-file
golden table.

All scoring scripts now select inputs through `datalake_artefacts.py`:
- **Pinned file:** a file passed on the command line is used as given, after a corpus check.
- **Single candidate:** otherwise, a file is chosen automatically only if it is the *only*
  candidate from the right corpus.
- **Ambiguous:** if several qualify, the script exits with status 2 and lists them.
- **Corpus guards:** a DB or CSV whose files are not drawn from the golden table's file set
  is rejected, so a TAB run cannot be scored against the 55-file golden labels.

The unpinned invocation `python datalake_evaluate_ontology.py` is therefore no longer
permitted. Pass `--run-tag` or per-method `--<method>-db/--<method>-owl` as in the map above.

---

## Two hardware profiles

The paper reports results from two platforms. Scripts under `cluster/` assume the second;
everything else runs on the first.

- **Commodity laptop:**
  - Intel Core Ultra 5 125U, 12 cores / 14 threads, 16 GB RAM, integrated graphics only,
    Windows 11.
  - All primary-corpus results and the laptop arm of the external validation were produced
    here.
  - A Phi-3 Mini pass takes about 20 minutes over the 55-file corpus and about 8 hours over
    the full 1,268-document TAB corpus.
- **GPU cluster:**
  - PBS Pro, containerised: no native Ollama install; see `cluster/`'s apptainer-based job
    scripts.
  - Used for the Phi-3 Mini anchor run, the LLaMA 3.1 8B scale probe, and the five decoding
    ablations.
  - Queue names, resource limits and the login host in `cluster/*.sh` are placeholders
    (`<your-hpc-login-node>`, `small_gpuq`, etc.). They are **site-specific** and must be
    edited for your own cluster before submitting.

## Two caveats that materially affect how to read the results

- **Span-level metrics for the four pipeline methods are a proxy, not a measurement.**
  - None of rule-based, TF-IDF/NLP or either LLM pipeline predicts character spans; each
    emits one document-level label.
  - `datalake_tab_span_eval.py` scores them anyway by treating a "confidential" document
    prediction as a single predicted span covering the *entire* document. That reflects
    document length, not real span extraction, and is **not** reported as a classifier
    result in the paper.
  - Only the human span-level agreement baseline (κ=0.416 exact-match, κ=0.492
    partial-overlap) is a sound measurement, and it is what the paper cites.
- **The TAB document-level label is noisy.**
  - Human pairwise agreement on the derived `security_confidential_only` label is κ=0.162
    on the adjudicated subset (115 docs, 1,066 annotator-pair comparisons): "slight"
    agreement on the conventional Landis & Koch scale.
  - Read every classifier kappa against this label against that ceiling, not against 1.0.
    See `external/tab_agreement.json` for the full agreement analysis.

## Platform differences on TAB

**Same tag, different labels across platforms.** The same model tag, prompt and
temperature gave different label distributions on the two platforms:
- **Laptop (native Ollama 0.40.7):** labelled all 1,268 TAB documents `confidential` for
  Phi-3 Mini (κ=0.000 by construction).
- **Containerised GPU run:** labelled 82 of them `general`.

**Cause unknown.** The cluster container's Ollama version was not recorded. Given the
runtime-version effect above, a version difference is a plausible cause, but it is
unconfirmed. The cluster runners now record it.

**If you re-run, report your full stack:** serving platform (native install or
container), OS, Ollama version, weights digest, and CPU/GPU, alongside the model tag.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Check any number in the paper without inference:
bash scripts/fetch_results_data.sh   # golden labels, predictions, DBs, ontologies (Zenodo v2)
# ...then run the matching command from the reproduction map.

# Regenerate predictions (needs Ollama; see "Pinning the model"):
bash scripts/fetch_corpus.sh     # the 55-file corpus (Zenodo)
python datalake_rule_based_pipeline.py
python datalake_nlp_pipeline.py
OLLAMA_MODEL=llama3.2:1b python datalake_llm_pipeline.py --require-digest sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45
OLLAMA_MODEL=phi3:mini   python datalake_llm_pipeline.py --require-digest sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf
python datalake_evaluate_all.py --rule-based output/metadata_rule_based_<ts>.csv --nlp output/metadata_nlp_<ts>.csv \
    --llama output/metadata_llm_llama32_1b_<ts>.csv --phi3 output/metadata_llm_phi3_mini_<ts>.csv
```

TAB external validation:

```bash
bash scripts/fetch_tab.sh               # clones TAB at the pinned commit
python external/tab_adapter.py          # builds tab_corpus/ and the three label derivations
python external/tab_agreement.py        # inter-annotator agreement (Table 8, §5.11)
python run_checkpointed_llm.py --model phi3:mini --swamp-dir tab_corpus --tag tab_full   # ~8 h, resumable
python datalake_tab_evaluate.py --csv rule_based=<csv> --csv nlp=<csv>   # pin whichever runs you produced
```

**GPU cluster steps** (Table 9, the anchor and scale-probe rows of Table 10):
- **Needs:** an apptainer-capable PBS cluster. See `cluster/run_tab_pbs.sh` and
  `cluster/run_ablations_pbs.sh`, and edit the site-specific placeholders first.
- **After a job finishes:**
  1. Convert its raw `cluster_results/<tag>.jsonl` with
     `python cluster/convert_cluster_results.py`.
  2. Re-run the TAB scoring scripts with the resulting `metadata_cluster_*.csv` files.

---

## Licence, citation, and attribution

- **Licence split:** MIT for code; CC BY 4.0 for the corpus, the Zenodo data and
  `evaluation/results/`.
- **Code:** MIT. See `LICENSE`.
- **Results** (`evaluation/results/`): CC BY 4.0. These are data and results, not code. See
  `evaluation/results/LICENSE`.
- **Zenodo deposit** (corpus, labels, predictions): CC BY 4.0. See `LICENSE-DATA`.
  https://zenodo.org/records/23551435, concept DOI 10.5281/zenodo.23551434.
- **Citing this work:** see `CITATION.cff`.
- **Third-party attribution:**
  - [Text Anonymization Benchmark (TAB)](https://github.com/NorskRegnesentral/text-anonymization-benchmark)
    (Pilán, Lison, Øvrelid, Papadopoulou, Sánchez & Batet, 2022, *Computational
    Linguistics*): fetched, not vendored. MIT licence, pinned at commit `558e09e` (see
    `scripts/fetch_tab.sh`).
  - [Ollama](https://ollama.com) is used for all local LLM inference.
  - Model weights are pulled via Ollama under their own licences: Microsoft's MIT licence
    for Phi-3, and Meta's Llama 3 Community License for LLaMA 3.2/3.1. The latter's full
    terms, including acceptable-use and attribution requirements, are embedded in
    `evaluation/model_llama32_1b_modelfile.txt`. Review that licence before redistributing
    any LLaMA-derived output.

## Hardware prerequisites

| Component | RAM | Disk | GPU |
|---|---|---|---|
| Scoring only (any row marked Z) | 2 GB | 1 GB | Not required |
| Rule-Based / NLP pipeline | 4 GB | 5 GB | Not required |
| LLM pipeline (LLaMA 3.2 1B, Phi-3 Mini) | 8 GB | 5 GB | Not required (CPU-only works, slowly) |
| LLM pipeline (LLaMA 3.1 8B / larger) | 16 GB+ | 10 GB+ | Strongly recommended |

**Other dependencies:**
- **Tesseract OCR** is required for the image-extraction path (`pytesseract`). It is not
  pip-installable: see https://github.com/UB-Mannheim/tesseract/wiki on Windows, or use your
  package manager on Linux/macOS.
- **faster-whisper** downloads its model automatically on first use and caches it locally.
