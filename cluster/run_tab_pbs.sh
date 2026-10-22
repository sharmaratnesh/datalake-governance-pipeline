#!/bin/bash
# PBS job script — Step 5b (cluster anchor / scale probe).
#
# Queue names, limits and the login host below are SITE-SPECIFIC — adjust them for your
# own cluster. Queue limits queried via `qstat -Qf small_gpuq med_gpuq` on
# <your-hpc-login-node>:
#   small_gpuq: resources_max.ncpus=12, resources_max.walltime=48:00:00, min ngpus=1
#   med_gpuq:   resources_max.ncpus=12, resources_max.walltime=24:00:00, min ngpus=1
#   Compute nodes (pbsnodes -a): 24-48 ncpus, 2 ngpus, 186-395gb mem per node.
# This script requests a modest 1 GPU / 4 cpus / 16gb — comfortably within both queues'
# per-chunk limits, and far below the 48h/24h walltime ceilings for the default 8h below.
#
# Ollama is not installed on this cluster (no module, no system binary) but apptainer and
# internet access from the login node are both available, so this job runs the official
# ollama/ollama container image via `apptainer exec --nv` (GPU passthrough) and drives it
# with cluster/ollama_tab_runner.py — a self-contained stdlib-only script (no pandas/
# networkx/pytesseract/faster_whisper — those are the laptop pipeline's deps, unverified
# on this cluster's python3.6.8) using the SAME prompt/parser as datalake_llm_pipeline.py
# for Step 5b's "same prompt, same parser" comparability requirement.
#
# Usage: submit from your HPC login node after manually staging ~/tab_experiment/ (an
# apptainer .sif pulled from docker://ollama/ollama, the TAB corpus text files under
# tab_experiment/corpus/, and this repo's cluster/ scripts under tab_experiment/scripts/):
#   qsub -v MODEL=phi3:mini,TAG=anchor_phi3_mini cluster/run_tab_pbs.sh
#   qsub -v MODEL=llama3.1:8b,TAG=scale_probe_llama31_8b cluster/run_tab_pbs.sh
#
#PBS -N tab_ollama
#PBS -q small_gpuq
#PBS -l select=1:ncpus=12:ngpus=1:mem=24gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o <path-to-your-home-directory>/tab_experiment/logs/

set -euo pipefail

: "${MODEL:?set via qsub -v MODEL=...}"
: "${TAG:?set via qsub -v TAG=...}"
LIMIT="${LIMIT:-}"

EXP_DIR="$HOME/tab_experiment"
SIF="$EXP_DIR/sif/ollama.sif"
CORPUS_SRC="$EXP_DIR/corpus"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"

SCRATCH="${TMPDIR:-/tmp/$PBS_JOBID}"
mkdir -p "$SCRATCH/models" "$SCRATCH/corpus" "$SCRATCH/results"

echo "=== Job $PBS_JOBID on $(hostname) — model=$MODEL tag=$TAG ==="
echo "Scratch: $SCRATCH"
nvidia-smi 2>&1 || echo "nvidia-smi not available on this node"

echo "Staging corpus to node-local scratch..."
cp -r "$CORPUS_SRC"/*.txt "$SCRATCH/corpus/"
echo "Staged $(ls "$SCRATCH/corpus" | wc -l) files"

echo "Starting Ollama server inside container (GPU passthrough via --nv)..."
# Ollama's Go runtime detects the NODE's full core count (24-48 here), not the PBS cgroup
# allocation, and sizes its internal thread pool accordingly -- this is what caused the
# first two attempts to be killed by PBS for exceeding their ncpus request even though the
# GPU was doing the actual inference. Cap it explicitly to match what was requested.
export APPTAINERENV_OLLAMA_MODELS="/models"
export APPTAINERENV_GOMAXPROCS=12
export APPTAINERENV_OLLAMA_NUM_PARALLEL=1
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS=1
apptainer exec --nv \
  --bind "$SCRATCH/models:/models" \
  "$SIF" ollama serve > "$SCRATCH/ollama_server.log" 2>&1 &
OLLAMA_PID=$!
sleep 8

PYBIN=python3
command -v python3 >/dev/null 2>&1 || PYBIN=python

LIMIT_ARG=()
if [ -n "$LIMIT" ]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

"$PYBIN" "$EXP_DIR/scripts/ollama_tab_runner.py" \
  --model "$MODEL" \
  --corpus-dir "$SCRATCH/corpus" \
  --out-jsonl "$SCRATCH/results/${TAG}.jsonl" \
  --platform "hpc-gpu-${TAG}" \
  "${LIMIT_ARG[@]}"

kill "$OLLAMA_PID" 2>/dev/null || true

cp "$SCRATCH/results/${TAG}.jsonl" "$EXP_DIR/results/" 2>/dev/null || echo "WARNING: no results file produced"
echo "=== Done. Results copied to $EXP_DIR/results/${TAG}.jsonl ==="
