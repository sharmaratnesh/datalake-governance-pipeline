#!/bin/bash
# PBS job script — Step 7 decoding ablations (LLaMA 3.2 1B, 5 conditions, full 1268 corpus).
# Same GOMAXPROCS fix as run_tab_pbs.sh (see that script's comment for why it's needed).
# Queue name/limits and the output path below are SITE-SPECIFIC — adjust for your cluster.
#
#PBS -N tab_ablations
#PBS -q small_gpuq
#PBS -l select=1:ncpus=12:ngpus=1:mem=24gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o <path-to-your-home-directory>/tab_experiment/logs/

set -euo pipefail

EXP_DIR="$HOME/tab_experiment"
SIF="$EXP_DIR/sif/ollama.sif"
CORPUS_SRC="$EXP_DIR/corpus"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"

SCRATCH="${TMPDIR:-/tmp/$PBS_JOBID}"
mkdir -p "$SCRATCH/models" "$SCRATCH/corpus" "$SCRATCH/results"

echo "=== Ablations job $PBS_JOBID on $(hostname) ==="
nvidia-smi 2>&1 || echo "nvidia-smi not available on this node"

echo "Staging corpus to node-local scratch..."
cp -r "$CORPUS_SRC"/*.txt "$SCRATCH/corpus/"
echo "Staged $(ls "$SCRATCH/corpus" | wc -l) files"

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

"$PYBIN" "$EXP_DIR/scripts/decoding_ablations_cluster.py" \
  --model "llama3.2:1b" \
  --corpus-dir "$SCRATCH/corpus" \
  --out-jsonl "$SCRATCH/results/ablations.jsonl" \
  --conditions a,b,c,d,e

kill "$OLLAMA_PID" 2>/dev/null || true

cp "$SCRATCH/results/ablations.jsonl" "$EXP_DIR/results/" 2>/dev/null || echo "WARNING: no results file produced"
echo "=== Done. Results copied to $EXP_DIR/results/ablations.jsonl ==="
