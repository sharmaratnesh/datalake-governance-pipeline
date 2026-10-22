#!/bin/bash
# Fetch the Text Anonymization Benchmark (TAB) at the exact commit this paper's results
# were produced from. Idempotent: safe to re-run, does not re-clone if already at the
# right commit. See external/TAB_VERSION.txt for provenance details.
set -euo pipefail

PINNED_SHA="558e09e26d6b36f5f78440074e6a233946d98bd9"
REPO_URL="https://github.com/NorskRegnesentral/text-anonymization-benchmark"
DEST="external/tab"

echo "=== TAB citation ==="
echo "Pilan, I., Lison, P., Ovrelid, L., Papadopoulou, A., Sanchez, D., & Batet, M. (2022)."
echo "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework"
echo "for Text Anonymization. Computational Linguistics."
echo "Licence: MIT (see ${REPO_URL}/blob/master/LICENSE)"
echo "============================================================================"

if [ -d "$DEST/.git" ]; then
    current_sha=$(git -C "$DEST" rev-parse HEAD)
    if [ "$current_sha" = "$PINNED_SHA" ]; then
        echo "Already at pinned commit $PINNED_SHA — nothing to do."
        exit 0
    fi
    echo "Existing clone at $current_sha, not the pinned commit — checking out $PINNED_SHA."
    git -C "$DEST" fetch origin
    git -C "$DEST" checkout "$PINNED_SHA"
else
    echo "Cloning $REPO_URL into $DEST ..."
    git clone "$REPO_URL" "$DEST"
    git -C "$DEST" checkout "$PINNED_SHA"
fi

actual_sha=$(git -C "$DEST" rev-parse HEAD)
if [ "$actual_sha" != "$PINNED_SHA" ]; then
    echo "FATAL: expected commit $PINNED_SHA, landed on $actual_sha instead." >&2
    exit 1
fi

echo "Checked out TAB at $actual_sha — matches external/TAB_VERSION.txt."
echo "Done. Run external/tab_adapter.py next to build tab_corpus/ and the derived labels."
