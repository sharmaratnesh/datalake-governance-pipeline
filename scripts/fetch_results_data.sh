#!/bin/bash
# Fetch the golden labels, per-document predictions, SQLite databases and OWL ontologies
# from version 2 of the Zenodo deposit, and place each file where the scripts expect it:
#
#   golden_labels.zip  -> evaluation/        (golden labels, OCR/audio ground truth)
#   predictions.zip    -> output/            (laptop metadata_*.csv + run logs, cluster CSVs)
#                         cluster_results/   (raw cluster *.jsonl)
#   databases.zip      -> output/            (*.db)
#   ontologies.zip     -> output/            (*.owl)
#
# This lets every scoring script be re-run without any LLM inference.
#
# Version 2 of the deposit: https://zenodo.org/records/23580222 (DOI 10.5281/zenodo.23580222).
# For testing against a local copy, set ZENODO_FILES_URL to a directory URL, e.g.
#   ZENODO_FILES_URL=file:///path/to/upload bash scripts/fetch_results_data.sh
set -euo pipefail

ZENODO_V2_RECORD_ID="23580222"
BASE_URL="${ZENODO_FILES_URL:-https://zenodo.org/api/records/$ZENODO_V2_RECORD_ID/files}"

declare -A MD5=(
    [golden_labels.zip]="eb716192dc5f4eaef43a6c177f5f8b63"
    [predictions.zip]="848c013cbcacd5cec33b5641fb7e0a75"
    [databases.zip]="2533bbd72bfd9b6aae64f1ee86055080"
    [ontologies.zip]="ee1183a8864cb6cfd78a686268669db8"
)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p evaluation output cluster_results

for f in golden_labels.zip predictions.zip databases.zip ontologies.zip; do
    if [[ "$BASE_URL" == file://* ]]; then
        cp "${BASE_URL#file://}/$f" "$TMP/$f"
    else
        curl -fL -o "$TMP/$f" "$BASE_URL/$f/content"
    fi
    actual=$(md5sum "$TMP/$f" | awk '{print $1}')
    if [ "$actual" != "${MD5[$f]}" ]; then
        echo "FATAL: $f checksum mismatch: expected ${MD5[$f]}, got $actual" >&2
        exit 1
    fi
    unzip -q -o "$TMP/$f" -d "$TMP/x"
    echo "$f: checksum OK"
done

cp "$TMP"/x/golden_labels/* evaluation/
cp "$TMP"/x/predictions/laptop/* output/
cp "$TMP"/x/predictions/cluster/*.csv output/
cp "$TMP"/x/predictions/cluster/*.jsonl cluster_results/
cp "$TMP"/x/databases/* output/
cp "$TMP"/x/ontologies/* output/

echo "Done: $(ls evaluation/*.xlsx | wc -l) label workbooks in evaluation/, $(ls output | wc -l) files in output/, $(ls cluster_results | wc -l) in cluster_results/."
echo "The three annotators' individual sheets are not published (see README); scores that need them are in evaluation/results/."
