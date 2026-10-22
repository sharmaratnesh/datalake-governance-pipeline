#!/bin/bash
# Fetch the synthetic "data swamp" corpus (55 files) from its Zenodo deposit into
# data_swamp/. The corpus is published there, not in this repo, so that no synthetic
# identity-document images or account numbers are stored in git history.
#
# Deposit: https://zenodo.org/records/23551435  (DOI 10.5281/zenodo.23551435)
set -euo pipefail

ZENODO_RECORD_ID="23551435"
ARCHIVE_NAME="data_swamp.zip"
MD5="908d1be961239d26f3d184d9d1edc7de"
DEST_DIR="data_swamp"

URL="https://zenodo.org/api/records/${ZENODO_RECORD_ID}/files/${ARCHIVE_NAME}/content"

if [ -d "$DEST_DIR" ] && [ "$(ls -A "$DEST_DIR" 2>/dev/null)" ]; then
    echo "$DEST_DIR already exists and is non-empty — skipping download. Remove it to re-fetch."
    exit 0
fi

echo "Downloading $ARCHIVE_NAME from Zenodo record $ZENODO_RECORD_ID ..."
curl -L -o "$ARCHIVE_NAME" "$URL"

echo "Verifying MD5 checksum (Zenodo's reported checksum algorithm)..."
actual_md5=$(md5sum "$ARCHIVE_NAME" | awk '{print $1}')
if [ "$actual_md5" != "$MD5" ]; then
    echo "FATAL: checksum mismatch. expected $MD5, got $actual_md5" >&2
    rm -f "$ARCHIVE_NAME"
    exit 1
fi

echo "Checksum OK. Extracting into $DEST_DIR/ ..."
mkdir -p "$DEST_DIR"
unzip -q "$ARCHIVE_NAME" -d "$DEST_DIR"
rm -f "$ARCHIVE_NAME"

echo "Done. $(ls "$DEST_DIR" | wc -l) files in $DEST_DIR/."
echo "NOTE: all records are synthetic — no real personal information. See the Zenodo"
echo "deposit's README_DATASET.md for the full synthetic-data notice."
