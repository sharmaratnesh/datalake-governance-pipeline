# Fetch the synthetic "data swamp" corpus (55 files) from its Zenodo deposit into
# data_swamp/. The corpus is published there, not in this repo, so that no synthetic
# identity-document images or account numbers are stored in git history.
#
# Deposit: https://zenodo.org/records/23551435  (DOI 10.5281/zenodo.23551435)
$ErrorActionPreference = "Stop"

$ZenodoRecordId = "23551435"
$ArchiveName = "data_swamp.zip"
$Md5 = "908d1be961239d26f3d184d9d1edc7de"
$DestDir = "data_swamp"

$Url = "https://zenodo.org/api/records/$ZenodoRecordId/files/$ArchiveName/content"

if ((Test-Path $DestDir) -and (Get-ChildItem $DestDir -ErrorAction SilentlyContinue)) {
    Write-Host "$DestDir already exists and is non-empty — skipping download. Remove it to re-fetch."
    exit 0
}

Write-Host "Downloading $ArchiveName from Zenodo record $ZenodoRecordId ..."
Invoke-WebRequest -Uri $Url -OutFile $ArchiveName

Write-Host "Verifying MD5 checksum (Zenodo's reported checksum algorithm)..."
$actualMd5 = (Get-FileHash -Path $ArchiveName -Algorithm MD5).Hash.ToLower()
if ($actualMd5 -ne $Md5.ToLower()) {
    Remove-Item -Force $ArchiveName
    Write-Error "FATAL: checksum mismatch. expected $Md5, got $actualMd5"
    exit 1
}

Write-Host "Checksum OK. Extracting into $DestDir/ ..."
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
Expand-Archive -Path $ArchiveName -DestinationPath $DestDir -Force
Remove-Item -Force $ArchiveName

$count = (Get-ChildItem $DestDir -File).Count
Write-Host "Done. $count files in $DestDir/."
Write-Host "NOTE: all records are synthetic — no real personal information. See the Zenodo"
Write-Host "deposit's README_DATASET.md for the full synthetic-data notice."
