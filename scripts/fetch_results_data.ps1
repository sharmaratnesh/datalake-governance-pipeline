# Fetch the golden labels, per-document predictions, SQLite databases and OWL ontologies
# from version 2 of the Zenodo deposit, and place each file where the scripts expect it.
# PowerShell twin of fetch_results_data.sh - see that file for the layout.
#
# Version 2 of the deposit: https://zenodo.org/records/23580222 (DOI 10.5281/zenodo.23580222).
# For testing against a local copy: $env:ZENODO_FILES_URL = "C:\path\to\upload"
$ErrorActionPreference = "Stop"

$ZenodoV2RecordId = "23580222"
$Md5 = @{
    "golden_labels.zip" = "eb716192dc5f4eaef43a6c177f5f8b63"
    "predictions.zip"   = "848c013cbcacd5cec33b5641fb7e0a75"
    "databases.zip"     = "2533bbd72bfd9b6aae64f1ee86055080"
    "ontologies.zip"    = "ee1183a8864cb6cfd78a686268669db8"
}

$Local = $env:ZENODO_FILES_URL

$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid())
New-Item -ItemType Directory -Force $Tmp, "evaluation", "output", "cluster_results" | Out-Null
try {
    foreach ($f in $Md5.Keys) {
        $dest = Join-Path $Tmp $f
        if ($Local) { Copy-Item (Join-Path $Local $f) $dest }
        else { Invoke-WebRequest -Uri "https://zenodo.org/api/records/$ZenodoV2RecordId/files/$f/content" -OutFile $dest }
        $actual = (Get-FileHash $dest -Algorithm MD5).Hash.ToLower()
        if ($actual -ne $Md5[$f]) { Write-Error "FATAL: $f checksum mismatch: expected $($Md5[$f]), got $actual"; exit 1 }
        Expand-Archive -Path $dest -DestinationPath (Join-Path $Tmp "x") -Force
        Write-Host "${f}: checksum OK"
    }
    $x = Join-Path $Tmp "x"
    Copy-Item "$x\golden_labels\*" "evaluation\" -Force
    Copy-Item "$x\predictions\laptop\*" "output\" -Force
    Copy-Item "$x\predictions\cluster\*.csv" "output\" -Force
    Copy-Item "$x\predictions\cluster\*.jsonl" "cluster_results\" -Force
    Copy-Item "$x\databases\*" "output\" -Force
    Copy-Item "$x\ontologies\*" "output\" -Force
    Write-Host "Done. The three annotators' individual sheets are not published (see README)."
} finally {
    Remove-Item -Recurse -Force $Tmp
}
