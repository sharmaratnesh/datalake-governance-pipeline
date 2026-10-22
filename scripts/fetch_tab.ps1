# Fetch the Text Anonymization Benchmark (TAB) at the exact commit this paper's results
# were produced from. Idempotent: safe to re-run, does not re-clone if already at the
# right commit. See external/TAB_VERSION.txt for provenance details.
$ErrorActionPreference = "Stop"

$PinnedSha = "558e09e26d6b36f5f78440074e6a233946d98bd9"
$RepoUrl = "https://github.com/NorskRegnesentral/text-anonymization-benchmark"
$Dest = "external/tab"

Write-Host "=== TAB citation ==="
Write-Host "Pilan, I., Lison, P., Ovrelid, L., Papadopoulou, A., Sanchez, D., & Batet, M. (2022)."
Write-Host "The Text Anonymization Benchmark (TAB): A Dedicated Corpus and Evaluation Framework"
Write-Host "for Text Anonymization. Computational Linguistics."
Write-Host "Licence: MIT (see $RepoUrl/blob/master/LICENSE)"
Write-Host "============================================================================"

if (Test-Path "$Dest/.git") {
    $currentSha = (git -C $Dest rev-parse HEAD).Trim()
    if ($currentSha -eq $PinnedSha) {
        Write-Host "Already at pinned commit $PinnedSha — nothing to do."
        exit 0
    }
    Write-Host "Existing clone at $currentSha, not the pinned commit — checking out $PinnedSha."
    git -C $Dest fetch origin
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }
    git -C $Dest checkout $PinnedSha
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
} else {
    Write-Host "Cloning $RepoUrl into $Dest ..."
    git clone $RepoUrl $Dest
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    git -C $Dest checkout $PinnedSha
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
}

$actualSha = (git -C $Dest rev-parse HEAD).Trim()
if ($actualSha -ne $PinnedSha) {
    Write-Error "FATAL: expected commit $PinnedSha, landed on $actualSha instead."
    exit 1
}

Write-Host "Checked out TAB at $actualSha — matches external/TAB_VERSION.txt."
Write-Host "Done. Run external/tab_adapter.py next to build tab_corpus/ and the derived labels."
