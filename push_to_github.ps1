# Publishes the repository in the "wavefusion-net-nigercon2026" folder next to this script.
# Run in PowerShell:  .\push_to_github.ps1
# Needs git and a GitHub login as praann07 (the one you already use).
#
# What it does:
#   1. clones https://github.com/praann07/wavefusion-net-nigercon2026 into a temporary folder
#   2. goes back to commit 250d271 (the last commit before the three that carry the
#      "Co-Authored-By: Claude" line; the v1.0 release points at an older commit and is not affected)
#   3. replaces all files with the contents of the wavefusion-net-nigercon2026 folder
#   4. shows you the list of changes, asks for "yes", then commits as you and force-pushes to main
#
# The weights in "release_assets" are already attached to the v1.0 release. They are a backup copy
# and are never committed to the repository (GitHub blocks files over 100 MB).

$ErrorActionPreference = "Stop"
$repoUrl = "https://github.com/praann07/wavefusion-net-nigercon2026.git"
$src = Join-Path $PSScriptRoot "wavefusion-net-nigercon2026"
$work = Join-Path $env:TEMP "wavefusion_publish"

if (-not (Test-Path $src)) { throw "Folder not found: $src" }
if (Test-Path $work) { Remove-Item $work -Recurse -Force }

git clone $repoUrl $work
Set-Location $work

git reset --hard 250d271
git rm -r -q --ignore-unmatch .
Get-ChildItem -Force $src | Copy-Item -Destination $work -Recurse -Force

git add -A
Write-Host ""
Write-Host "These changes will be committed:" -ForegroundColor Cyan
git status --short
Write-Host ""

$answer = Read-Host "Commit and force-push to main on GitHub? (type yes)"
if ($answer -ne "yes") { Write-Host "Stopped. Nothing was pushed."; exit 0 }

git commit -m "Add importable model, inference script and tests; update README, figures and citation"
git push --force-with-lease origin main

Write-Host ""
Write-Host "Done. Contributors GitHub now lists (the list can take a few minutes to refresh):" -ForegroundColor Green
try { gh api repos/praann07/wavefusion-net-nigercon2026/contributors -q ".[].login" } catch { Write-Host "(open the repository page to check)" }
