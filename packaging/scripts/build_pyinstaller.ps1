$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

Write-Host "Repo: $RepoRoot"
Write-Host "Running PyInstaller (onedir)..."

if (-not (Get-Command pyinstaller -ErrorAction SilentlyContinue)) {
    Write-Error "PyInstaller not found. Run: pip install -r requirements-build.txt"
    exit 1
}

pyinstaller -y --clean (Join-Path $RepoRoot "packaging\pyinstaller\betasafe.spec")

Write-Host "Done. Output under dist\"
if (Test-Path (Join-Path $RepoRoot "dist\BetaSafe\BetaSafe.exe")) {
    Write-Host "Windows exe: dist\BetaSafe\BetaSafe.exe"
}
