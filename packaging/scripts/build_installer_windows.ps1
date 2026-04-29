$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

& (Join-Path $PSScriptRoot "prepare_inno_wizard_images.ps1")

$distExe = Join-Path $RepoRoot "dist\BetaSafe\BetaSafe.exe"
if (-not (Test-Path $distExe)) {
    Write-Error "Missing $distExe — run packaging\scripts\build_pyinstaller.ps1 first."
    exit 1
}

$iscc = $null
foreach ($c in @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )) {
    if (Test-Path $c) { $iscc = $c; break }
}
if (-not $iscc) {
    Write-Error "ISCC.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isdl.php and re-run."
    exit 1
}

$iss = Join-Path $RepoRoot "packaging\installer\BetaSafe.iss"
Write-Host "Using $iscc $iss"
& $iscc $iss
Write-Host "Installer output: packaging\out\installer\ (see BetaSafe.iss OutputDir)"
