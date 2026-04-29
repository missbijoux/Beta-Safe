param(
    [ValidateSet("strict", "balanced", "smooth", "aggressive")]
    [string]$Preset = "balanced",
    [string]$AdultOnnxPath = "",
    [string]$AdultHfModel = "",
    [switch]$DebugAdult
)

$ErrorActionPreference = "Stop"

# Always run from the repository root (directory containing this script).
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $repoRoot
if (-not (Test-Path -Path (Join-Path $repoRoot "src"))) {
    throw "Could not find 'src' under $repoRoot. Run this script from the Beta-Safe repo copy."
}

function Set-BetaSafePreset {
    param(
        [string]$Name
    )

    # Performance baseline
    $env:BETASAFE_TARGET_FPS = "12"
    $env:BETASAFE_DETECT_EVERY = "3"
    $env:BETASAFE_MAX_PROCESS_WIDTH = "720"

    # Keep weak proxy off by default: this tends to cause false positives.
    $env:BETASAFE_ADULT_SKIN_HEURISTIC = "0"
    $env:BETASAFE_ADULT_COMBINE = "any"

    switch ($Name) {
        "strict" {
            # Most conservative: fewer false positives, may miss borderline content.
            $env:BETASAFE_ADULT_ONNX_THRESHOLD = "0.94"
            $env:BETASAFE_ADULT_MIN_RECTS = "0"
            $env:BETASAFE_ADULT_MAX_ONNX_CROPS = "8"
            $env:BETASAFE_MAX_CENSOR_AREA = "0.70"
        }
        "balanced" {
            # Good default for day-to-day use.
            $env:BETASAFE_ADULT_ONNX_THRESHOLD = "0.92"
            $env:BETASAFE_ADULT_MIN_RECTS = "0"
            $env:BETASAFE_ADULT_MAX_ONNX_CROPS = "8"
            $env:BETASAFE_MAX_CENSOR_AREA = "0.75"
        }
        "aggressive" {
            # Catches more content, but can block more benign UI/images.
            $env:BETASAFE_ADULT_ONNX_THRESHOLD = "0.88"
            $env:BETASAFE_ADULT_MIN_RECTS = "4"
            $env:BETASAFE_ADULT_MAX_ONNX_CROPS = "12"
            $env:BETASAFE_ADULT_TILE = "96"
            $env:BETASAFE_MAX_CENSOR_AREA = "0.85"
        }
        "smooth" {
            # Stability-focused: less flicker/pop-in, more complete coverage.
            $env:BETASAFE_ADULT_ONNX_THRESHOLD = "0.92"
            $env:BETASAFE_DETECT_EVERY = "1"
            $env:BETASAFE_ADULT_MIN_RECTS = "2"
            $env:BETASAFE_ADULT_MAX_ONNX_CROPS = "12"
            $env:BETASAFE_ADULT_TILE = "96"
            $env:BETASAFE_MAX_CENSOR_AREA = "0.80"
        }
        default {
            throw "Unknown preset: $Name"
        }
    }
}

Write-Host "BetaSafe launcher preset: $Preset"
Set-BetaSafePreset -Name $Preset

if ($AdultOnnxPath -and $AdultHfModel) {
    throw "Choose only one model source: -AdultOnnxPath or -AdultHfModel."
}

if ($AdultOnnxPath) {
    $resolved = Resolve-Path -Path $AdultOnnxPath -ErrorAction SilentlyContinue
    if (-not $resolved) {
        throw "ONNX file not found: $AdultOnnxPath"
    }
    $env:BETASAFE_ADULT_ONNX_PATH = $resolved.Path
    Remove-Item Env:BETASAFE_ADULT_HF_MODEL -ErrorAction SilentlyContinue
}
elseif ($AdultHfModel) {
    $env:BETASAFE_ADULT_HF_MODEL = $AdultHfModel
    Remove-Item Env:BETASAFE_ADULT_ONNX_PATH -ErrorAction SilentlyContinue
}
else {
    Write-Warning "No adult model provided. Blocking may be broad/random without an adult model."
    Write-Warning "Use -AdultOnnxPath <path> or -AdultHfModel <model-id>."
}

if ($DebugAdult) {
    $env:BETASAFE_DEBUG_ADULT = "1"
    $env:BETASAFE_DEBUG_ADULT_WORKER = "1"
    Write-Host "Adult debug logs: ON"
}

Write-Host ""
Write-Host "Effective key settings:"
Write-Host "  BETASAFE_ADULT_ONNX_THRESHOLD=$($env:BETASAFE_ADULT_ONNX_THRESHOLD)"
Write-Host "  BETASAFE_ADULT_MIN_RECTS=$($env:BETASAFE_ADULT_MIN_RECTS)"
Write-Host "  BETASAFE_ADULT_MAX_ONNX_CROPS=$($env:BETASAFE_ADULT_MAX_ONNX_CROPS)"
Write-Host "  BETASAFE_MAX_CENSOR_AREA=$($env:BETASAFE_MAX_CENSOR_AREA)"
Write-Host "  BETASAFE_ADULT_SKIN_HEURISTIC=$($env:BETASAFE_ADULT_SKIN_HEURISTIC)"
if ($env:BETASAFE_ADULT_ONNX_PATH) {
    Write-Host "  BETASAFE_ADULT_ONNX_PATH=$($env:BETASAFE_ADULT_ONNX_PATH)"
}
if ($env:BETASAFE_ADULT_HF_MODEL) {
    Write-Host "  BETASAFE_ADULT_HF_MODEL=$($env:BETASAFE_ADULT_HF_MODEL)"
}
Write-Host ""

python -m src
