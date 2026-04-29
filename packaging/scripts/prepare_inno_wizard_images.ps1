$ErrorActionPreference = "Stop"
# Resize PNG wizard art to BMP sizes expected by classic Inno Setup wizard images.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$InstallerAssets = Join-Path $RepoRoot "packaging\assets\installer"

Add-Type -AssemblyName System.Drawing

function Export-InnoWizardBmp {
    param(
        [string]$PngPath,
        [string]$BmpPath,
        [int]$Width,
        [int]$Height
    )
    if (-not (Test-Path $PngPath)) { return $false }
    $src = [System.Drawing.Image]::FromFile($PngPath)
    try {
        $bmp = New-Object System.Drawing.Bitmap $Width, $Height
        $g = [System.Drawing.Graphics]::FromImage($bmp)
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $g.DrawImage($src, 0, 0, $Width, $Height)
        $g.Dispose()
        $bmp.Save($BmpPath, [System.Drawing.Imaging.ImageFormat]::Bmp)
        $bmp.Dispose()
    }
    finally {
        $src.Dispose()
    }
    return $true
}

$largePng = Join-Path $InstallerAssets "wizard-large.png"
$smallPng = Join-Path $InstallerAssets "wizard-small.png"
$largeBmp = Join-Path $InstallerAssets "wizard-large.bmp"
$smallBmp = Join-Path $InstallerAssets "wizard-small.bmp"

$n = 0
if (Test-Path $largePng) {
    Export-InnoWizardBmp $largePng $largeBmp 164 314 | Out-Null
    Write-Host "Wrote $largeBmp"
    $n++
}
if (Test-Path $smallPng) {
    Export-InnoWizardBmp $smallPng $smallBmp 55 58 | Out-Null
    Write-Host "Wrote $smallBmp"
    $n++
}
if ($n -eq 0) {
    Write-Host "No wizard-large.png / wizard-small.png found; Inno will use default wizard art."
}
