# Builds the Windows standalone release.
#
# Usage (from the project root, any shell):
#   ./scripts/build_windows.ps1
#
# Produces:
#   build/DeltaHarmonicaScore/            (PyInstaller workdir, gitignored)
#   dist/DeltaHarmonicaScore/             (onedir app, gitignored)
#   release/DeltaHarmonicaScore-v<version>-windows-x64.zip

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No venv found at .venv. Create it and install requirements.txt + pyside6 + pyinstaller first."
}

$version = (& $python -c "from app.version import VERSION; print(VERSION)").Trim()
Write-Host "Building Delta Harmonica Score v$version"

# 1. run the test suite quickly (without the slow real-model test)
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Error "Tests failed; aborting the build." }

# 2. clean previous artifacts
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist, release
New-Item -ItemType Directory -Force -Path release | Out-Null

# 3. PyInstaller (onedir, windowed)
& (Join-Path $root ".venv\Scripts\pyinstaller.exe") DeltaHarmonicaScore.spec --noconfirm
if ($LASTEXITCODE -ne 0) { Write-Error "PyInstaller failed." }

# 4. sanity check the layout
$exe = Join-Path $root "dist\DeltaHarmonicaScore\DeltaHarmonicaScore.exe"
if (-not (Test-Path $exe)) { Write-Error "Expected EXE not found at $exe" }

# 5. zip the onedir folder for a GitHub release asset
$zip = Join-Path $root "release\DeltaHarmonicaScore-v$version-windows-x64.zip"
Compress-Archive -Path (Join-Path $root "dist\DeltaHarmonicaScore") -DestinationPath $zip -Force

$sizeGB = [math]::Round((Get-ChildItem -Recurse (Join-Path $root "dist\DeltaHarmonicaScore") | Measure-Object Length -Sum).Sum / 1GB, 2)
Write-Host ""
Write-Host "Release ready:"
Write-Host "  $zip ($sizeGB GB)"
