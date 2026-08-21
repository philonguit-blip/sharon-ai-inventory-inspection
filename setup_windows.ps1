param(
    [switch]$SkipFoundation
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$VenvPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$EnvExample = Join-Path $BackendRoot ".env.example"
$EnvFile = Join-Path $BackendRoot ".env"

Write-Host "[1/7] Checking Python 3.12..." -ForegroundColor Cyan
$BootstrapPython = $null
if (Test-Path -LiteralPath $VenvPython) {
    $BootstrapPython = $VenvPython
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $BootstrapPython = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BootstrapPython = (Get-Command python).Source
}
else {
    throw "Python 3.12 x64 was not found. Install it or provide backend/.venv first."
}
$Version = & $BootstrapPython -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($Version -ne "3.12") {
    throw "Python 3.12 x64 is required. Detected: $Version"
}

Write-Host "[2/7] Creating backend virtual environment..." -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $BootstrapPython -m venv (Join-Path $BackendRoot ".venv")
}

Write-Host "[3/7] Installing dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $BackendRoot "requirements-dev.txt")

Write-Host "[4/7] Preparing local environment file..." -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $EnvFile)) {
    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
    Write-Warning "Created backend/.env. Fill the real R2, KiotViet and n8n credentials before starting."
}

if (-not $SkipFoundation) {
    Write-Host "[5/7] Provisioning Foundation assets..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $ProjectRoot "scripts\provision_foundation.py")
} else {
    Write-Warning "Foundation provisioning skipped; AUTO will still use YOLO safely."
}

Write-Host "[6/7] Verifying production artifacts..." -ForegroundColor Cyan
if ($SkipFoundation) {
    & $VenvPython (Join-Path $ProjectRoot "scripts\verify_production.py")
} else {
    & $VenvPython (Join-Path $ProjectRoot "scripts\verify_production.py") --require-foundation
}

Write-Host "[7/7] Running automated tests..." -ForegroundColor Cyan
Push-Location $BackendRoot
try {
    & $VenvPython -m pytest tests -q
}
finally {
    Pop-Location
}

if (Get-Command node -ErrorAction SilentlyContinue) {
    & node (Join-Path $ProjectRoot "n8n\test_workflow4_outbound.mjs")
} else {
    Write-Warning "Node.js was not found; skipped the n8n workflow simulation."
}

Write-Host "Setup completed. Review backend/.env, then run start_backend.bat and start_worker.bat." -ForegroundColor Green
