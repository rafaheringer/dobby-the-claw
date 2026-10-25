# run-windows.ps1 - Sobe tunel SSH (OpenClaw/HA no Raspberry Pi), daemon do Reachy e o bridge Dobby neste PC.
# Uso: powershell -ExecutionPolicy Bypass -File scripts\run-windows.ps1 
#   Painel web: http://localhost:18900 (e pelo celular via Cloudflare, veja docs/web-ui.md).
param([string]$Pi = "dobby@192.168.68.56")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $env:USERPROFILE "dobby-run"
New-Item -ItemType Directory -Force $logs | Out-Null

function Test-Port([int]$Port) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

# 1) Tunel: OpenClaw exige conexao vinda de localhost; 18800 e o webhook reverso de notificacoes.
if (-not (Test-Port 18789)) {
    Write-Host "Abrindo tunel SSH para $Pi ..."
    Start-Process ssh -WindowStyle Hidden -ArgumentList @(
        "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "BatchMode=yes",
        "-L", "18789:127.0.0.1:18789", "-L", "8123:127.0.0.1:8123", "-R", "18800:127.0.0.1:18800", "-R", "18900:127.0.0.1:18900", $Pi)
    Start-Sleep 3
    if (-not (Test-Port 18789)) { throw "Tunel SSH nao subiu." }
}

# 2) Daemon do Reachy (robo plugado neste PC).
function Start-Daemon {
    if (Test-Port 8000) { return }
    Write-Host "Iniciando daemon do Reachy ..."
    $daemon = Join-Path $env:USERPROFILE "reachy_mini_env\Scripts\reachy-mini-daemon.exe"
    Start-Process $daemon -WindowStyle Hidden -ArgumentList "--no-wake-up-on-start" `
        -RedirectStandardOutput (Join-Path $logs "daemon.log") -RedirectStandardError (Join-Path $logs "daemon.err.log")
    for ($i = 0; $i -lt 40 -and -not (Test-Port 8000); $i++) { Start-Sleep 1 }
    if (-not (Test-Port 8000)) { throw "Daemon do Reachy nao subiu (veja $logs\daemon.log)." }
}
Start-Daemon

# 3) Bridge: carrega .env e roda a partir de src/ (caminhos relativos ../.dobby e ../models).
Get-Content (Join-Path $root ".env") | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2].Trim() }
}
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $root "src")
$python = Join-Path $root ".venv\Scripts\python.exe"
Write-Host "Painel web: http://localhost:18900"

# Codigo 75: o bridge perdeu a conexao com o daemon e saiu de proposito; sobe tudo de novo.
while ($true) {
    & $python -m bridge.main
    if ($LASTEXITCODE -ne 75) { break }
    Write-Host "Conexao com o daemon perdida; reiniciando o bridge ..."
    Start-Sleep 3
    Start-Daemon
}
