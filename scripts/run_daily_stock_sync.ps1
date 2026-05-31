param(
    [string]$ProjectRoot = "E:\openclaw",
    [string]$EnvFile = "",
    [string]$TableName = "",
    [string]$StorageMode = "",
    [switch]$SkipDownload,
    [string]$InputDir = ""
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $ProjectRoot

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $ProjectRoot ".env.local"
}

$LogDir = Join-Path $ProjectRoot ".cache\sync_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "daily_stock_sync_$Stamp.log"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ArgsList = @(
    "scripts\sync_daily_stock_data.py",
    "--env-file", $EnvFile
)

if (-not [string]::IsNullOrWhiteSpace($TableName)) {
    $ArgsList += @("--table-name", $TableName)
}

if (-not [string]::IsNullOrWhiteSpace($StorageMode)) {
    $ArgsList += @("--storage-mode", $StorageMode)
}

if ($SkipDownload) {
    $ArgsList += "--skip-download"
}

if (-not [string]::IsNullOrWhiteSpace($InputDir)) {
    $ArgsList += @("--input-dir", $InputDir)
}

& $Python @ArgsList *> $LogFile
exit $LASTEXITCODE
