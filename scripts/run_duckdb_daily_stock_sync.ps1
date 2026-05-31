param(
    [string]$ProjectRoot = "E:\openclaw",
    [string]$EnvFile = "",
    [string]$DuckDbPath = "",
    [string]$RowTable = "",
    [string]$IntradayTable = "",
    [string]$IntradayIntervals = "",
    [string]$IntradayRetentionDays = "",
    [switch]$SkipDownload,
    [switch]$SkipIntraday,
    [switch]$AllIntradayDates,
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
$LogFile = Join-Path $LogDir "duckdb_daily_stock_sync_$Stamp.log"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ArgsList = @(
    "scripts\sync_duckdb_daily_stock_data.py",
    "--env-file", $EnvFile
)

if (-not [string]::IsNullOrWhiteSpace($DuckDbPath)) {
    $ArgsList += @("--duckdb-path", $DuckDbPath)
}

if (-not [string]::IsNullOrWhiteSpace($RowTable)) {
    $ArgsList += @("--row-table", $RowTable)
}

if (-not [string]::IsNullOrWhiteSpace($IntradayTable)) {
    $ArgsList += @("--intraday-table", $IntradayTable)
}

if (-not [string]::IsNullOrWhiteSpace($IntradayIntervals)) {
    $ArgsList += @("--intraday-intervals", $IntradayIntervals)
}

if (-not [string]::IsNullOrWhiteSpace($IntradayRetentionDays)) {
    $ArgsList += @("--intraday-retention-days", $IntradayRetentionDays)
}

if ($SkipDownload) {
    $ArgsList += "--skip-download"
}

if ($SkipIntraday) {
    $ArgsList += "--skip-intraday"
}

if ($AllIntradayDates) {
    $ArgsList += "--all-intraday-dates"
}

if (-not [string]::IsNullOrWhiteSpace($InputDir)) {
    $ArgsList += @("--input-dir", $InputDir)
}

& $Python @ArgsList *> $LogFile
exit $LASTEXITCODE
