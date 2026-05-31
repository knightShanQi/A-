param(
    [string]$ProjectRoot = "E:\openclaw",
    [string]$TaskName = "OpenClaw DuckDB Daily Stock Sync",
    [string]$DailyAt = "17:20",
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

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $ProjectRoot ".env.local"
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Env file not found: $EnvFile. Create it from .env.example before installing the scheduled task."
}

$Runner = Join-Path $ProjectRoot "scripts\run_duckdb_daily_stock_sync.ps1"
$TaskArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Runner`"",
    "-ProjectRoot", "`"$ProjectRoot`"",
    "-EnvFile", "`"$EnvFile`""
)

if (-not [string]::IsNullOrWhiteSpace($DuckDbPath)) {
    $TaskArgs += @("-DuckDbPath", "`"$DuckDbPath`"")
}

if (-not [string]::IsNullOrWhiteSpace($RowTable)) {
    $TaskArgs += @("-RowTable", "`"$RowTable`"")
}

if (-not [string]::IsNullOrWhiteSpace($IntradayTable)) {
    $TaskArgs += @("-IntradayTable", "`"$IntradayTable`"")
}

if (-not [string]::IsNullOrWhiteSpace($IntradayIntervals)) {
    $TaskArgs += @("-IntradayIntervals", "`"$IntradayIntervals`"")
}

if (-not [string]::IsNullOrWhiteSpace($IntradayRetentionDays)) {
    $TaskArgs += @("-IntradayRetentionDays", "`"$IntradayRetentionDays`"")
}

if ($SkipDownload) {
    $TaskArgs += "-SkipDownload"
}

if ($SkipIntraday) {
    $TaskArgs += "-SkipIntraday"
}

if ($AllIntradayDates) {
    $TaskArgs += "-AllIntradayDates"
}

if (-not [string]::IsNullOrWhiteSpace($InputDir)) {
    $TaskArgs += @("-InputDir", "`"$InputDir`"")
}

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($TaskArgs -join " ")
$Trigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Download daily stock files and upsert them into the local OpenClaw DuckDB database." `
    -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' at $DailyAt."
