param(
    [string]$ProjectRoot = "E:\openclaw",
    [string]$TaskName = "OpenClaw Daily Stock Sync",
    [string]$DailyAt = "17:00",
    [string]$EnvFile = "",
    [string]$TableName = "",
    [string]$StorageMode = "",
    [switch]$SkipDownload,
    [string]$InputDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($EnvFile)) {
    $EnvFile = Join-Path $ProjectRoot ".env.local"
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Env file not found: $EnvFile. Create it from .env.example before installing the scheduled task."
}

$Runner = Join-Path $ProjectRoot "scripts\run_daily_stock_sync.ps1"
$TaskArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Runner`"",
    "-ProjectRoot", "`"$ProjectRoot`"",
    "-EnvFile", "`"$EnvFile`""
)

if (-not [string]::IsNullOrWhiteSpace($TableName)) {
    $TaskArgs += @("-TableName", "`"$TableName`"")
}

if (-not [string]::IsNullOrWhiteSpace($StorageMode)) {
    $TaskArgs += @("-StorageMode", "`"$StorageMode`"")
}

if ($SkipDownload) {
    $TaskArgs += "-SkipDownload"
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
    -Description "Download daily stock files and upsert them into Supabase PostgreSQL." `
    -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' at $DailyAt."
