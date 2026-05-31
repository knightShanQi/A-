param(
    [int]$StartYear = 2000,
    [int]$EndYear = 2026,
    [string]$ProjectRoot = "E:\openclaw",
    [string]$StorageMode = "",
    [string]$TableName = ""
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $ProjectRoot

$LogDir = Join-Path $ProjectRoot ".cache\sync_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

for ($Year = $EndYear; $Year -ge $StartYear; $Year--) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = Join-Path $LogDir "daily_stock_backfill_${Year}_$Stamp.log"
    Write-Host "Backfilling $Year -> $LogFile"
    $ArgsList = @("scripts\sync_daily_stock_data.py", "--backfill-years", "$Year")
    if (-not [string]::IsNullOrWhiteSpace($StorageMode)) {
        $ArgsList += @("--storage-mode", $StorageMode)
    }
    if (-not [string]::IsNullOrWhiteSpace($TableName)) {
        $ArgsList += @("--table-name", $TableName)
    }
    & $Python @ArgsList *> $LogFile
    if ($LASTEXITCODE -ne 0) {
        throw "Backfill failed for $Year. See $LogFile"
    }
}
