$ErrorActionPreference = "Stop"

$ProjectRoot = "E:\openclaw"
$Runner = Join-Path $ProjectRoot "scripts\run_historical_backfill_2000_2021.ps1"

& $Runner
exit $LASTEXITCODE
