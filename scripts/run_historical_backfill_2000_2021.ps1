$ErrorActionPreference = "Stop"

$ProjectRoot = "E:\openclaw"
$Runner = Join-Path $ProjectRoot "scripts\backfill_daily_stock_history.ps1"

& $Runner -ProjectRoot $ProjectRoot -StartYear 2000 -EndYear 2021 -StorageMode "series" -TableName "a_share_daily_price_series"
exit $LASTEXITCODE
