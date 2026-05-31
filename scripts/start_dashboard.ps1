$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$stdoutPath = Join-Path $projectRoot "streamlit.out.log"
$stderrPath = Join-Path $projectRoot "streamlit.err.log"
$port = 8501

if (-not (Test-Path $pythonPath)) {
    throw "未找到虚拟环境 Python：$pythonPath"
}

$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    try {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
    } catch {
        Write-Warning "停止旧的 $port 端口监听进程失败：$($listener.OwningProcess)"
    }
}

Start-Sleep -Seconds 2

if (Test-Path $stdoutPath) {
    Remove-Item -LiteralPath $stdoutPath -Force
}
if (Test-Path $stderrPath) {
    Remove-Item -LiteralPath $stderrPath -Force
}

try {
    Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @("scripts\run_streamlit_app.py") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
} catch [System.ArgumentException] {
    # Some Windows sessions expose both Path and PATH. Start-Process can fail
    # while cloning that environment, so fall back to ProcessStartInfo.
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "cmd.exe"
    $psi.Arguments = '/c ""{0}" "scripts\run_streamlit_app.py" 1>"{1}" 2>"{2}""' -f $pythonPath, $stdoutPath, $stderrPath
    $psi.WorkingDirectory = $projectRoot
    $psi.UseShellExecute = $false
    $process = [System.Diagnostics.Process]::Start($psi)
    if ($null -eq $process) {
        throw "Failed to start dashboard process via ProcessStartInfo."
    }
}

Start-Sleep -Seconds 6

try {
    $health = Invoke-WebRequest -Uri "http://localhost:$port/healthz" -UseBasicParsing -TimeoutSec 10
    Write-Output "Dashboard started: http://localhost:$port"
    Write-Output "Health status: $($health.StatusCode)"
} catch {
    Write-Warning "Dashboard 进程已启动，但健康检查未在预期时间内返回。可查看日志：$stderrPath"
}
