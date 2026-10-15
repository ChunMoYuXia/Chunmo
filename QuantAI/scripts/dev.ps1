#requires -Version 7.0
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$apiPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$clientRoot = Join-Path $projectRoot 'apps\client'
$viteEntry = Join-Path $clientRoot 'node_modules\vite\bin\vite.js'
if (!(Test-Path -LiteralPath $apiPython) -or !(Test-Path -LiteralPath $viteEntry)) {
    throw '请先安装项目的 Python 和 npm 依赖，安装步骤见 README。'
}

$logRoot = Join-Path $projectRoot '.cache\dev'
$dataRoot = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $logRoot, $dataRoot -Force | Out-Null
$previousToken = $env:PORTFOLIO_SESSION_TOKEN
$env:PORTFOLIO_SESSION_TOKEN = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).ToLowerInvariant()
$env:PYTHONUTF8 = '1'
$backend = $null
$frontend = $null

try {
    $backend = Start-Process -FilePath $apiPython -ArgumentList @('-m', 'portfolio_client', '--port', '8787', '--data-dir', $dataRoot) -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'api.out.log') -RedirectStandardError (Join-Path $logRoot 'api.err.log')
    $frontend = Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList @($viteEntry, '--host', '127.0.0.1', '--port', '5173', '--strictPort') -WorkingDirectory $clientRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'client.out.log') -RedirectStandardError (Join-Path $logRoot 'client.err.log')
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($backend.HasExited -or $frontend.HasExited) { throw "启动失败，日志位于 $logRoot" }
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:8787/health' -TimeoutSec 1 | Out-Null
            Invoke-WebRequest -Uri 'http://127.0.0.1:5173' -TimeoutSec 1 | Out-Null
            $ready = $true
            break
        } catch { Start-Sleep -Milliseconds 250 }
    }
    if (!$ready) { throw "服务未能就绪，日志位于 $logRoot" }
    Write-Host '组合研究台已启动：http://127.0.0.1:5173；按 Ctrl+C 结束。'
    if (!$NoBrowser) {
        $launchUrl = 'http://127.0.0.1:5173/#session=' + $env:PORTFOLIO_SESSION_TOKEN + '&backend=http%3A%2F%2F127.0.0.1%3A8787'
        Start-Process -FilePath $launchUrl
    }
    while (!$backend.HasExited -and !$frontend.HasExited) { Start-Sleep -Milliseconds 500 }
} finally {
    foreach ($process in @($frontend, $backend)) {
        if ($null -ne $process -and !$process.HasExited) {
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
        }
    }
    $env:PORTFOLIO_SESSION_TOKEN = $previousToken
}
