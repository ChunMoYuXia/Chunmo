#requires -Version 7.0
param([switch]$PortableOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$localCargo = Join-Path $projectRoot '.cache\rust-toolchain\bin\cargo.exe'
if (Test-Path -LiteralPath $localCargo) {
    $env:PATH = (Split-Path -Parent $localCargo) + ';' + $env:PATH
    $env:CARGO_HOME = Join-Path $projectRoot '.cache\cargo'
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$hasMsvc = (Test-Path -LiteralPath $vswhere) -and [bool](& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if (!$PortableOnly -and (!(Get-Command cargo.exe -ErrorAction SilentlyContinue) -or !$hasMsvc)) {
    throw '原生安装包需要 Rust MSVC 工具链和 Visual Studio C++ Build Tools/Windows SDK。可先使用 -PortableOnly 构建已验证的浏览器客户端。'
}
function Invoke-Build([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "构建失败，退出码 $LASTEXITCODE" }
}
Push-Location $projectRoot
try {
    $env:PYTHONUTF8 = '1'
    Invoke-Build { npm.cmd --prefix apps/client run build }
    Invoke-Build { & $python scripts/build_sidecar.py --tauri --portable }
    $portable = Join-Path $projectRoot 'dist\stock-portfolio-client'
    New-Item -ItemType Directory -Force -Path $portable | Out-Null
    $launcher = '@echo off' + "`r`n" + '"%~dp0portfolio-api\portfolio-api.exe" --port 0 --open-browser --data-dir "%LOCALAPPDATA%\local.portfolio.research"' + "`r`n"
    Set-Content -LiteralPath (Join-Path $portable 'Start-Portfolio.cmd') -Value $launcher -Encoding ascii
    Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md') -Destination (Join-Path $portable 'README.md') -Force
    Compress-Archive -LiteralPath $portable -DestinationPath (Join-Path $projectRoot 'dist\stock-portfolio-client-portable.zip') -Force
    if (!$PortableOnly) {
        Invoke-Build { npm.cmd --prefix apps/client run tauri -- build }
        $nativeConfig = Get-Content -LiteralPath (Join-Path $projectRoot 'apps/client/src-tauri/tauri.conf.json') -Raw | ConvertFrom-Json
        $nativeName = "$($nativeConfig.productName)_$($nativeConfig.version)_x64-setup.exe"
        $nativeOutput = Join-Path $projectRoot "apps/client/src-tauri/target/release/bundle/nsis/$nativeName"
        Copy-Item -LiteralPath $nativeOutput -Destination (Join-Path $projectRoot "dist/stock-portfolio-client_$($nativeConfig.version)_x64-setup.exe") -Force
    }
    Write-Host "本地产物：$portable"
} finally { Pop-Location }
