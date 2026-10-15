#requires -Version 7.0
param([switch]$SkipE2E)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
function Invoke-Check([scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "检查失败，退出码 $LASTEXITCODE" }
}
Push-Location $projectRoot
try {
    Invoke-Check { & $python -m pytest apps/api/tests -q --tb=short -W error }
    Invoke-Check { & $python -m ruff check apps/api scripts }
    Invoke-Check { & $python -m mypy --config-file apps/api/pyproject.toml apps/api/src }
    Invoke-Check { & $python scripts/generate_contracts.py --check }
    Invoke-Check { npm.cmd --prefix apps/client run check }
    Invoke-Check { npm.cmd --prefix apps/client run lint }
    Invoke-Check { npm.cmd --prefix apps/client run test -- --run }
    Invoke-Check { npm.cmd --prefix apps/client run build }
    if (!$SkipE2E) { Invoke-Check { npm.cmd --prefix apps/client run test:e2e } }
} finally { Pop-Location }
