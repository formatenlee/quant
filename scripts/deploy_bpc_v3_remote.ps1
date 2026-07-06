# BPC-v3 远程测试主机（4× RTX A6000 48GB）
# 凭据见 scripts/remote_bpc_v3.env（已 gitignore，勿提交）

param(
    [switch]$DryRun,
    [switch]$Preflight,
    [switch]$DeployOnly
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$py = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$args = @("scripts/deploy_bpc_v3_remote.py")
if ($DryRun) { $args += "--dry-run" }
if ($Preflight) { $args += "--run-preflight" }

& $py @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $DryRun -and -not $DeployOnly) {
    Write-Host ""
    Write-Host "远程训练示例:"
    Write-Host "  ssh -p 31407 user@183.232.132.248"
    Write-Host "  source ~/pdl/venv/bin/activate"
    Write-Host "  cd /home/user/pdl/mylab/quant"
    Write-Host "  export PYTHONPATH=/home/user/pdl/mylab/quant:`$PYTHONPATH"
    Write-Host "  bash quant_cursor/bpc_v3/remote/train.sh --dev --epochs 100"
}
