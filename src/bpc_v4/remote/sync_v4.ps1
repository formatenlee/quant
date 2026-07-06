# BPC-v4 代码同步脚本 (PowerShell for Windows)
# 方式1（推荐，无需 rsync）:
#   $env:QUANT_SSH_PASSWORD='your_password'
#   ..\.venv\Scripts\python.exe ..\..\scripts\sync_to_server.py --only bpc_v4
#
# 方式2（需 Git Bash / WSL 中的 rsync）:

$localDir = "E:\mylab\quant\src\bpc_v4\"
$remoteDir = "/home/user/pdl/mylab/quant/quant_cursor/bpc_v4/"
$sshTarget = "user@183.232.132.248"

Write-Host "同步 BPC-v4 到服务器 (端口 31407)..."
rsync -avz --delete `
  --exclude='__pycache__/' `
  --exclude='*.pyc' `
  --exclude='.ipynb_checkpoints/' `
  -e "ssh -p 31407" `
  $localDir "${sshTarget}:${remoteDir}"

Write-Host "同步完成。建议在服务器运行前激活 venv 并检查 Kronos 模型。"
