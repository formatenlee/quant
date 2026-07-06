#!/bin/bash
# BPC-v4 代码同步脚本 (Linux/macOS/WSL/Git Bash)
# 用法: ./sync_v4.sh
# 首次需 ssh-copy-id 或输入密码

LOCAL_DIR="E:/mylab/quant/src/bpc_v4/"
REMOTE_DIR="/home/user/pdl/mylab/quant/quant_cursor/bpc_v4/"
SSH_HOST="quant-bpc-v4"  # 或 user@183.232.132.248 -p 31407

echo "同步 BPC-v4 代码到服务器..."
rsync -avz --delete \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='kronos_model/' \
  -e "ssh -p 31407" \
  "$LOCAL_DIR" "user@183.232.132.248:$REMOTE_DIR"

echo "同步完成。服务器上运行: python -m quant_cursor.bpc_v4.train ..."
