#!/usr/bin/env bash
# 将本地 src/ 同步到远程 quant/quant_cursor/（服务器上 src 即 quant_cursor 包）
#
# 用法:
#   export QUANT_SYNC_HOST=183.232.132.248
#   export QUANT_SYNC_PORT=31407
#   export QUANT_SYNC_USER=user
#   export QUANT_SYNC_REMOTE=/home/user/pdl/mylab/quant/quant_cursor
#   export QUANT_SYNC_PASSWORD='...'   # 可选；推荐 SSH key
#   bash scripts/sync_src_to_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="${QUANT_SYNC_LOCAL:-$ROOT/src}"
HOST="${QUANT_SYNC_HOST:?set QUANT_SYNC_HOST}"
PORT="${QUANT_SYNC_PORT:-22}"
USER="${QUANT_SYNC_USER:?set QUANT_SYNC_USER}"
REMOTE="${QUANT_SYNC_REMOTE:?set QUANT_SYNC_REMOTE}"

RSYNC_SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "${QUANT_SYNC_PASSWORD:-}" ]]; then
  if command -v sshpass >/dev/null 2>&1; then
    RSYNC_SSH=(sshpass -p "$QUANT_SYNC_PASSWORD" ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
  else
    echo "WARN: sshpass 未安装，请配置 SSH key 或: apt install sshpass" >&2
  fi
fi

echo "[sync] $LOCAL_DIR/ -> $USER@$HOST:$REMOTE/"
rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '*-Copy1.py' \
  --exclude 'quant_cursor/' \
  -e "${RSYNC_SSH[*]}" \
  "$LOCAL_DIR/" "$USER@$HOST:$REMOTE/"

echo "[sync] done"
