#!/usr/bin/env bash
# 将 bpc_v4 同步到远程 GPU 服务器（不含密码，请自行 export 环境变量）
#
# 用法:
#   export BPC_V4_SYNC_HOST=183.232.132.248
#   export BPC_V4_SYNC_PORT=31407
#   export BPC_V4_SYNC_USER=user
#   export BPC_V4_SYNC_REMOTE=/home/user/pdl/mylab/quant/quant_cursor/bpc_v4
#   export BPC_V4_SYNC_PASSWORD='...'   # 或使用 SSH key，则无需密码
#   bash scripts/sync_bpc_v4_to_server.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DIR="${BPC_V4_SYNC_LOCAL:-$ROOT/src/bpc_v4}"
HOST="${BPC_V4_SYNC_HOST:?set BPC_V4_SYNC_HOST}"
PORT="${BPC_V4_SYNC_PORT:-22}"
USER="${BPC_V4_SYNC_USER:?set BPC_V4_SYNC_USER}"
REMOTE="${BPC_V4_SYNC_REMOTE:?set BPC_V4_SYNC_REMOTE}"

RSYNC_SSH=(ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
if [[ -n "${BPC_V4_SYNC_PASSWORD:-}" ]]; then
  if command -v sshpass >/dev/null 2>&1; then
    RSYNC_SSH=(sshpass -p "$BPC_V4_SYNC_PASSWORD" ssh -p "$PORT" -o StrictHostKeyChecking=accept-new)
  else
    echo "WARN: sshpass 未安装，请配置 SSH key 或: apt install sshpass" >&2
  fi
fi

echo "[sync] $LOCAL_DIR -> $USER@$HOST:$REMOTE"
rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  -e "${RSYNC_SSH[*]}" \
  "$LOCAL_DIR/" "$USER@$HOST:$REMOTE/"

echo "[sync] done"
