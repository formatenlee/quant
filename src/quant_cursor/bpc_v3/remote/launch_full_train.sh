#!/usr/bin/env bash
# 远程全量训练（与本地约定参数一致）
set -euo pipefail

PROJECT_ROOT="${BPC_V3_PROJECT_ROOT:-/home/user/pdl/mylab/quant}"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
source ~/pdl/venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

LOG_DIR="${PROJECT_ROOT}/logs/bpc_v3"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${STAMP}.log"

# 已有兼容缓存时不要加 --force-rebuild-preprocessed（磁盘紧张）
EXTRA_FLAGS=()

echo "Logging to ${LOG_FILE}"
nohup python -m quant_cursor.bpc_v3.train \
  --save-preprocessed ./data/bpc_v3_preprocessed \
  --epochs 1000 \
  --device cuda \
  --seed 42 \
  --val-ratio 0.1 \
  --log-every 25 \
  --val-every 100 \
  --day-lookback 40 \
  --week-lookback 24 \
  --batch-size 1024 \
  --num-workers 16 \
  --prefetch-factor 256 \
  --purity-weight 0.35 \
  --extended-purity-weight 0.15 \
  --diversity-weight 0.20 \
  --commitment-cost 0.4 \
  --vq-dead-code-threshold 0.01 \
  --num-symbols 250 \
  --labeling-mode per_stock \
  --save-every 1000 \
  --start 2019-01-01 \
  --end 2026-12-31 \
  --max-instruments 250 \
  --num-coarse 128 \
  "${EXTRA_FLAGS[@]}" \
  >>"$LOG_FILE" 2>&1 &

echo $! > "${LOG_DIR}/train_${STAMP}.pid"
echo "PID=$(cat "${LOG_DIR}/train_${STAMP}.pid")"
echo "tail -f ${LOG_FILE}"
