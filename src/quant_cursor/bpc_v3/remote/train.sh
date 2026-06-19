#!/usr/bin/env bash
# BPC-v3 远程训练入口（在 SSH 主机上执行）
#
# 用法:
#   source ~/pdl/venv/bin/activate
#   cd /home/user/pdl/mylab/quant
#   export PYTHONPATH=/home/user/pdl/mylab/quant:$PYTHONPATH
#   bash quant_cursor/bpc_v3/remote/train.sh --dev --epochs 100
#
# 4×48GB：默认单卡 cuda:0；可用 CUDA_VISIBLE_DEVICES=0,1 后自行扩展多卡（当前 train 为单卡）

set -euo pipefail

PROJECT_ROOT="${BPC_V3_PROJECT_ROOT:-/home/user/pdl/mylab/quant}"
cd "$PROJECT_ROOT"

if [[ -f "$HOME/pdl/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/pdl/venv/bin/activate"
else
  echo "WARN: venv 未找到，请先 source ~/pdl/venv/bin/activate" >&2
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DEVICE="${BPC_V3_DEVICE:-cuda:0}"
BATCH="${BPC_V3_BATCH_SIZE:-4096}"

echo "[bpc_v3] PROJECT_ROOT=$PROJECT_ROOT"
echo "[bpc_v3] PYTHON=$(which python)"
echo "[bpc_v3] DEVICE=$DEVICE BATCH=$BATCH"
python -c "import torch; print('[bpc_v3] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

exec python -m quant_cursor.bpc_v3.train \
  --device "$DEVICE" \
  --batch-size "$BATCH" \
  "$@"
