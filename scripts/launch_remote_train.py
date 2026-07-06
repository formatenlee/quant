#!/usr/bin/env python3
"""在远程主机后台启动 BPC-v3 训练（避免 shell 转义问题）。"""

from __future__ import annotations

import sys
from pathlib import Path

import paramiko

ENV_FILE = Path(__file__).resolve().parent / "remote_bpc_v3.env"

# RUN_DIR：checkpoint / train.log / metrics 统一目录；nohup 额外写 stdout 副本
REMOTE_SCRIPT = r"""
set -euo pipefail
cd /home/user/pdl/mylab/quant
source ~/pdl/venv/bin/activate
export PYTHONPATH=/home/user/pdl/mylab/quant:$PYTHONPATH

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${BPC_V3_RUN_DIR:-logs/bpc_v3/run_${STAMP}}"
mkdir -p "$RUN_DIR"
NOHUP_LOG="$RUN_DIR/nohup.log"
PID_FILE="$RUN_DIR/train.pid"

nohup python -m quant_cursor.bpc_v3.train \
  --run-dir "$RUN_DIR" \
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
  --prefetch-factor 12 \
  --purity-weight 0.35 \
  --extended-purity-weight 0.10 \
  --diversity-weight 0.20 \
  --recon-weight 0.6 \
  --commitment-cost 0.9 \
  --vq-dead-code-threshold 0 \
  --num-symbols 1000 \
  --labeling-mode per_stock \
  --save-every 100 \
  --start 2019-01-01 \
  --end 2026-12-31 \
  --max-instruments 250 \
  --num-coarse 128 \
  --amp \
  >>"$NOHUP_LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "STARTED PID=$(cat "$PID_FILE")"
echo "RUN_DIR=$RUN_DIR"
echo "TRAIN_LOG=$RUN_DIR/train.log"
echo "NOHUP_LOG=$NOHUP_LOG"
sleep 15
tail -n 35 "$RUN_DIR/train.log" 2>/dev/null || tail -n 35 "$NOHUP_LOG" || true
"""


def main() -> int:
    cfg: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=cfg["REMOTE_HOST"],
        port=int(cfg.get("REMOTE_PORT", "22")),
        username=cfg["REMOTE_USER"],
        password=cfg["REMOTE_PASSWORD"],
        timeout=30,
    )
    try:
        stdin, stdout, stderr = client.exec_command(f"bash -s <<'EOF'\n{REMOTE_SCRIPT}\nEOF", timeout=180)
        del stdin
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(out)
        if err.strip():
            print(err, file=sys.stderr)
        code = stdout.channel.recv_exit_status()
        return code
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
