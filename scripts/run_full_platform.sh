#!/usr/bin/env bash
# 全量正式流程：BC 预训练 → DQN 训练（默认 max_steps=0 / utility / lookahead）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cuda}"
LOG="${LOG_DIR:-runs/report_full}"

python scripts/pretrained_platform_bc.py --side worker --device "$DEVICE" \
  --max-projects 0 --max-steps 0 --episodes 8 --log-dir "$LOG/bc_platform"
python scripts/pretrained_platform_bc.py --side requester --device "$DEVICE" \
  --max-projects 0 --max-steps 0 --episodes 8 --log-dir "$LOG/bc_platform"

BC_W=$(ls -td "$LOG/bc_platform"/bc_platform_worker_*/checkpoints/worker_best.pt 2>/dev/null | head -1)
BC_R=$(ls -td "$LOG/bc_platform"/bc_platform_requester_*/checkpoints/requester_best.pt 2>/dev/null | head -1)

python scripts/train_platform_dqn.py --device "$DEVICE" \
  --max-projects 0 --max-steps 0 --episodes 30 \
  --worker-pretrained "$BC_W" --requester-pretrained "$BC_R" \
  --log-dir "$LOG/platform_dqn"

echo "Done. Run baselines with checkpoints from $LOG/platform_dqn"
