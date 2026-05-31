# 全量正式流程：BC → DQN（PowerShell）
$ErrorActionPreference = "Stop"
$Device = if ($env:DEVICE) { $env:DEVICE } else { "cuda" }
$Log = if ($env:LOG_DIR) { $env:LOG_DIR } else { "runs/report_full" }

python scripts/pretrained_platform_bc.py --side worker --device $Device `
  --max-projects 0 --max-steps 0 --episodes 8 --log-dir "$Log/bc_platform"
python scripts/pretrained_platform_bc.py --side requester --device $Device `
  --max-projects 0 --max-steps 0 --episodes 8 --log-dir "$Log/bc_platform"

$BcW = Get-ChildItem -Path "$Log/bc_platform" -Recurse -Filter "worker_best.pt" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$BcR = Get-ChildItem -Path "$Log/bc_platform" -Recurse -Filter "requester_best.pt" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1

python scripts/train_platform_dqn.py --device $Device `
  --max-projects 0 --max-steps 0 --episodes 30 `
  --worker-pretrained $BcW.FullName --requester-pretrained $BcR.FullName `
  --log-dir "$Log/platform_dqn"

Write-Host "Done. Checkpoints under $Log/platform_dqn"
