# 众包任务推荐 · 动态双边 DQN 实验

> 使用 AI 协作请先阅读 [agent.md](agent.md)。实验报告大纲见 [docs/report_outline.md](docs/report_outline.md)，模型过程记录见 [docs/experiment_process.md](docs/experiment_process.md)。

## 当前主线

本仓库现在以 **动态双边平台仿真** 为主实验：

- worker 到达时，Worker-DQN 从当前开放且未关闭的 project 中推荐 1 个任务。
- project 收到候选 worker 后，Requester-DQN 决定 `WAIT` 继续等待，或从申请池中选出 winner。
- project 选出 winner 后关闭；未中标 worker 会被释放并重新进入推荐队列。
- reward 同时记录 worker 收益、requester 收益、platform 总收益和 project 等候时间成本。
- **默认训练目标**为 utility 利益 proxy（`--reward-mode utility`）；`legacy` 模式以历史 hit 为主，用于对照。
- **Requester 默认批量决策**：申请池攒够 8 人或临近 deadline 才触发选人；`--immediate-requester-decision` 恢复「每进 1 人即决策」。

旧的独立 worker/requester 实验仍保留为 legacy 对照入口。

## 目录结构

```text
├── src/
│   ├── dataset.py              # 原始 Crowdspring 数据读取、时间划分、缓存
│   ├── platform_dataset.py     # 动态平台统一事件和 outcome 索引
│   └── features.py             # worker/project 特征
├── env/
│   ├── platform_env.py         # 动态双边联合仿真环境（主线）
│   ├── worker_env.py           # legacy: 独立参与者侧环境
│   └── requester_env.py        # legacy: 独立请求者侧环境
├── models/
│   ├── dqn.py                  # Vanilla / Dueling / Double DQN
│   ├── platform_training.py    # 双 agent 异步训练/评估循环
│   ├── platform_baselines.py   # 动态平台启发式基线
│   └── training_log.py
├── scripts/
│   ├── train_platform_dqn.py
│   ├── evaluate_platform.py
│   ├── run_platform_baselines.py
│   ├── pretrained_platform_bc.py
│   ├── analyze_pool_candidates.py
│   └── train_worker_dqn.py / train_requester_dqn.py / evaluate.py / run_baselines.py  # legacy
└── runs/
```

## 快速开始

```bash
pip install -r requirements.txt

# 数据检查
python -m src.dataset --max-projects 50

# 动态平台启发式基线 smoke
python scripts/run_platform_baselines.py --split train --max-projects 50 --max-steps 100

# 正式训练（默认：全量、完整 episode、utility、cuda）
python scripts/train_platform_dqn.py --device cuda

# smoke
python scripts/train_platform_dqn.py --max-projects 50 --episodes 2 --max-steps 100 --device cpu

# 申请池 / 候选规模诊断
python scripts/analyze_pool_candidates.py --split train --max-projects 50 --max-steps 200

# Platform BC 预训练
python scripts/pretrained_platform_bc.py --side worker --max-projects 50 --episodes 3

# 旧 hit 导向对照
python scripts/train_platform_dqn.py --reward-mode legacy --immediate-requester-decision --no-mixed-recall

# 动态平台 DQN test 评估
python scripts/evaluate_platform.py --split test --max-projects 0 \
  --worker-policy dqn \
  --requester-policy dqn \
  --worker-checkpoint runs/platform/.../checkpoints/worker_best.pt \
  --requester-checkpoint runs/platform/.../checkpoints/requester_best.pt
```

默认主实验不强制把真实标签注入候选集。如需诊断候选集排序上限，可加 `--include-truth-in-candidates`。

## 动态平台 MDP

**Worker-DQN**

| 要素 | 定义 |
|------|------|
| 状态 | 当前 worker 历史画像 + K 个 active project 动态特征（14 维，含 industry_match）+ mask |
| 动作 | 从候选 project 中推荐 1 个（混合召回：匹配/热门/低等待/随机） |
| reward | **utility 模式（默认）**：奖金、匹配、类目能力、竞争度；hit 仅弱信号。legacy 模式以历史命中为主 |

**Requester-DQN**

| 要素 | 定义 |
|------|------|
| 状态 | 当前 project 动态状态（17 维上下文，含申请池 quality 统计）+ 申请池 worker 特征 + mask |
| 动作 | `WAIT` 或选择 1 个 worker 为 winner |
| 触发 | 默认 batch：池≥8 / 距 deadline≤24h / 池满 / deadline 强制 |
| reward | **utility 模式**：quality、预期分数、匹配、活跃度；legacy 以 winner/finalist 为主 |

输出指标包括：

```text
worker_hit_rate, requester_hit_rate, worker_reward, requester_reward,
platform_reward, project_wait_cost, avg_project_wait_days,
filled_project_rate, winner_quality, rerouted_workers,
closed_projects, unfilled_projects, steps,
worker_recall_at_k, requester_recall_at_k,
platform_reward_per_project, platform_reward_per_step,
avg_requester_pool_size, avg_worker_utility, avg_requester_utility
```

> **注意**：`runs/report_full_20260529` 为改 batch / utility / 特征维度**之前**的正式实验，与当前默认配置不兼容，需 utility 模式全量重训后再填报告数字。

## Legacy 对照

以下入口保留用于复现旧实验，但不再作为报告主线：

```bash
python scripts/train_worker_dqn.py --max-projects 50 --episodes 10
python scripts/train_requester_dqn.py --max-projects 50 --episodes 10
python scripts/run_baselines.py --side worker --split test --max-projects 50
python scripts/run_baselines.py --side requester --split test --max-projects 50
```
