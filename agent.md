# Agent 协作指南（强化学习大作业 · 众包任务推荐）

> **读者**：Cursor / 其它 AI Agent、使用 AI 改代码的同学。  
> **用途**：统一项目目标、实现现状、修改边界与协作方式。人类用户请同时参考 `README.md`、`docs/report_outline.md` 与 `docs/experiment_process.md`。

---

## 1. 项目目标（不可偏离）

### 1.1 课程作业要求

| 项 | 内容 |
|----|------|
| 场景 | Crowdspring 众包平台历史日志；**每次只推荐 1 个对象** |
| 问题 1 | 强化学习做**任务推荐**，最大化**参与者（worker）**利益 |
| 问题 2 | 强化学习做推荐，最大化**请求者（project 发布方）**利益 |
| 方法 | **必须使用 DQN 系列**（Vanilla / Double / Dueling） |
| 数据 | `data/data/`；自行划分 train/val/test；可参考 `sample_read_data.py` |
| 交付 | 实验报告（流程、设计、结果）+ 分组汇报；截止以课程通知为准 |
| 模型过程记录 | 必须记录模型实验中遇到的困难，特别是每次模型/实验调整的目的、具体改动、指标变化，以及可能原因 |

### 1.2 本仓库的实现取向

- **离线强化学习**：用历史 `(state, action, reward, next_state)` 事件流仿真，非在线 API 环境。
- **动态双边平台主线**：Worker-DQN 推荐 project，Requester-DQN 在同一平台状态中选择 `WAIT` 或 winner；旧双端独立 MDP 仅作 legacy 对照。
- **工程目标**：可复现训练 → 评估 → 基线对比 → 填报告表格。
- **模型过程可追溯**：与模型效果相关的困难、调整目的、调整内容、指标变化与原因分析统一写入 `docs/experiment_process.md`；环境配置、依赖安装、路径、缓存、日志目录等工程问题不写入该文件。

---

## 2. 仓库结构（修改前先读）

```
强化学习/
├── agent.md                 # 本文件（AI 协作入口）
├── README.md                # 人类快速上手
├── configs/default.yaml     # 数据路径、划分比例、DQN 默认超参
├── data/data/               # 原始数据（勿改内容；只读）
│   ├── project_list.csv
│   ├── worker_quality.csv
│   ├── project/project_{id}.txt
│   └── entry/entry_{id}_{offset}.txt
├── src/
│   ├── config.py            # 加载 YAML
│   ├── dataset.py           # 数据加载、划分、事件流、cache v2
│   ├── platform_dataset.py  # 动态平台统一事件/outcome 索引
│   └── features.py          # Worker(12) / Legacy Project(13) / Platform 专用维度
├── env/
│   ├── platform_env.py      # 动态双边平台 MDP（主线）
│   ├── worker_env.py        # legacy 参与者 MDP
│   └── requester_env.py     # legacy 请求者 MDP
├── models/
│   ├── dqn.py               # Q 网络、DQNAgent、checkpoint
│   ├── baselines.py         # legacy 基线策略
│   ├── platform_baselines.py # 动态平台基线
│   ├── platform_training.py # 双 agent 异步训练/评估
│   ├── eval_utils.py        # 评估循环
│   ├── eval_runner.py       # evaluate_one（脚本共用）
│   ├── train_utils.py       # 训练 episode 循环
│   └── training_log.py      # CSV/JSON 日志
├── scripts/
│   ├── train_platform_dqn.py
│   ├── evaluate_platform.py
│   ├── run_platform_baselines.py
│   ├── pretrained_platform_bc.py # Platform 侧 BC 预训练（utility/legacy 标签）
│   ├── analyze_pool_candidates.py # 申请池规模 & worker 候选数分布诊断
│   ├── train_worker_dqn.py      # legacy
│   ├── train_requester_dqn.py   # legacy
│   ├── evaluate.py          # 单策略评估
│   ├── run_baselines.py     # 批量基线 + 可选 DQN
│   ├── smoke_env.py
│   └── smoke_requester.py
├── docs/report_outline.md   # 实验报告大纲（填结果用）
├── docs/experiment_process.md # 模型实验过程、困难、调整与原因记录
├── cache/                   # dataset_*.pkl（自动生成，可删后重建）
└── runs/                    # 训练/评估输出（勿提交超大 checkpoint 除非课程要求）
```

### 2.1 关键数据约定

- Entry JSON 中 worker 字段名为 **`author`**（不是 `worker`）；`dataset.py` 已处理。
- `project_list.csv`：`project_id, entry_count`；entry 文件分页偏移 0,24,48,...
- 划分：**按项目 `start_date` 排序** 后 70% / 15% / 15% → train/val/test（在 `dataset.py`）。
- 过滤：`start_date >= 2018-01-01`（与 `sample_read_data.py` 一致）。

### 2.2 观测张量约定（DQN 输入）

`env.worker_env.Observation` 被两侧复用，**字段名不变、语义不同**：

| 字段 | 参与者侧 | 请求者侧 |
|------|----------|----------|
| `worker_feat` | worker 特征 (12) | **项目上下文** (13) |
| `candidate_feat` | K 个项目特征 (K×13) | K 个 worker 特征 (K×12) |
| `action_mask` | 合法候选槽位 | 同上 |

动态平台中 Worker-DQN 使用 `anchor_dim=12, candidate_dim=14`（`PLATFORM_PROJECT_FEAT_DIM`，含 `industry_match`）；Requester-DQN 使用 `anchor_dim=17, candidate_dim=12`（`REQUESTER_CONTEXT_FEAT_DIM`，含申请池 quality 统计 4 维），且动作 0 表示 `WAIT`。Legacy 环境仍为 13 维 project 特征，与 platform 维度分离。

---

## 3. 已实现 vs 未完成

### 3.1 已完成（勿重复造轮子）

- [x] 数据管道 `CrowdsourcingDataset` + pickle 缓存 v2
- [x] `iter_worker_events` / 活跃项目查询
- [x] 参与者 MDP `WorkerRecommendationEnv`
- [x] 请求者 MDP `RequesterRecommendationEnv`
- [x] DQN / Double / Dueling + Replay + target 网络
- [x] 训练日志 `metrics.csv`、`config.json`、checkpoint（best/ep/final）
- [x] 基线：`random`, `popularity`, `category_match`, `award`（worker）；`worker_quality`, `worker_activity`（requester）
- [x] `evaluate.py`、`run_baselines.py`
- [x] 动态平台主线：`PlatformDataset`、`PlatformSimulationEnv`、`train_platform_dqn.py`、`evaluate_platform.py`、`run_platform_baselines.py`
- [x] 报告大纲 `docs/report_outline.md`
- [x] `include_truth_in_candidates` 消融（train/eval/baseline 已支持 CLI 开关）
- [x] 学习曲线出图脚本（可从 `metrics.csv` 绘制）
- [x] BC预训练 `scripts/pretrained_bc.py`（legacy）
- [x] Platform BC 预训练 `scripts/pretrained_platform_bc.py`
- [x] Worker 混合召回（match / 热门 / 低等待 / 随机，`--no-mixed-recall` 可关）
- [x] Requester 批量/延迟决策（默认 `batch_size=8`，`--immediate-requester-decision` 恢复旧行为）
- [x] Platform 特征增强：`industry_match`、申请池 quality 统计（mean/max/std/top_gap）
- [x] **Utility reward 模式**（默认 `--reward-mode utility`；`legacy` 为 hit 导向对照）
- [x] 归一化/诊断指标：`platform_reward_per_project`、`platform_reward_per_step`、`worker/requester_recall_at_k`、`avg_requester_pool_size`、`avg_worker/requester_utility`
- [x] 训练超参优化（2026-05-31）：全量默认、`worker/requester` 分离 batch/buffer/ε 衰减、checkpoint 指标补 requester hit

### 3.2 未完成（优先任务）

- [ ] **Utility 模式**下全量重训 + test 基线（旧 checkpoint 与 `report_full_20260529` 不兼容）
- [ ] 三种 DQN 变体系统对比并填入报告表
- [ ] **实验报告正文**（PDF/Word）与 **PPT**
- [ ] 持续维护 `docs/experiment_process.md`，只记录模型相关困难、调整目的、具体改变、指标变化与可能原因
- [ ] 数据分析 EDA 图表写入报告 §2
- [ ] 可选：TensorBoard、GPU 默认配置

---

## 4. AI Agent 行为标准

### 4.1 通用原则

1. **先读后改**：修改前阅读 `agent.md` → 相关模块 → 调用方脚本；不要假设 API。
2. **小步提交**：单次 PR/任务只解决一个明确问题（如一侧环境、一脚本、一表）。
3. **可运行验证**：改完后至少运行相关 smoke 或 `evaluate`；全量训练由用户触发。
4. **不破坏数据**：不修改 `data/data/` 内原始文件；不提交 `cache/`、`runs/` 大文件除非用户要求。
5. **记录模型实验过程**：每次训练、评估、消融或影响模型结果解释的调整后，更新 `docs/experiment_process.md`。至少写清模型相关困难/现象、调整目的、具体调整、指标变化、可能原因、后续动作；不要记录环境配置、依赖安装、路径、缓存、日志目录等工程问题。
6. **中文注释适度**：公开 API 与复杂逻辑用简短中文/英文均可；避免冗长注释。
7. **匹配现有风格**：dataclass 配置、`build_dataset()`、`Observation.to_dict()` 等沿用现有模式。

### 4.2 禁止事项

- 勿删除或重命名 `runs/` 下用户实验结果（除非用户明确要求清理）。
- 勿将 `test` 集用于调参或 early stopping（仅最终报告）。
- 勿在 `_build_candidates` 中用「随机填充至 K」死循环逻辑（历史 bug，已修复）。
- 勿用 `worker` 字段读 entry（应用 `author`）。
- 勿引入与作业无关的大型依赖（如 Ray、完整 RLlib）除非用户明确要求。
- 勿擅自写长篇 `.md`（除 `agent.md`、`report_outline` 和用户点名要的文档）。

### 4.3 修改范围指南

| 你想做的事 | 应改文件 | 避免改 |
|------------|----------|--------|
| 特征工程 | `src/features.py` | 直接改 JSON 数据 |
| 参与者奖励/候选 | `env/worker_env.py` | `models/dqn.py` 网络结构（除非必要） |
| 请求者奖励/候选 | `env/requester_env.py` | `worker_env.py` |
| 网络/算法 | `models/dqn.py` | 环境 step 逻辑混杂进 Agent |
| 训练流程 | `scripts/train_*.py`, `models/train_utils.py` | 复制粘贴成第三套训练脚本 |
| 评估/基线 | `models/baselines.py`, `models/eval_runner.py`, `scripts/evaluate.py` | 重写 dataset |
| 超参默认值 | `configs/default.yaml` + 脚本 argparse | 硬编码在多处 |

### 4.4 测试命令（改代码后）

```bash
# 最快：动态平台启发式 smoke
python scripts/run_platform_baselines.py --split train --max-projects 50 --max-steps 100

# 数据
python -m src.dataset --max-projects 50

# 正式训练（默认全量 + utility + 分离超参）
python scripts/pretrained_platform_bc.py --side worker  --max-projects 0 --episodes 5 --max-steps 0 --device cuda
python scripts/pretrained_platform_bc.py --side requester --max-projects 0 --episodes 5 --max-steps 0 --device cuda
python scripts/train_platform_dqn.py --device cuda \
  --worker-pretrained runs/bc_platform/.../worker_best.pt \
  --requester-pretrained runs/bc_platform/.../requester_best.pt

# smoke（须显式缩数据/步数）
python scripts/train_platform_dqn.py --max-projects 50 --episodes 2 --max-steps 100 --device cpu

# 申请池 / 候选规模诊断
python scripts/analyze_pool_candidates.py --split train --max-projects 50 --max-steps 200

# Platform BC → DQN（utility 标签为候选内 argmax U）
python scripts/pretrained_platform_bc.py --side worker --max-projects 50 --episodes 3
python scripts/train_platform_dqn.py --max-projects 50 --episodes 5 --worker-pretrained runs/bc/.../best.pt

# 复现旧 hit 导向实验
python scripts/train_platform_dqn.py --reward-mode legacy --no-mixed-recall --immediate-requester-decision

# legacy 评估
python scripts/run_baselines.py --side worker --split test --max-projects 50

# legacy 消融实验示例
python scripts/run_baselines.py \
    --side worker \
    --split test \
    --max-projects 50 \
    --no-truth-in-candidates

# 学习曲线单个 run 示例
python scripts/plot_learning_curve.py \
    --run-dir runs/worker/worker_dqn_worker_dqn_no_truth_20260523_222938

# 学习曲线多个 run 对比示例
python scripts/plot_learning_curve.py \
    --run-dir runs/worker/run1 \
    --compare-runs runs/worker/run2 runs/worker/run3

# BC 预训练示例，worker 可以换成 requester
python scripts/pretrained_bc.py \
    --side worker 
python scripts/train_worker_dqn.py \
    --pretrained runs/bc/.../checkpoints/best.pt
# 以上两者请连起来做，否则爆炸

# legacy smoke
python scripts/smoke_env.py
python scripts/smoke_requester.py
```


---

## 5. 已知陷阱与领域知识

### 5.1 `include_truth_in_candidates=True`（默认）

训练与评估时，**真实标签几乎总在 K 个候选里**。会导致：

- `category_match` 等基线 Hit@1 **虚高**（子集实验上可达 ~1.0）。
- DQN 的 Hit@1 相对基线优势可能被低估或对比失真。

-(此行删除)**Agent 任务**：若做公平对比，应实现 CLI `--no-truth-in-candidates` 并跑消融；报告中必须说明两种设定。
- 训练与评估时，默认会将真实标签强制加入 K 个候选中：

```python
include_truth_in_candidates=True
```
### 5.2 Cache

- 路径：`cache/dataset_{all|n50}.pkl`，version=2 字典序列化。
- 从脚本运行 pickle 安全；缓存损坏时会自动全量重解析。
- 换 `max_projects` 会生成不同 cache 文件。

### 5.3 Entry 字段

- `revisions[].score`、`winner`、`finalist` 用于奖励。
- `withdrawn=True` 的 entry 不进入事件流。

### 5.4 性能

- 全量事件上万步；应用 `max_steps` 做调试，正式实验去掉步数上限。
- 特征已用 `bisect` 优化历史查询；避免在 `step()` 里全表扫描。

### 5.5 Platform 环境机制（2026-05-30 后）

- **Reward 双模式**：`utility`（默认）用可观测利益 proxy 训练；`hit_rate` 仅诊断。`legacy` 复现旧 hit 导向实验。
- **Requester 触发条件**（默认非即时）：申请池 ≥ `requester_batch_size`（8）/ 距 deadline ≤ `requester_deadline_buffer_hours`（24h）/ 池满 32 / deadline 强制。
- **申请池过小**：即时选人时池子恒为 1；batch 模式下 train 上 `avg_requester_pool_size` 约 7–9。诊断：`python scripts/analyze_pool_candidates.py --max-projects 50 --max-steps 200`。
- **`max_steps_per_episode` 未在 `step()` 内强制**：评估/诊断脚本须自行限制步数；`wait_until_deadline` 全 WAIT 时会无限循环。
- **旧 checkpoint 不兼容**：特征 14/17 维 + batch requester + utility reward 均与 `runs/report_full_20260529` 不同，需重训。

### 5.6 Platform DQN 训练超参（2026-05-31 默认）

`train_platform_dqn.py` **不读取** `configs/default.yaml` 的 dqn 段；正式口径如下：

| 项 | Worker | Requester |
|----|--------|-----------|
| `batch_size` | 64（`--worker-replay-batch`） | 32（`--requester-replay-batch`） |
| `buffer_size` | 100000（`--worker-replay-buffer`） | 50000（`--requester-replay-buffer`） |
| `epsilon_decay_steps` | **15000** | **8000**（按梯度步；全量约 ~1800/433 步/ep） |
| `lr` | 3e-4 | 3e-4（`--requester-lr` 可单独设） |

脚本默认：`max_projects=0`，`max_steps=0`，`episodes=20`，`device=cuda`。smoke 请加 `--max-projects 50 --max-steps 100 --device cpu`。

Checkpoint（utility）：`worker_U + 5×requester_U + 0.05×worker_hit + 0.10×requester_hit + 0.02×requester_recall@k`。

---

## 6. 多 Agent 协作分工建议

| 角色 | 职责 | 主要文件 | 交接物 |
|------|------|----------|--------|
| **Data** | EDA、划分说明、特征文档 | `dataset.py`, `features.py`, 报告 §2 | 统计表、图 |
| **Env** | MDP 定义、奖励、候选逻辑 | `env/*.py`, 报告 §3.2–3.3 | 公式与参数表 |
| **RL** | DQN 变体、训练、调参 | `models/dqn.py`, `train_*.py` | `runs/*/checkpoints/best.pt` |
| **Eval** | 基线、test 评估、填表 | `baselines.py`, `run_baselines.py` | `comparison.csv` |
| **Report** | 正文、PPT、可复现命令 | `docs/report_outline.md` | PDF + 命令清单 |

**协作规则**：

1. 每 Agent 开工前读取 `agent.md` 与 `runs/*/config.json`（如有）。
2. 改接口（Observation 字段、checkpoint 格式）必须在 `agent.md` 或 PR 说明中**显式通知**其它角色。
3. 主实验结果只认 **`split=test`** 且写入 `runs/platform_baselines/platform_test_*/comparison.csv` 的行；旧 `runs/baselines/*_test/comparison.csv` 仅作 legacy 对照。
4. 任何会影响报告解释的模型问题都必须同步写入 `docs/experiment_process.md`，包括奖励设计、状态/动作定义、候选集设定、特征工程、Q 网络结构、DQN 变体、超参、训练不稳定、指标异常等；工程性问题不写入该文件。



## 7. 推荐提示词（复制给 AI）

### 7.1 新 Agent 入门


你正在参与「强化学习大作业 · 众包任务推荐」项目。
请先阅读仓库根目录 agent.md、README.md，再读你要改的文件。
目标：课程要求的双端 DQN 推荐；不要偏离 offline RL + 双 MDP 架构。
改完后运行 agent.md §4.4 中最相关的 smoke/评估命令并汇报结果。


### 7.2 跑全量实验


在 agent.md 约束下，使用全量数据（--max-projects 0）完成：
1) train_platform_dqn（episodes≥10，保存 worker_best.pt 和 requester_best.pt）；
2) test 集 run_platform_baselines（含 DQN checkpoint）；
3) 将 platform comparison.csv 关键指标总结为 Markdown 表格。
不要改动 data/data/；记录完整命令与 runs 路径；同时更新 docs/experiment_process.md，说明模型相关困难、调整目的、具体改变、指标变化与可能原因。


### 7.3已删除



### 7.4 写报告某节

```
根据 docs/report_outline.md 第 N 节提纲撰写正文。
引用仓库已实现公式与文件路径；实验数字只使用 runs/ 下真实 CSV/JSON，缺失处写「待填」勿编造。
中文，学术语气，篇幅与提纲匹配。
```

### 7.5 Debug 训练慢/卡住

```
阅读 agent.md §5.4 与 worker_env._build_candidates。
用 python -u 对 step() 做 50 步计时；检查 while 死循环、特征全量扫描、错误 cache。
修复后更新 agent.md 已知陷阱一节。
```

---

## 8. 全量实验前检查清单（Agent 可代为执行）

```
□ python -m src.dataset --max-projects 0  # 确认能加载
□ run_platform_baselines smoke 通过
□ 确定 episodes、K、seed、DQN 变体列表
□ 决定 include_truth 是否做消融
□ train_platform_dqn → run_platform_baselines --worker-checkpoint/--requester-checkpoint
□ test 集结果写入 report_outline §4.4 表格
□ 从 metrics.csv 出学习曲线图
□ docs/experiment_process.md 已补充本轮模型相关困难、调整、指标变化和原因分析
```

---

## 9. 文档与入口索引

| 文档 | 受众 |
|------|------|
| `agent.md`（本文件） | AI Agent、协作规范 |
| `README.md` | 人类快速命令 |
| `docs/report_outline.md` | 实验报告结构 |
| `docs/experiment_process.md` | 模型实验过程、困难、调整目的、指标变化与原因分析 |

**人类同学**：改代码用 AI 时，在对话首条 @ 本文件或粘贴 §7 提示词。  
**AI Agent**：把本文件当作 Source of Truth；与 `README` 冲突时，以**作业要求**和**本文件 §1、§3** 为准。

---

*最后更新：2026-05-31 — 含训练超参优化（分离 worker/requester、全量默认）。*
