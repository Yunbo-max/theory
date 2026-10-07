# Recursive SSD：完整多 benchmark 实验队列

基于 [Apple SSD](https://github.com/apple-aiml-research/ml-ssd) 的递归自策略训练。每轮从上一轮权重继续，用自己的回答训练；训练不读取正确答案、测试通过信号、外部教师或奖励模型。

当前入口是 `python scripts/research.py`。已实现完整的方法目录、必要对照、开发与确认阶段、原生评分、可恢复依赖队列和统计分析。旧 pilot CLI 已停止接收执行任务。**评测使用原生 Conda/Python，不使用 Docker。**

**尚无本项目的 2080 Ti 实验结果。** 软件检查、官方评分器参考回放和科学结论是不同层面的证据；代码完成不能证明递归提升或方法新颖性。

## Local Codex: start here

用户已明确：Web 完成代码与完整实验设计，**由 Local 来测试和运行实验**。
先在实际交付 commit 阅读 [AGENTS.md](AGENTS.md)、
[LOCAL_AGENT_RUNBOOK.md](LOCAL_AGENT_RUNBOOK.md) 和
[当前 Web handoff](rounds/2026-10-07-local-handoff/WEB_HANDOFF.md)。
[模型/数据下载入口](LOCAL_AGENT_RUNBOOK.md#download-datasets-and-models) 指向
[完整资产清单](docs/LOCAL_ASSETS.md)；[输入契约与模板](docs/LOCAL_INPUT_CONTRACTS.md)
说明如何从真实证据构造缺失的 native/protocol/dispatch 输入。

Local Codex 在用户电脑上通过 SSH 控制远程 GPU，远程不需要安装 Codex。
新交付的修改标记 `generated_unexecuted`；历史 332 项 CPU 检查不代表这些修改
或目标机器已经验收。以下命令必须结合 runbook 的顺序和真实授权使用；旧日期
不是现在可直接重开的八小时预算。

## 完整设计

| 环节 | 固定设置 |
|---|---|
| 方法 | 已选 15 个方法及逐方法必要对照；原始 20 个数学候选保留 |
| 核心对照 | 初始模型、递归 hard SFT、full soft、固定首轮数据；各方法再配更强简单基线和机制消融 |
| GKD 对照 | FKL / RKL / JSD × 每轮缓存 / 每次更新用当前策略采样，共 6 个条件 |
| 训练输入 | 与 MBPP+ 无精确 ID 重叠的 clean-v2 32 条题目文本，每条 2 个自己的回答 |
| 开发 | seed 17；HumanEval+ 开发集 32 题；保留 R1/R2/R3 |
| 公平调参 | 每个指定标量最多 3 个预定值；候选与简单基线同等时间额度；保留每次尝试 |
| 确认 | 最多 2 个开发入选方法；新种子 23/47/71/101/131；固定 R3 端点 |
| HumanEval+ | 原始 164 题：开发 32、独立确认 132；每题 10 次，pass@1 / pass@10 |
| MBPP+ | v0.2.0 全部 378 题；每题 10 次，pass@1 / pass@10 |
| LiveCodeBench | release_v5 全部 880 题；每题 10 次，pass@1 / pass@5 |
| 边界实验 | R5 与 0.5B 模型分别建立独立队列；不替代主要 R3 确认结果 |
| 统计 | 配对任务 × 训练种子 bootstrap；预定完整比较族；缺失、失败、不合格对照全部保留 |

完整设计、实现覆盖和当前证据见 [G01 完整覆盖](research/suite-v3/G01_COVERAGE.md)，逐方法问题和对照见 [方法矩阵](research/design-v2/METHOD_MATRIX.md)。[design-v2](research/design-v2/G01_EXPERIMENT_DESIGN.md) 保留历史审查状态。实际执行契约见 [运行说明](docs/RUN_MULTIBENCH.md)、[基准适配器](docs/BENCHMARK_ADAPTERS.md)、[分析规则](docs/ANALYSIS.md)。

完整设计可以大于一次八小时窗口。队列先用目标主机实际测量的成本判断**整组比较**是否装得下；不通过删题、减少确认种子或漏掉难跑对照来声称完成。未启动或未完成的项目保留为 pending/incomplete。

## 安装与准备

Linux，Python 3.11/3.12，目标主机一张 RTX 2080 Ti。FP16 基座配 LoRA；不依赖 BF16、vLLM、FlashAttention 或 bitsandbytes。完整 LCB 原始文件约 4.35 GB；还需模型、环境和每次隔离执行的副本空间，准备充足磁盘。

```bash
git clone https://github.com/Yunbo-max/theory.git
cd theory
# Conda 环境创建和激活按 LOCAL_AGENT_RUNBOOK.md 的预算内 harness 步骤进行。
# 以下仅适用于已激活且经核对的 native Conda 环境。
: "${SSD_ORIGINAL_START:?先读取实际原始授权及已用预算}"
python scripts/research.py budget --queue runs/multibench-v3 \
  --original-start "$SSD_ORIGINAL_START"
bash scripts/setup.sh --queue runs/multibench-v3
```

原计划为 2026-10-06 英国夏令时 16:00；不能用当前时间重开八小时。首次建立账本时按 runbook 计入真实已有消耗；恢复已有账本不再次传入累计消耗。后续队列用 `--budget-from runs/multibench-v3` 共用原始预算。

安装、依赖检查、软件测试、数据与模型下载都经仓库固定版本的 research-autopilot harness。也可分步执行：

```bash
python scripts/research.py setup --queue runs/multibench-v3 --seconds 1800
python scripts/research.py check --queue runs/multibench-v3 --seconds 300
python scripts/research.py assets --queue runs/multibench-v3 --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py model --queue runs/multibench-v3 --data artifacts/multibench-v3 --seconds 1800
python scripts/research.py qualify --queue runs/multibench-v3 --data artifacts/multibench-v3 --benchmark humaneval
python scripts/research.py qualify --queue runs/multibench-v3 --data artifacts/multibench-v3 --benchmark mbpp
python scripts/research.py qualify --queue runs/multibench-v3 --data artifacts/multibench-v3 --benchmark livecodebench
```

`qualify` 的参考答案/已发布模型输出回放只检验评分器。真实模型基线资格、显存与吞吐量、Parent / Gate 0 / IPCG 和冻结协议仍需要真实证据。[科学准入说明](docs/SCIENTIFIC_ADMISSION.md) 给出这些绑定的格式；程序不会代造通过记录。

## 编译、运行与恢复

```bash
python scripts/research.py design --stage tuning --output runs/tuning-suite.json
python scripts/research.py design --stage development --output runs/development-preview.json
python scripts/research.py build --suite runs/tuning-suite.json \
  --data artifacts/multibench-v3 --queue runs/tuning \
  --budget-from runs/multibench-v3 --original-start "$SSD_ORIGINAL_START"
python scripts/research.py status --queue runs/tuning
```

`build` 不启动 GPU。先按 runbook 完成实际调参及选择，再生成带真实调参结果的开发 suite 和队列；不要把 preview 当调参结果或覆盖已绑定的不可变队列。跨阶段共用预算，恢复不会重新获得八小时。

开发队列建立后，准备并检查真实主机、基线、协议和方法证据，通过 `native-assets`、`protocol`、`freeze`、`admit` 建立准入绑定，再运行：

```bash
python scripts/research.py admit --queue runs/multibench-v3 --bundle M03 \
  --bindings research/m03-dispatch-bindings.json
bash scripts/run_8h.sh --queue runs/multibench-v3 --bundle M03
python scripts/research.py resume --queue runs/multibench-v3 --bundle M03
python scripts/research.py report --queue runs/multibench-v3
python scripts/research.py select --queue runs/multibench-v3 \
  --output runs/multibench-v3/finalists.json
python scripts/research.py collect --queue runs/multibench-v3 \
  --output returns/multibench-v3.tar.gz
```

完整调参 → 开发选择 → 确认流程、冻结文件与命令参数见 [RUN_MULTIBENCH.md](docs/RUN_MULTIBENCH.md)。确认阶段不会依据确认成绩重新挑超参数、种子或 checkpoint。训练中的 teacher 始终在每轮冻结；per-update GKD 改变的是生成前缀的当前策略。

确认和两个边界目录自动继承入选开发队列的配置、调参结果和已测校准；不同覆盖值会被拒绝。R5 的 head 温度在 R4/R5 预先固定沿用开发 R3；0.5B 边界将开发模型的温度作为迁移设置，不能把它报告为 0.5B 上已匹配的分布校准。

## 结果与可复现性

每次尝试保留原始生成、官方逐题判定、优化事件、checkpoint 哈希、完整输入与源代码身份、失败原因和实际成本。LiveCodeBench 使用逐题索引和缓存读取原始测试，保留全部 880 题和所有原始测试，不把测试内容交给生成器。

[实现记录](docs/MULTIBENCH_IMPLEMENTATION_PLAN.md) 跟踪本次修改；[research/STATUS.json](research/STATUS.json) 区分工程证据、主机证据和科学结论。旧 `native-v1` / `design-v2` 记录保留其原始含义；新的验证不会追认旧数据或过期 PASS。
