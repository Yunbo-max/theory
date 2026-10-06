# Recursive SSD：单张 2080 Ti 的 8 小时研究实验

这是基于 [Apple SSD](https://github.com/apple-aiml-research/ml-ssd) 的**递归自策略训练实验代码**。每轮用自己的上一轮模型生成未经验证的回答，再从上一轮权重继续训练。训练不使用正确答案、测试通过信号、外部教师或奖励模型。

**当前状态：工程实现与 CPU 检查；尚无真实 2080 Ti 跑分。不能据此断言“没人做过”或“能够递归提升”。** 迭代自蒸馏已有相关研究。本项目研究具体约束能否优于简单递归和软目标基线。

## 先安装，再在下午 4 点启动

环境：Linux、Python 3.11 或 3.12、可用的 NVIDIA 驱动和 Docker。默认只使用 `CUDA_VISIBLE_DEVICES=0`，FP16，不使用 BF16、vLLM、FlashAttention 或 bitsandbytes。请在计时前完成下载与安装。需要约 15–25 GB 可用磁盘，实际取决于环境缓存和输出。

```bash
git clone https://github.com/Yunbo-max/theory.git
cd theory
bash scripts/setup.sh
```

如果系统只有 Python 3.12：

```bash
PYTHON=python3.12 bash scripts/setup.sh
```

安装脚本依次安装 CUDA 11.8 对应的 PyTorch、运行 CPU 测试、构建隔离评测镜像、下载固定版本模型/数据、执行 GPU 显存预检。预检失败时先看报错；它不会自动替换模型或悄悄缩减实验。主配置是 `configs/2080ti_8h.json`。

**下午 4 点在你的机器执行：**

```bash
bash scripts/run_8h.sh
```

运行器从启动时计时 8 小时，到期停止新任务，保存已完成的优化步和报告。也可以提前启动等待指定时刻；时间必须带时区。例如 **2026-10-06 英国夏令时 16:00**（如果你指其他时区，请改偏移）：

```bash
bash scripts/run_8h.sh --start-at '2026-10-06T16:00:00+01:00'
```

这是本地命令，仓库本身没有替你启动或预约 GPU。终端可用 `tmux` 保持会话。实时查看：

```bash
tail -f runs/2080ti-8h/worker.log
```

## 默认实验做什么

| 项目 | 设置 |
|---|---|
| 模型 | 固定 revision 的 Qwen2.5-Coder-1.5B-Instruct |
| 训练 | FP16 冻结基座 + FP32 LoRA，rank 16，q/k/v/o 投影 |
| 数据 | MBPP 的 train split，固定 32 条**题目文本**，每题 2 个自己的样本 |
| 生成 | 温度 1.5 → top-k 20 → top-p 0.8；最多 384 个回答 token |
| 递归 | 最多 3 轮；每轮继承上一轮 adapter、冻结 teacher、重新生成数据 |
| 评测 | 固定 32 道 HumanEval+ 原生题目，每题 2 次采样，官方 EvalPlus 0.3.1 |
| 确认集 | 剩余 132 道题保留；默认试跑不使用它们选择方法 |
| 截止 | 最长 8 小时；未完成项目明确标记 pending/incomplete |

先评估原模型，再按轮交错执行下列方法，优先得到相同轮数的比较：

1. `hard`：原始硬目标递归 SSD。
2. `full_soft`：直接学习经过采样变换的完整分布 μ，作为强简单基线。
3. `M03` / `floor_projection`：软目标满足初始分布的概率下限。
4. `arithmetic_anchor`：简单混合 `(1−c)μ+c p₀`，检验复杂约束是否有价值。
5. `fixed_data`：一直训练第一轮数据，区分“刷新自生成数据”和“增加训练步数”。
6. `lower_lr`：硬 SSD 的一半学习率，检验收益是否只是减小更新。

**不承诺 8 小时跑完全部轮次。** 实测吞吐决定完成数量。每个方法保留实际生成 token 数、训练步数、溢出/跳过、耗时、截断比例、显存和评测原始结果。不同输出长度/参考前向会改变成本；这不是天然的等 FLOPs 比较。

## 优先候选的数学含义

在相同前缀上，令 `p₀` 是最初模型，`μ` 是当前模型经过温度和截断后的分布。求：

$$\min_q D_{KL}(\mu\|q),\quad q_v\ge c p_{0v},\quad\sum_v q_v=1.$$

它的解为：

$$q_v=\max(c p_{0v}, k\mu_v),$$

其中 `k` 用二分法求到归一化。默认 `c=0.1`。q 在本轮训练中由冻结 teacher 与初始模型构造，下轮随新 teacher 刷新。

**这只约束软目标。有限步 LoRA 训练后的模型未必严格满足下限。** 因此程序还在共同的初始模型生成前缀上测量实际 KL、熵、Gini 多样性和概率下限违反量。这些量也不能保证答案正确。

基础推导、限制与反例见 [research/THEORY.md](research/THEORY.md)。20 个候选经过条件数学审查后选择 15 个实现，见 [research/CANDIDATES.md](research/CANDIDATES.md)。审查由同一助手完成，非独立同行评审，也不是自动定理证明。

## 其他 14 个实现怎么启用

`recursive_ssd/methods.py` 提供 M01–M15；M12 的梯度投影、M13 的实际 KL 回溯在 `train.py`，M14 的题目采样分配、M15 的历史策略混合生成在 `runner.py` / `model.py`。

复制配置并把相应 ID 加入 `arms`，仍保留 `hard` 与 `full_soft`。例如 `M04` 为总温度预算、`M08` 为历史 teacher 平均。M12 应同时加入 `full_soft_sgd`，M09/M11 应加入 `fixed_alpha` / `fresh_alpha` 等卡片指定的控制。配置不包含自动根据测试结果挑选方法的逻辑。

修改配置后重新预检，并使用新的运行目录；**不要修改正在运行的配置、代码或数据**。相同运行目录重启会继续原截止时间，不能凭重启多获得 8 小时。

## 中断、结果和回传

同一代码和配置下重新运行原命令，可以复用已生成记录和最近完整优化步；每步保存 optimizer、scaler、随机数状态和父模型哈希。源代码、数据或评测镜像变化会拒绝混合续训。最终 adapter 是本地产物，保留在各轮 `adapter.pt` 中。

```bash
.venv/bin/python -m recursive_ssd.cli report
.venv/bin/python -m recursive_ssd.cli collect
```

结果：

- `runs/2080ti-8h/REPORT.md`：完成情况和 HumanEval+ pass@1。
- `runs/2080ti-8h/report.json`：成对任务 bootstrap 区间、分布诊断、优化有效性。
- 每个 round 内：未经筛选的生成记录、teacher 父哈希、训练日志、原生评测结果。
- `returns/2080ti-8h.tar.gz`：可回传的结果包，包含失败/未完成记录，排除模型权重。

把结果包发回即可继续分析。它不会自动向 GitHub 上传你的本地文件。

## 与原始 SSD 的差异与证据边界

上游 commit 固定为 `2637d2021f1bc523385b48a1f88ea9aa4812b0a9`。`vendor/apple-ssd/` 保留检查过的原始生成代码、配置、模板和许可证；训练题目的提示词实际复用其 function 模板。上游未公开 SFT trainer，因此这里补写了共享基座的 LoRA trainer。

这里改变了模型规模、训练集、精度、优化方式、生成长度和评测集；不做上游默认的按长度去除底部 10% 样本，也不做正确性筛选。它是资源受限的机制研究，**不是 SSD 论文的精确复现**。所有合法生成记录（含 EOS-only、长度截断、错误代码）都保留。

CPU 测试中的小型随机模型和有限概率向量只是工程夹具。没有伪造 GPU 结果，也没有把它们当作新 benchmark。32 题、1 个 seed 的结果一律先标为 `INCONCLUSIVE`；更强结论需要完整对照、确认集、额外 seeds 和成本匹配。

相关论文及模型/数据固定版本见 [research/SOURCES.md](research/SOURCES.md)，实验协议见 [docs/PROTOCOL.md](docs/PROTOCOL.md)，工程验证范围见 [docs/VALIDATION.md](docs/VALIDATION.md)。
