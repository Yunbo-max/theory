# 完整实验设计与实现覆盖：suite-v3

本轮落实 design-v2 的全部比较义务，复用已经保留的 20 个数学候选、前 15 名选择和逐方法预测。研究问题、数学条件、证伪标准不因修复代码而改变。本文是可核对的实现覆盖及执行计划；正式设计资格仍等待目标主机证据和科学准入。

## 问题、比较与判定

研究问题：训练仅使用自身策略生成的回答和自身分布，递归更新能否提高原生正确率，并保留多次采样可解题目的覆盖？训练不读取正确答案、隐藏测试、通过率、奖励模型或外部教师。开发集成绩只用于预先规定的选择；确认集不参与训练或选择。

每个候选均有初始模型、递归 hard SFT、full-soft、固定首轮数据和逐方法必需对照。GKD 启发的同信息对照包含 FKL/RKL/JSD × 按轮缓存/每次更新采样六种条件；教师每轮冻结，按更新采样改变的是学生的生成策略。这不是原 GKD 外部教师设置的精确复现。

完整问题、最强替代解释、比较臂和反驳条件见 [逐方法矩阵](../design-v2/METHOD_MATRIX.md)。公式边界只能支持其条件内的目标性质，不能推导神经训练必然提升或无限递归稳定。保留型方法的主要确认标准沿用 R3 pass@1 改善 2pp、相对必要简单替代的覆盖改善 2pp、pass@1 非劣界 -1pp；M07 另需质量非劣和完整训练成本下降至少 10%。所有条件按同时置信区间及实际资格判定，缺失证据为 INCONCLUSIVE。

## 基准与执行覆盖

| 基准 | 固定数据与范围 | 生成和原生指标 | 实现与当前证据 |
|---|---|---|---|
| HumanEval+ | v0.1.10，164；固定哈希开发 32 / 确认 132 | n=10，T=.8，top-p=.95，最大生成 384；EvalPlus 0.3.1 全部 base+plus，pass@1/pass@10 | `benchmarks.py`、`suite_score.py`；全量资产已核对，官方两题参考回放通过，目标主机模型运行待做 |
| MBPP+ | v0.2.0，完整 378 | 与 HE 相同生成设置；官方转换及全部测试，pass@1/pass@10 | 同上；全量资产已核对，官方两题参考回放通过 |
| LiveCodeBench | code_generation_lite release_v5，完整 880；数据 revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | n=10，T=.2，top-p=.95，最大提示 4096 / 生成 2000；原生 pass@1/pass@5 | `lcb_score.py`；全量五个原始文件及 880 行索引核对完成，官方 evaluator `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`；本环境 IPC 限制使参考回放未通过 |

LCB 按原始行字节位置流式载入，每题全部原始测试和 10 个生成均保留；没有因文件大而删题、截取测试或改评分规则。生成器仅收到独立题面文件。评分器基础设施错误与模型答错分开，不能进入合格结果缓存。

clean-v2 训练只保留固定 32 条 MBPP 题目文本，每条两个自生成回答；它们与 MBPP+ 的精确 ID 重叠为零。必须从原始基座开启此新数据分支，不能继续旧重叠数据的 adapter。此检查不等于排除语义近重复或预训练污染。

## 有限且完整的实验库存

| 阶段 | 固定范围 | 选择与输出 |
|---|---|---|
| 公平调参 | M03/算术锚定的 floor={.03,.1,.3}；M01/同平滑算术的 alpha={.25,.5,.75}；seed 17，HE 开发集，固定 R3 | 每族三次；候选/对照同等实际墙钟额度，失败计费；完整结果才允许选择 |
| 开发 | 15 个候选的全部必要比较；seed 17；HE 开发集；R1/R2/R3 | 按相对最强合格简单基线的 R3 pass@10 差值排序，要求 pass@1 降幅不超过 1pp；最多两名进入确认 |
| 确认 | 入选者及其完整比较；独立训练种子 23/47/71/101/131；三个完整基准；固定 R3 | 自动继承开发选择的参数及适用校准；不选最佳轮次、种子或确认超参数 |
| 递归边界 | 相同入选比较，独立 suite 身份，训练到 R5 | R5 估计仅描述该边界，不借用 R3 主判据发 PASS |
| 模型边界 | 相同入选比较，固定 Qwen2.5-Coder-0.5B-Instruct revision，R3 | 独立模型、成本与基线资格；不能从 1.5B 结果推断 |

主模型是固定 revision 的 Qwen2.5-Coder-1.5B-Instruct，FP16 基座与 LoRA rank 16。实际配置、模型文件、数据、解码、种子、代码和环境均进入冻结身份。跨比较复用基线必须满足相同绑定。生成随机数按预定任务/样本身份配对，轮次和每题多个生成不算独立训练重复。

`suite_design.py` 编译完整库存，`suite_queue.py` 构造训练 → 校准 → 评测依赖，`suite_train.py` 执行每轮训练。开发、调参目录随本轮保存；确认目录只能由真实完整开发选择产生，当前没有捏造入选者或未来 checkpoint。

## 逐项关闭旧实现缺口

| design-v2 义务 | 本轮实现 | 对应工程检查 |
|---|---|---|
| SOFT_LR、SOFT_MIX、SMOOTHING_CONTROL | full-soft 低学习率、确定性软混合、相同 epsilon 的算术/几何对照 | `test_controls.py`、`test_suite_design.py` |
| HEAD_FACTORIAL、HEAD_TEMPERATURE | 头部形状 × 质量转移四条件；开发前缀匹配温度后冻结 | `test_controls.py`、`test_suite_calibration.py`、`test_suite_queue.py` |
| TAIL_IID、NOISE_CONTROL | 同头部/抽样数的 iid 尾部；相同二次噪声预算的常量 alpha | `test_controls.py` |
| TEMPORAL_CONTROLS、MEAN_GATE、MEAN_WEIGHT | 当前-当前、滞后教师、常量均值门、常量均值权重 | `test_controls.py`、`test_suite_train.py` |
| SGD_CONTROLS、CLOCK_CONTROL | SGD 算术/等范数对照，真实回溯步长、训练时间和全轮成本匹配 | `test_training.py`、`test_suite_train.py`、`test_suite_calibration.py` |
| ALLOCATION_CONTROL、EXPLORATION_FACTORIAL | 均匀代理、去权重校正、祖先生成 × 目标锚定完整因子 | `test_controls.py`、`test_suite_train.py`、`test_suite_design.py` |
| DECODE_CONDITIONS、GKD | 截断/无截断及根式/一次温度日程，六种 GKD 条件 | `test_controls.py`、`test_suite_design.py`、`test_suite_train.py` |
| TUNING、ANALYSIS | 等额三值调参，开发选择锁定，完整库存与任务×种子配对 bootstrap | `test_suite_queue.py`、`test_analysis.py` |
| BENCHMARKS、NATIVE | 三个原生加载器/评分器、完整分母、逐题缓存和进程清理 | `test_benchmarks.py`、`test_native_runtime.py`，实际参考回放记录 |
| PROTOCOL、COST、PARENT | 实际 skill 协议编译、冻结、准入、共享预算接口 | `test_suite_admission.py`、`test_suite_queue.py`；接口检查不代替尚缺的真实主机/科学证据 |

旧 [implementation-gaps.json](../design-v2/implementation-gaps.json) 保留当时状态；本文和 `implementation-coverage.json` 是本轮覆盖记录，不倒改历史证据。

## 成本、统计和停止规则

只有用户原有的一张 2080 Ti、原始八小时总授权。不同队列和阶段共享同一个授权账本；预处理、失败、重试、官方回放均计费。候选调度需要目标主机上训练、生成、额外前向、评测、保存和收尾的实测上界，按整组比较预留。未测耗时/显存不能用一个名义八小时字段代替。完整库存可能超过该窗口；超额项目保留 pending，不能自动开启新窗口或减少必要对照。

训练时间报告包含历轮生成、代理前向、优化、保存、加载；优化时间另外报告。匹配时间不达标、超出公平调参额度或优化无效的观察不参与正式估计。所有失败、原始生成、逐题官方判定、优化事件、父子 checkpoint 哈希和实际费用留存。普通恢复不自动重跑失败；显式开发重试最多一次并预留额外成本，确认不自动重试。

时间匹配对照的 `config.clock_match_tolerance_seconds` 必须在协议中冻结，并以目标主机实测单步粒度说明依据；程序默认零容差，不会把一次提前结束或任意偏差自动算成匹配。该容差不能根据确认分数事后扩大。R5 的头温度在 R4/R5 沿用已冻结的开发 R3 值；0.5B 边界使用开发模型参数迁移时，报告明确禁止将其称为新模型上的同分布匹配。两种边界的结果保持描述性，额外科学主张需要相应独立协议。

配对任务×独立训练种子交叉 bootstrap，分析 seed=20261006，各基准/模型分开，完整预定比较族采用 Bonferroni 同时区间；采样次数按整个比较族的尾部精度在看到结果前固定。缺失和无效比较保留分母，不因失败缩小多重比较族。机制四条件保留交互项，不凭熵/Gini 变化替代任务端点。

正式执行仍需目标主机原生评分资格、真实基线/自然失败、Parent/Gate 0/IPCG、当前设计审查、冻结协议与 E04。程序拒绝缺失或过期绑定。当前没有本项目 GPU 结果，因此没有已证实的递归收益、新颖性或科学 PASS。
