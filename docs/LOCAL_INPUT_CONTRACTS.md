# Local 执行输入：字段、证据来源与尚未关闭的阻塞

本页补充 [运行说明](RUN_MULTIBENCH.md) 中四个尚未生成的路径。它基于
`f8a4ca035424668dc8bb805017d1dbe93a9a3a65` 和本轮新增的 qualification/child
源代码静态阅读；新增路径的状态是 `generated_unexecuted`。本次没有运行 Python、
测试、评分器或 GPU。现有历史工程证据继续有效于原记录的范围，不覆盖本轮新代码。

模板位于 [research/templates](../research/templates)。它们是 **JSONC 文档**，
含注释及故意保留的 `null`，不满足执行 schema。CLI 的 `read_json` 使用严格
JSON，不能直接执行这些文件。Local agent 必须先从真实证据构造对应 `.json`；
删除注释不等于补齐证据。模板没有虚构 SHA、GPU UUID、评分阈值或批准记录。

| 目标文件 | 本次交付 | 真正生成/消费它的入口 |
|---|---|---|
| `research/native-assets-input.json` | [native-assets-input.template.jsonc](../research/templates/native-assets-input.template.jsonc) | `native-assets --spec` → `prepare_native_assets` |
| `research/protocol-input.json` | [protocol-input.template.jsonc](../research/templates/protocol-input.template.jsonc) | `protocol --spec` → `prepare_native_protocol` |
| `research/m03-dispatch-bindings.json` | [m03-dispatch-bindings.template.jsonc](../research/templates/m03-dispatch-bindings.template.jsonc) | `admit --bundle M03 --bindings` → `admit_bundle` |
| `research/frozen-m03.json` | [frozen-m03.output-notice.jsonc](../research/templates/frozen-m03.output-notice.jsonc) | `protocol` 先产生草案，`freeze` 才执行真实冻结事务 |

## 共用引用与当前已有材料

每个文件引用严格为 `{"path": "项目内相对路径", "sha256": "实际文件字节的 SHA-256"}`。
文件必须存在、哈希一致，不能用 Git blob SHA 替代 SHA-256，不能引用项目外路径或符号链接。
原始字节哈希与对象规范序列化哈希是不同操作；配置、代码引用列表和 suite 的摘要应遵循
[`recursive_ssd/io.py`](../recursive_ssd/io.py) 的 `object_hash` 及编译器输出。

以下已有文件可以阅读或引用；**它们不替代目标主机测量或科学批准**：

| 已有路径 | 可用范围 |
|---|---|
| `research/suite-v3/catalogs/development.json` | 当前完整开发目录；M03 共 16 个比较臂，开发 seed 17、R1/R2/R3 |
| `research/suite-v3/catalogs/tuning.json` | 独立公平调参目录；不把开发目录当成调参目录 |
| `research/suite-v3/implementation-batch.json` | 当前代码发现/验证批次输入；不能冒充尚缺的完整设计验证批次 |
| `research/suite-v3/implementation-check.json` | 历史实际校验输出；不是 `verification_ref` 所需的批次输入 |
| `research/design-v2/data/humaneval-manifest.json` | 固定的 32 个开发 ID 与 132 个确认 ID；只用于已核对的拆分契约，不复用其旧训练数据 |
| `research/design-v2/data/train_prompts-clean-v2.jsonl` | clean-v2 的 32 条 prompt-only 训练输入 |
| `research/suite-v3/native-qualification/` | 原作者环境的参考回放及失败证据；LCB 的 IPC 失败不是模型错误或评分器资格通过 |
| `research/suite-v3/G01_COVERAGE.md`、`docs/ANALYSIS.md` | 设计覆盖与统计规则的文字来源；尚不能当成正式冻结协议 |

`assets` 和 `model` 在目标主机生成 `artifacts/multibench-v3/benchmarks-manifest.json`
及 `model-manifest.json`，连同它们列出的实际原始文件。GitHub 保留的资产身份副本没有携带
完整 4.35 GB LCB 数据，不能将副本路径替换为本地数据路径而保留旧引用。

## 1. Native assets 输入

真实 CLI：

```bash
python scripts/research.py native-assets --spec research/native-assets-input.json
```

源码入口为 [`prepare_native_assets`](../recursive_ssd/suite_admission.py)。根对象字段
为 `asset_manifest_ref`、`groups`、`definitions`、`output_dir`，以及 HE 子集必需的
`human_eval_split_ref`、`subset_support_ref`。`groups` 与 `definitions` 的键必须完全一致。
模板一次列出四组：`he-dev`、`he-confirm`、`mbpp-full`、`lcb-full`；组名是项目名字，
必须在后续协议和 dispatch 中一致，并非论文宣称的上游官方 split 名称。

每组 definition 的字段及来源：

| 字段 | 需要的真实内容 |
|---|---|
| `benchmark_id`、`benchmark_revision` | 认证后的上游身份；模板只保留已知发布版本 |
| `published_at`、`source_url`、`publication_refs` | 有来源支撑的发布时间、原始网址、留存原始文件引用；不能用本次下载时间当发布时间 |
| `upstream_split` | 上游原始 split；编译器自行派生项目 HE split，不手填 `splits` 或 `project_split_adapter_ref` |
| `metrics[].output_path` | 官方 JSON 中每个指标的实际路径数组；不要猜 `pass@1` 在根目录 |
| `prediction_format` | 官方接受的 `json/jsonl`、记录路径和题目 ID 路径 |
| `sampling` | 本项目已经固定的解码策略及参数；`n=10` 必须保留。HE/MBPP 为 T=.8、top-p=.95、最大新 token 384；LCB 为 T=.2、top-p=.95、最大新 token 2000、最大 prompt 4096（须在完整解码绑定中保留） |
| `budget` | 每个 native 遥测项目的有限正数上界；项目名必须与实际 receipt 中的遥测对应 |
| `scorer` | 官方身份/revision、全部来源与代码引用、真实 argv/cwd、JSON 输出位置、真实分母路径 |
| `subset_support_ref` | 官方评分器确实支持指定 HE 题目集合的源码证据，不是项目自行声称支持的说明 |

`scorer.command` 是 argv 数组，不能填 shell 字符串或 Docker。现有原生校验只接受占位符
`{predictions}`、`{samples}`、`{labels}`、`{output}`、`{seed}`；文件输出模式要求 argv
包含 `{output}`。不能把项目包装器标为官方评分器。若实际采用包装器，保留官方 definition，
在后续 `scorer_spec.scorer` 提供 `faithful_harness`、真实 parity 引用及官方与包装器的现场回放。
详见 [`native-eval-contract.schema.json`](../vendor/research_autopilot/schemas/native-eval-contract.schema.json)
和 [`_native_eval.py`](../vendor/research_autopilot/scripts/_native_eval.py)。

成功返回的 `benchmark_manifest[group]` 是新生成的 native **样本清单引用**；
`native_definition_refs[group]` 是 native definition 引用。后续协议必须使用这些返回值，
不能把下载资产总清单放进 `benchmark_manifest[group]`。LCB 的五个原始文件、索引、来源
sidecar 会形成完整引用闭包，不能只保留 880 行索引。

## 2. Protocol 输入与 M03 比较臂

真实 CLI：

```bash
python scripts/research.py protocol --spec research/protocol-input.json
```

根字段是 `benchmark_manifest`、`arms`、`seeds`、`groups`、`scorer_spec`、`bindings`、
`output`。模板只对应 **M03 开发**：`groups=["he-dev"]`、`seeds=[17]`。前两种 group
映射的键必须与 `groups` 完全一致。`arms`、逐对照资格规则和最终 dispatch 必须保留完整
目录的 16 臂：

`base`、`hard`、`full_soft`、`fixed_data`、`M03`、`arithmetic_anchor`、`lower_lr`、
`floor_zero`、`arithmetic_anchor_tuned`、`full_soft_lr_half`，以及 FKL/RKL/JSD 各自的
cached/update 六臂。模板采用 `M03 → treatment`、`hard → baseline`、`base → initial`；
其他臂保持同名 control role。上游契约中 base 的真实 `name` 是 `initial`。

每个 arm 必须有实际 `revision` 和 `implementation_refs`；不要把所有方法标成 `hard`
以通过 allowlist。每组 `scorer_spec` 需要 native definition 引用、事前规定且有真实来源
的 baseline qualification，以及 **每个** control 的 qualification。`operator` 只允许
`ge/gt/le/lt/eq`；阈值必须事前有依据，不能填 0 来绕过资格。

`bindings.protocol` 由已审查的 G01 内容提供，其 schema 见
[`gate-a-protocol.schema.json`](../vendor/research_autopilot/schemas/gate-a-protocol.schema.json)。
模板保留待补的 evidence mode、evidence snapshot、criteria、guardrails、统计设计、
analysis plan 和 claim IDs。不能因为开发只有一个 seed 就把主确认统计要求降成一次运行。
编译器自行写入 `seed_policy`、`required_groups`、contrasts、native contracts 与摘要；
输入禁止预写 `frozen_at`、`design_verified`、`baseline_qualified` 或 `protocol_digest`。

候选路径还依赖项目根的真实 `event-ledger.jsonl`、`ledger-anchor.json`、
`research-state.json`，以及相互一致的 `parent_problem_ref`、`natural_gate_0_ref`、
`importance_decision_ref` 和当前 candidate。当前仓库没有完整科学准入证据。不要手工建立
“通过”的账本记录。`bindings.method_discovery` 引用真实当前代码批次；正式运行另外还要
完整设计批次，二者用途不同。

`suite_binding` 必须逐项绑定实际 queue 的 suite、benchmark/model manifests、排序后的
code refs 摘要、environment、精确 config 和 effective calibration 摘要。调参阶段还须
额外绑定同额 `tuning_budget`；development 模板不能直接用于 tuning。

`suite_qualification_evidence[group]` 需要真实 qualification protocol 和 run manifest，
对应 provenance 至少包含 model revision、environment digest、data revision。新的分来源格式为
`{"comparators": {"主协议中的非treatment角色": {"protocol_ref": 实际引用,
"manifest_ref": 实际引用, "source_arm_role": "来源协议中的实际角色"}}}`。
它必须覆盖主协议每个非 treatment role；同一个来源 protocol/manifest 会合并为一次回放，
但各角色必须映射到同一真实实现身份。原单一 `{protocol_ref, manifest_ref}` 格式仍要求该
来源包含完整比较身份。既有两题 canonical reference 回放不能替代实际模型完整比较的资格。

## 3. 本轮新增的有界对照资格与候选子协议

原 `baseline-calibration` / `native-evaluator-qualification` 保持 narrow allowlist。
本轮增加普通对照 `comparator-qualification` 与保留候选资格的 child 路径，解决源代码层面的
编排缺口；二者尚未执行验证，不能称为已通过的端到端运行。

普通对照使用 [comparator-qualification.fragment.template.jsonc](../research/templates/comparator-qualification.fragment.template.jsonc)，
将其内容放入完整 protocol-input 的 `bindings.qualification`，并去除该资格协议的
`bindings.method_discovery`。该 fragment 的字段均由实际源码消费：

| 字段 | 必须绑定的内容 |
|---|---|
| `purpose` | `comparator-qualification` |
| `allowed_arm_roles`、`arm_bindings` | 相同完整角色集合；role → exact catalog arm ID，不能重复映射 |
| `suite_ref`、`bundle_id` | 真实 development/tuning suite 及所属比较 bundle；目录必须与编译器产生的完整目录一致 |
| `method_verification_ref` | **当前**代码批次，接受实际实现引用检查；本轮源代码修改后不能直接沿用旧通过报告 |
| `config`、`calibration` | 精确设置；普通 qualifier 的 calibration 覆盖必须为 `{}`，适用校准来自已绑定 suite 或真实依赖 |
| `execution_provenance` | 恰好 model revision、data revision、environment digest 三项，与运行环境一致 |
| `tuning_budget` | development 为 null/省略；tuning 为带真实来源的相同 `trial_wall_seconds` 上限 |

fragment 示例将 qualifier 的 `treatment` 设为 `fixed_data`，`baseline` 设为 `hard`，
另保留 `full_soft`。这只是一个有界来源契约，不是 M03 的完整比较，不提供 M03 训练通道。
扩充或建立其他来源契约时必须保留 native 必需角色和各角色的真实资格规则。
当前 bundle 的 treatment 和任何 underlying `method` 为选中 Mxx 的别名都被拒绝；
例如 `floor_zero` 仍是 M03，不能因为名字看似消融就当普通 baseline。
tuning 阶段只允许对应 bundle 的实际三值算术对照族和 base；候选三值族继续使用严格
child 路径。普通 tuning 控制会把已付费的祖先训练与本次墙钟上限一起计入相同 trial 额度。

执行 CLI 不变，但 request 改为只有 `node_id`：

```bash
python scripts/research.py qualify --job research/qualification-node-request.json \
  --bindings research/qualification-bindings.json --queue runs/multibench-v3 \
  --data artifacts/multibench-v3 --seconds "$QUALIFICATION_SECONDS"
```

对应 [request](../research/templates/qualification-node-request.template.jsonc) 与
[bindings](../research/templates/qualification-bindings.template.jsonc) 模板需从实际目录和协议补齐。
`QUALIFICATION_SECONDS` 是经审查且仍在原始授权内的有限上界，不是新预算。控制器由 node ID
构造实际 job；不能在 request 中注入训练设置、shell argv 或猜测 checkpoint。
返回的 `dependency_state_ref` 绑定真实 native 节点记录。后续节点需要将实际祖先记录合成
恰好包含全部祖先的状态文件，再引用该文件；不得把不存在的父节点标成 completed。
真实 candidate child 可以作为校准来源，普通控制依赖仍验证原始授权中的实际费用与身份。
`arithmetic_anchor_tuned` 还需要真实完整 tuning selection，不能手填“最佳 floor”。

候选产生的校准或候选别名必须使用
[calibration-child.fragment.template.jsonc](../research/templates/calibration-child.fragment.template.jsonc)，
放在 `bindings.protocol.suite_calibration_child`。源码为
[`suite_precursor.py`](../recursive_ssd/suite_precursor.py)。它保留实际 `method_discovery`、
当前 canonical 状态、冻结 child protocol、真实 native 回放和自己的完整设计验证批次。
字段为 `purpose="native-comparator-calibration"`、`suite_ref`、`bundle_id`、完整
`node_ids`、`node_roles`、逐节点 `node_seconds`、`config`、`calibration`、`tuning_budget`、
`execution_provenance`。development 的 `tuning_budget` 必须为 null；tuning 必须为实际
冻结的 `{trial_wall_seconds, source_refs}`，执行 bindings 保持相同值。
`node_roles[id]` 严格为 `{group, arm_role}`；所列节点及全部祖先
必须属于该方法自身，不能把另一个候选放进同一个 child 批准。

child 仍走上述 `qualify --job` CLI，使用
[child bindings 模板](../research/templates/calibration-child-bindings.template.jsonc)，
bindings 还必须提供完整 design batch
`verification_ref`、实际独占单 GPU 资源，以及有限 replay timeout/调用数；
峰值显存未知时，这个有界测量子协议允许 `gpu_peak_mib=null`、`memory_profile_ref=null`
并独占设备测量；已有实测 profile 则完整绑定。OOM 原样保留，主完整队列仍必须实测 profile。
`--seconds` 必须等于冻结的该节点上界。它预留整个 child 库存并逐节点记账，scope 为
`candidate-calibration-child`，费用和回放都来自原来同一预算。它的结果只能为真实依赖与
校准提供证据，不能自动赋予主比较 PASS。child protocol 的 `evidence_mode` 是 canonical
freeze 要求的 `prospective_confirmatory`，实际 native child 执行由控制器限定为
`developmental`；协议冻结身份与执行证据范围不能混淆。缺少任何科学前提依然阻塞，不能
补造通过记录。

这些入口是本轮**已生成、未执行**的代码；Local 需要先验证新源码、重新建立当前 code/design
绑定，再按实际证据运行。不能因有了 JSONC 模板就宣称完整 M03 已可运行或已完成。

## 4. Dispatch bindings 与现场成本

真实 CLI（只有上面证据和原始预算仍有效时）：

```bash
python scripts/research.py admit --queue runs/multibench-v3 --bundle M03 \
  --bindings research/m03-dispatch-bindings.json
```

| 字段 | 来源与要求 |
|---|---|
| `protocol_ref` | 真实 freeze 完成后文件的当前 SHA-256；草案哈希会过期 |
| `verification_ref` | 当前完整 design-verification batch；不能指向 `implementation-check.json` |
| `calibration_ref` | `suite-resource-calibration-v1`：实际 host_ref、当前绑定及完整节点 costs |
| `gpu_uuid` | 用户授权那张 RTX 2080 Ti 的实际 `GPU-...` UUID |
| `resources` | 实际有限 CPU/RAM、实测 `gpu_peak_mib` 和 `memory_profile_ref`；候选执行禁止共享 GPU |
| `groups` | 开发为 `humaneval/dev → he-dev`；确认要有 HE confirm、MBPP full、LCB full，并使用独立确认协议 |
| `arm_roles` | 所选 bundle 每个节点都能解析到契约中的同一 arm identity；模板列出全部 M03 映射 |
| `config`、`calibration` | 冻结的精确有效值；没有覆盖值也应明确 `{}` 并绑定其摘要，不能沿用模板 null |
| `replay_timeout_seconds` | 有限正数，不超过 3600；计入全组资源预留 |
| `replay_calls_per_node` | 从真实 qualifier 的 role 库存求和；faithful harness 的每个 role 计两次；不是固定猜测值 |
| `finalization_reserve_seconds` | 收尾预留；模板保留控制器默认 900 秒，不代表实际成本已测量 |

calibration 的 `bindings` 必须精确匹配 environment digest、model revision、code digest、
benchmark manifest；`host_ref` 必须有真实一张 2080 Ti 的观测和原始 evidence refs。
`costs` 需覆盖整个 bundle 的每个节点，值包含 `upper_seconds` 和真实完成的 native
`receipt_ref`。receipt 必须是被实现接受的真实科学 calibration 范围（baseline、普通
comparator 或审查后的 candidate child），完整匹配其 scope/节点/实现身份，界限不能低于
已观测耗时。把同一个无关 receipt 复制到所有节点并不能提供可信的成本模型。

## 5. frozen-m03.json 的正确生命周期

`protocol --spec` 写入 `output` 指定路径；即使路径叫 `frozen-m03.json`，这时也仍是
**草案**。然后 Local 才能在真实可用授权内执行：

```bash
python scripts/research.py freeze --protocol research/frozen-m03.json \
  --queue runs/multibench-v3 --replay-calls "$NATIVE_REPLAY_CALLS"
```

`NATIVE_REPLAY_CALLS` 来自上一节的实际资格库存。冻结会现场重放 native qualification、
检查真实 canonical prerequisites，并调用固定版本 skill 的锁定事务。只改 JSON 中的
`frozen_at` 没有效力。冻结后重新取得 protocol 引用，完成设计批次校验，再写 dispatch
bindings；最后 `admit` 仍会检查全部条件和整组成本。

确认必须由完整真实开发选择生成，使用新种子 23/47/71/101/131、固定 R3，分别绑定
`he-confirm`、`mbpp-full`、`lcb-full`；不能把开发模板的 group/seed 手改几项当作已完成
确认冻结。R5 和 0.5B 同样需要独立队列身份及适用证据。

此前讨论的原始窗口为 2026-10-06 16:00 英国夏令时起 8 小时，现已过去且没有观察到
实际 GPU 开始。上述文档不会重开预算。`status`、实际日志和返回包决定是否已有执行；
不得用模板完成、自动任务完成或历史 CPU 检查数代替 GPU 实验完成。

## 静态核对范围

本页逐字段对照了 `scripts/research.py`、`suite_admission.py` 中 native-assets/protocol/
qualification/freeze/calibration 入口、`suite_queue.py` 中 admit/qualification/resource/
replay inventory 逻辑、`suite_precursor.py` 新 child 接口、固定版本 native 与 gate-a schemas，
以及实际 development 目录。
本次只完成源代码与文档核对，没有新运行结果或新的 scientific PASS。
