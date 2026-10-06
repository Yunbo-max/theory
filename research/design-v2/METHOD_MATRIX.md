# 15 项方法：完整比较义务

共同的 benchmark、数据、统计、阈值、资源及 E04 要求见 [G01_EXPERIMENT_DESIGN.md](G01_EXPERIMENT_DESIGN.md)。此表是条件设计覆盖，不是 design_verified 或执行凭证。

| 方法 | 科学问题 | 必需基线 | 区别性/替代对照 | 证伪条件 |
|---|---|---|---|---|
| M03 / floor_projection | 保留被截断的概率质量，是否能保留可解题目的覆盖，并优于简单混合？ | hard, full_soft, arithmetic_anchor, lower_lr | floor_zero, arithmetic_anchor_tuned, full_soft_lr_half | 算术混合或较小步长解释全部收益；或者目标满足下限而拟合后的学生没有改善。 |
| M04 / temperature_budget | 累积温度变换是否足以解释递归漂移？ | full_soft, hard | temperature_fixed_untruncated, temperature_budget_untruncated, temperature_once_then_one | 只改温度就解释主候选收益，或根式日程在真实拟合中无收益；无截断恒等式不适用于截断实验。 |
| M05 / head_mass | 保留尾部质量和重塑头部形状，是否各有独立作用？ | full_soft, hard | soft_current_mix, head_shape_only, mass_transfer_only | 确定性软混合给出相同或更优原生指标；所谓头部结构作用消失。 |
| M01 / geometric_anchor | 几何锚定是否比同信息的算术锚定更有价值？ | full_soft, arithmetic_anchor | arithmetic_same_smoothing, geometric_weight_one | 优势只来自 epsilon 平滑/参考前向/更弱更新；算术组合在公平开发调参后不劣。 |
| M07 / tail_stratified | 精确头部加分层尾部，能否在任务质量相近时节省蒸馏成本？ | full_soft, hard | head_exact_tail_iid, fresh_hard | 只有局部方差差异，原生质量/成本没有优势；当前 dense 构造比 full_soft 更慢即不能声称高效。 |
| M02 / ratio_cap | 控制相对初始策略的比率，是否优于普通锚定？ | M01, arithmetic_anchor, full_soft | ratio_bound_unlimited, arithmetic_same_smoothing | 真实学生仍严重偏移或相同成本的简单锚定不劣；只能证明目标上界。 |
| M06 / head_diversity | 头部内部的多样性约束能否改善可解题覆盖？ | full_soft, M04 | head_retention_zero, temperature_matched_head | 只有熵/Gini 增大，native pass@10 无改善，或尾部丢失仍主导。 |
| M08 / temporal_mean | 历史教师能否减少有害轮间波动，而不只是减小更新？ | full_soft | full_soft_lr_half, temporal_current_twice, lag_only | 较小学习率解释全部收益，或陈旧教师损害正确率。 |
| M09 / disagreement_gate | 时间分歧是否提供了有用的位置级更新分配信息？ | full_soft, M08 | soft_current_mix, constant_mean_gate, full_soft_lr_half | 匹配平均更新强度的常量门不劣；分歧没有额外选择价值。 |
| M10 / prefix_damping | 抑制已经被放大的路径，是否优于统一减小 loss？ | full_soft | constant_mean_weight, full_soft_lr_half, prefix_weight_one | 同均值常量权重或较小学习率获得相同结果；路径依赖没有必要。 |
| M11 / noise_allocation | 固定局部噪声预算内，位置分配能否优于统一缩放？ | full_soft, fresh_alpha | fresh_alpha_budget_matched, fresh_hard | 相同预算的常量 a 或完整软目标更好；提高收益仅来自更大平均更新。 |
| M12 / gradient_projection | 方向修正是否比等幅缩步更能保留任务能力？ | full_soft_sgd | arithmetic_anchor_sgd, sgd_norm_matched, projection_disabled | SGD 对照不劣，或一阶约束在有限步中没有任务收益。 |
| M13 / kl_backtrack | 实际步长回溯是否值得它增加的前向与拒绝成本？ | full_soft | full_soft_lr_half, full_soft_lr_matched, backtrack_disabled | 更小学习率在同总时间内不劣；回溯拒绝耗尽预算；只保证检查批次上的 KL。 |
| M14 / prompt_allocation | 用自身首位置分布分配生成预算，是否比均匀采样更有效？ | hard | uniform_counts_weighted, allocation_without_weight_correction | 均匀分配不劣，或改善来自改变了题目权重而非方差分配。 |
| M15 / ancestral_exploration | 历史策略参与生成，是否比仅在目标中保留初始分布更有效？ | hard, arithmetic_anchor | ancestral_mix_zero, ancestral_soft_anchor, hard_equal_clock | 仅目标锚定或给 hard 相同总时间已解释收益；额外生成前向不划算。 |

每项还需完成最近方法比较、原生资格与依赖检查。控制项中的公式边界可复用已核验的完全等价臂，不能为了计数重复运行；数据/解码条件不同则必须单独记账。
