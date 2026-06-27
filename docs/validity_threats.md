# 效度威胁与对策清单 (Validity Threats & Controls)

对应计划第 9 节。每条威胁列出:对策、本仓库中落实的代码位置、以及对应的度量。

| 威胁 | 对策 | 代码落实 | 度量/检查 |
| --- | --- | --- | --- |
| 维度纠缠(改 S1 牵动长度/其他 O) | 操纵检查 + 长度控制;只允许目标维度变化 | `data/manipulation_check.py::check_pair`(`off_target_drift`, `length_ok`) | 操纵检查通过率 `batch_check`;配对漂移列表 |
| 位置/顺序偏差 (R) | 位置计数平衡(全排列/轮转) + 位置入协变量 | `task/protocol.py::build_candidate_sets(counterbalance="all_positions")`;回归含 `target_position` | `analysis/heterogeneity.py::position_bias`;`metrics.position_adjusted_rate` |
| 模型随机性 | 多 seed / 温度重复采样后聚合 | `experiment/runner.py`(`seeds`, `temperature`) | 跨 seed 选择率方差 |
| 冗长偏好 (verbosity bias) | 等长约束(±10%);长度作为协变量 | `manipulation_check.length_ok`;生成器锁定篇幅 | `length_ok` 通过率 |
| 自偏好 / 熟悉度 | 多源多样内容 + 合成/新颖主题防污染 | `data/generation.py::make_queries`(合成主题) | 跨模型一致性 `cross_model_consistency` |
| 生成器泄漏(造数据带入风格偏差) | 模板化受控生成 + 操纵检查 + 跨生成器交叉验证 | 模板路线 `build_article_text`;LLM 路线 `llm_edit_instruction` 二次校验 | 操纵检查通过率;人工抽检 |
| 混杂偏倚(观察数据 S→Y 被 Q/S 混杂) | do 后门调整(定理 1) + 实验 do 双路线交叉验证 | `causal/backdoor.py`;`causal/graph.py` 混杂集 | `metrics.do_route_consistency`(两路线 EI 差) |
| 解析失败(模型不按 JSON 输出) | 防御式解析 + 标记 `parse_ok` | `task/protocol.py::parse_decision` | `metrics.validity_report` 的 `parse_ok_rate` |
| 完美分离 / 不收敛(回归) | L2 正则回退;聚类稳健 SE | `analysis/regression.py::logit_with_clusters` | 系数表 `method` 列 |
| S→O 依赖导致不自然样本 | O 维度优先走 SCM 后门路线;O4 只在 S1 存在时操纵 | `studies/design.py::ofat_pairs`(O4 基于 S1=top) | 操纵检查 |

## 评估指标总览 (plan 第 8 节)

- 主指标 EI / 归一化 EI~ 及 determinism/degeneracy 分解: `analysis/metrics.py::ei_leverage_table`, `causal/ei.py`。
- 每维 ATE(方向/幅度): `analysis/ate.py::ate_table`(配对/边际)。
- 位置校正后的选择率: `analysis/metrics.py::position_adjusted_rate`。
- 跨模型一致性(Kendall W / Spearman / EI 方差): `analysis/metrics.py::cross_model_consistency`。
- 领域/提示敏感度(分层 EI): `studies/study3_generalization.py`。
- 反向: 欺骗增益 / 折扣 / 脆弱性 / ΔEI: `analysis/metrics.py::deception_gain`, `studies/study4_adversarial.py`。
- 有效性: do 两路线一致性、操纵检查通过率、解析率: `do_route_consistency`, `batch_check`, `validity_report`。
