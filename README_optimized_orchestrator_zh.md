# 多智能体群聊调度器 README（中文版）

## 1. 项目目标

本项目是一个面向多智能体群聊的最小 MVP 调度器。

它的职责是**只做调度**，不负责最终回复文本生成风格，也不负责上游的话题跟踪。
每一轮核心回答三个问题：

1. 这一轮应该由谁发言？
2. 谁担任 `main / supplement / check / summary` 角色？
3. 如何抑制重复、识别冲突信号，并决定是否触发 `summary / return / end`？

当前主文件：

- `demo_orchestrator_optimized.py`

相关运行 / 测试文件：

- `planner_regression_suite.py`
- `sample_cases.py`
- `qwen_runner.py`

---

## 2. 当前优化版做了什么

这一版把四个实验模块中真正有效的部分合并到了一个统一调度器里：

- `query_management`
- `main_tiebreak`
- `supplement_diversity`
- `trigger_boundary`

这样做的目的，是解决之前“每个分模块只能过自己那一部分 case，但没有任何一个版本能稳定扫完整套 case”的问题。

### 2.1 主要改动

#### A. Query 管理不再只盯当前用户一句话

优化版引入了：

- `context_query`
- `effective_query`
- `query_source`

它解决了一个很常见的问题：**用户不是每一轮都会发言**。

如果当前轮是“继续推进型”轮次，那么调度器可以继承上一轮上下文，而不是过度依赖“继续 / 展开 / 再说”这类过弱 query。

这样可以显著提升 continuation case 中的角色打分稳定性。

#### B. main 的近分 tie-break 显式化

以前多个 agent 在 `main` 上分数很接近时，容易只靠原始分数排序，导致不稳定翻转。

现在 main 的裁决链路变成：

1. `final_score`
2. `scene role prior`
3. `query_focus`
4. `topic_focus`
5. `less_recent`（repeat_penalty 更小）
6. 原始 agent_list 顺序

这样能避免近分场景里主答频繁翻转。

#### C. supplement 不再只是“第二高分”

现在补充角色会显式考虑：

- `diversity_from_main`
- 与主答的 `functional_fit`
- `topic_focus`
- `repeat_penalty`

也就是说，supplement 不再只是“下一个分高的人”，而是真正承担“补一个新角度”的角色。

#### D. trigger boundary 更清楚

优化版保留了更明确的边界，用来区分：

- `check_trigger`
- `summary_trigger`
- `topic_action = maintain | return | end`

这样可以减少之前常见的误触发：

- relation verify 提前触发
- 还没到该总结的时候就 pre-summary
- return / end 和普通调度混在一起

#### E. return / end 特殊路径与普通调度分离

现在两类特殊路径会直接走专门逻辑：

- `topic_action = return`：强制 summary，并把话题拉回上一个稳定主线
- `topic_action = end`：直接进入收束 / 告别逻辑

这样可以避免旧版本中“用户已经明显想结束，但系统还在继续推进话题”的问题。

---

## 3. 四个模块各自对应的职责

虽然当前已经统一实现，但原来的四个模块仍然对应四类明确职责。

### 3.1 `query_management`

作用：

- 决定这一轮真正该拿什么文本作为调度锚点
- 区分“用户本轮新输入”和“沿用上一轮上下文”

核心贡献：

- `effective_query`
- `query_source`
- silent / continuation round 处理

适合解决：

- 用户这一轮没有显式说话的 case
- 短弱 query 扰乱 relevance 打分的问题

### 3.2 `main_tiebreak`

作用：

- 解决主答候选 close score conflict

核心贡献：

- 显式 main tie-break 规则
- scene-aware role prior

适合解决：

- 两个 agent 都满足阈值且分数非常接近
- 不同运行间主答来回翻转的问题

### 3.3 `supplement_diversity`

作用：

- 让 supplement 和 main 分工更清楚，避免重复

核心贡献：

- `diversity_from_main`
- `functional_fit`

适合解决：

- supplement 实际上在重复 main
- 一轮里出现“两个 main”而不是“main + supplement”

### 3.4 `trigger_boundary`

作用：

- 决定什么时候触发 check、summary、return、end

核心贡献：

- 普通调度路径与特殊控制路径更清晰
- relation check / redundancy check / pre-summary / return / end 分界更明确

适合解决：

- check / summary 误触发
- 路径覆盖不稳定
- 用户结束意图未被正确识别

---

## 4. 为什么现在要统一到底层共享实现

如果四个文件继续各自独立维护，会出现两个问题：

1. **逻辑漂移**：一个模块修了 query，另一个模块修了 tie-break，但这些修复不会同时生效
2. **case 不一致**：每个分支能过自己那部分 case，却无法稳定扫完整套 case

所以当前策略是：

- 保留四个思路的概念分工
- 把真正有效的逻辑合并到底层统一实现
- 用统一文件作为实际回归目标

这样更容易：

- 调试
- 冻结版本
- 做组会汇报
- 写 README 和实验记录

---

## 5. 当前分数与阈值

当前优化版使用的主要阈值如下：

- `min_main_relevance = 0.18`
- `min_main_score = 0.18`
- `min_supplement_relevance = 0.14`
- `min_supplement_score = 0.14`
- `medium_redundancy_threshold = 0.45`
- `high_redundancy_threshold = 0.52`
- `conflict_similarity_threshold = 0.22`
- `pre_summary_points_threshold = 3`
- `summary_points_threshold = 4`
- `repeat_penalty_value = 0.08`
- `tie_margin_main = 0.02`
- `tie_margin_supplement = 0.02`
- `close_score_delta_main = 0.02`
- `close_score_delta_supplement = 0.02`
- `close_query_focus_delta = 0.015`
- `close_topic_focus_delta = 0.015`
- `close_diversity_delta = 0.05`
- `functional_main_fallback_floor = 0.14`
- `functional_main_fallback_delta = 0.05`

### 5.1 Agent 最终分数

当前 agent 分数主要由以下部分组成：

- 与轮次锚点的 relevance
- scene bonus
- character role bonus
- 如果上一轮刚说过，则减去 repeat penalty

公式可以概括为：

`final_score = relevance + scene_bonus + character_role_bonus - repeat_penalty`

### 5.2 Main 选择

main 候选先过阈值过滤，然后在近分场景下使用显式 tie-break。

### 5.3 Supplement 选择

supplement 候选先过阈值过滤，然后综合：

- final score
- 与 main 的 functional fit
- 与 main 的 diversity
- topic focus

---

## 6. 当前效果

当前回归结果：

- 共 17 个 case
- 共 137 个 checks
- 当前版本全部通过

说明当前统一版已经能够：

- 稳定跑完整套 case
- 同时覆盖 `main / supplement / check / summary / skip / return`
- 不再像之前那样“某个模块修好一个问题，却把另外几个 case 弄坏”

---

## 7. 如何扫全部 case

下面给出 PowerShell 下可以直接运行的命令。

### 7.1 扫全部 case，并导出 json 和 txt

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_optimized.py --json-output outputs\optimized_regression_report.json --txt-output outputs\optimized_regression_report.txt
```

### 7.2 只看总览，不打印每个 case 细节

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_optimized.py --quiet --json-output outputs\optimized_regression_report.json --txt-output outputs\optimized_regression_report.txt
```

### 7.3 如果你想扫某个单模块文件

例如扫 `query_management`：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_query_management.py --json-output outputs\query_management_report.json --txt-output outputs\query_management_report.txt
```

例如扫 `main_tiebreak`：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_main_tiebreak.py --json-output outputs\main_tiebreak_report.json --txt-output outputs\main_tiebreak_report.txt
```

例如扫 `supplement_diversity`：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_supplement_diversity.py --json-output outputs\supplement_diversity_report.json --txt-output outputs\supplement_diversity_report.txt
```

例如扫 `trigger_boundary`：

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
python planner_regression_suite.py --orchestrator-module demo_orchestrator_trigger_boundary.py --json-output outputs\trigger_boundary_report.json --txt-output outputs\trigger_boundary_report.txt
```

---

## 8. 导出结果说明

执行后会得到两个文件：

- `*.json`：完整结构化结果，适合后续分析、可视化、做 sweep 对比
- `*.txt`：文本摘要报告，适合直接查看和汇报

推荐保留方式：

- 每次冻结一版代码，就同时保存一份 `json + txt`
- 文件名带上版本号或日期，便于后续和 baseline 对比

例如：

- `outputs/freeze_v1_regression.json`
- `outputs/freeze_v1_regression.txt`

---

## 9. 相比 baseline 的主要提升

相对最初 baseline，这一版主要增加了：

1. continuation round 的 query 继承能力
2. main 近分冲突的显式 tie-break
3. supplement 的 diversity / functional fit 机制
4. check / summary / return / end 的边界拆分
5. 更完整的 case 校验项，包括：
   - `preferred_main_any`
   - `preferred_supplement_any`
   - `forbid_main`

因此这版不只是“规则更多”，而是：

- 调度更稳定
- case 通过率更高
- 解释性更强
- 更适合冻结为当前实验版本

---

## 10. 还可以继续优化的点

虽然现在已经能稳定跑全套 case，但后面还可以继续优化：

### 10.1 语义冲突检测仍然比较轻量

当前 check 层更多还是基于 relation / overlap / redundancy 的轻量判断，后面可以继续加强真正的“语义冲突 / 立场冲突”检测。

### 10.2 当前仍然是规则主导

这符合最小 MVP 阶段目标，但后续如果要做更复杂场景，可以考虑：

- 更细的动态阈值
- 更细的用户意图继承
- 更强的 topic shift / return 识别

### 10.3 目前 case 集仍然有限

虽然已经能覆盖主路径，但后续可以继续补：

- 多轮 return case
- 多轮高冗余收束 case
- 用户主动切题 case
- check 与 summary 连续触发 case

---

## 11. 当前建议的使用方式

现阶段建议：

- 用 `demo_orchestrator_optimized.py` 作为主实验文件
- 四个分模块保留，作为逻辑来源和汇报说明
- 每次改动后都跑完整套 case，并导出 `json + txt`
- 当通过结果稳定后，及时冻结版本

