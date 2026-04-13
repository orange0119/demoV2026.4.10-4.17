# demoV2026.4.10-4.17
多方对话最小 MVP 调度器 README（中文版更新稿）
1. 项目定位
本项目实现的是一个多智能体群聊调度器最小 MVP。它不直接负责生成最终文案风格，而是负责在每一轮对话中决定：
谁担任主答（main）
谁担任补充（supplement）
是否需要进入检查（check）
是否应该做总结（summary）
哪些角色需要跳过（skip / mute）
最终给下游生成模块什么执行计划（final_plan / execution_steps）
该版本重点解决的是：
多角色发言时的主答 / 补充分工
近分候选的tie-break 稳定选择
结合任务类型与话题严肃程度的角色优先级调度
下游生成时的回复对象定向控制
在不破坏原有 benchmark 的前提下，增加 task-type 定向测试覆盖
---
2. 本次更新的核心目标
本轮更新主要围绕两个问题展开：
当多个 agent 分数接近时，不能只靠 `final_score / query_focus / topic_focus` 来做选择，而需要进一步结合：
任务类型（task_type）
当前话题严肃程度（topic_seriousness）
调度层不仅要决定“谁说”，还要为生成层提供更明确的对象约束，避免输出变成“谁都像在对空气说话”。因此在 `execution_steps` 中新增：
`address_to`
`address_style`
`opening_hint`
换句话说，这一版的核心思想是：
> **调度不仅决定角色顺序，还要提前规定“这一句主要是说给谁听、应该怎么开口”。**
---
3. 本次更新后新增的关键能力
3.1 显式任务类型识别
系统当前支持以下 7 类任务类型：
`practical_help`：实际求助
`relationship_reading`：关系揣测 / 对方态度判断
`decision_support`：该不该、要不要、怎么选
`emotional_support`：情绪支持
`clarification`：澄清 / 追问 / 纠偏
`open_discussion`：开放讨论 / 观点交流
`light_trouble`：轻麻烦 / 生活小故障
本次更新后，任务类型不再只是生成层参考信息，而是直接进入调度逻辑。
---
3.2 引入话题严肃程度（topic_seriousness）
为了让不同任务类型下的角色选择更稳定，本版将话题严肃程度显式分为：
`high`
`medium`
`low`
它的作用不是单独替代任务类型，而是和任务类型一起控制：
主答是否应该更稳、更能接住用户
补充是否应该继续面向用户，而不是只对 main 说话
opening hint 应该是“先共情”还是“先拆问题”还是“先直接观点回应”
---
3.3 main / supplement 的 tie-break 分开处理
本次更新前，近分冲突大多沿用统一排序逻辑；本次更新后，主答与补充的 tie-break 彻底分开。
主答（main）
主答的近分裁决逻辑为：
`final_score`
`task_seriousness_fit`
`query_focus`
`topic_focus`
`less_recent`
`agent_list_order`
也就是说，主答优先考虑“谁更适合在这个任务类型与严肃程度下承担第一发言”。
补充（supplement）
补充的近分裁决逻辑为：
`final_score`
`task_seriousness_fit`
`diversity_from_main`
`topic_focus`
`less_recent`
`agent_list_order`
也就是说，补充不仅要“能说”，还要“和 main 形成互补”。
---
3.4 execution_steps 增强：把“回复对象”写进计划
本版在 `execution_steps` 中显式输出：
`task_type`
`topic_seriousness`
`address_to`
`address_style`
`opening_hint`
示意如下：
```json
{
  "agent_id": "leonard",
  "role": "main",
  "task_type": "decision_support",
  "topic_seriousness": "medium",
  "address_to": ["user"],
  "address_style": "direct_user_solution",
  "opening_hint": "先替用户把选择问题框清楚，例如：‘这件事其实可以先拆成两个判断。’"
}
```
这样做的价值在于：
planner 不再只输出“角色”，而是输出“角色 + 面向对象 + 开头方式”
qwen 之类的下游生成模型更容易产出像下面这种更自然的发言：
`Penny -> 用户：我知道，你现在是真的很累。`
`Leonard -> 用户：这件事其实可以先拆成两个判断。`
`Sheldon -> 用户：这里的关键不是 A，而是 B。`
---
4. 任务类型与角色优先级的设计思路
4.1 main 的优先角色
当前版本的设计倾向如下：
`emotional_support`：Penny / Leonard 优先做 main
`relationship_reading`：Leonard / Penny 优先做 main
`decision_support`：Leonard / Sheldon 优先做 main
`practical_help`：Leonard / Sheldon 优先做 main
`clarification`：Sheldon / Leonard 优先做 main
`open_discussion`：Leonard / Sheldon 优先做 main
`light_trouble`：Penny / Leonard 优先做 main
4.2 supplement 的优先原则
补充位不是简单选择“第二高分”，而是强调：
是否与 main 的角色功能互补
是否能从不同角度补一个新增点
在高严肃度任务中，是否还应该继续对用户说话
因此，supplement 会同时参考：
`task_seriousness_fit`
`functional_fit`
`diversity_from_main`
---
5. 本轮修复的主要问题
5.1 修复 task type 分类边界错误
在 task-type 定向 case 中，之前主要暴露了几类错误：
`relationship_reading` 被误判成 `practical_help`
`open_discussion` 被误判成 `practical_help`
`light_trouble` 被误判成 `emotional_support`
`decision_support` 虽然识别正确，但主答优先级不够硬，Penny 仍可能抢主答
本轮修复中，重点修改了：
task type 判定顺序
`practical_help` 的匹配边界
`light_trouble` 的前置与增强
`decision_support` 下 main 的 task-fit 权重
---
5.2 修复 opening_hint 没有真正喂给生成层的问题
之前 planner 虽然已经在 `execution_steps` 中计算出了：
`address_style`
`opening_hint`
但这些字段并未完整进入下游生成 prompt。
本轮修复后，`qwen_runner.py` 会将 `opening_hint` 一并带入 prompt，使生成层真正感知：
应该先接情绪
还是先框架化拆分问题
还是先直接回应观点核心
---
5.3 修复轻麻烦场景的风格不自然问题
新增 `light_trouble` 后，本轮进一步把它从泛化的 `default` 风格中独立出来，使其更适合：
轻接用户的小麻烦
不把所有轻微问题都提升成高强度情绪支持
让表达更加贴近日常交流而不是过度分析
---
6. 本轮回归结果
6.1 基准 case 集
原始基准集共 17 个 case，更新后仍保持：
`passed_cases = 17 / 17`
`passed_checks = 137 / 137`
路径覆盖完整，无缺失必需路径
6.2 task-type 扩展 case 集
扩展后共 24 个 case。本轮修复后已经达到：
`passed_cases = 24 / 24`
`passed_checks = 241 / 241`
路径覆盖完整，无缺失必需路径
这意味着：
旧 benchmark 没有被破坏
新增的 task type / seriousness / address_to / opening_hint 机制已经跑通
---
7. 当前冻结版包含的关键文件
建议冻结以下文件作为当前稳定版本：
`demo_orchestrator_optimized.py`
`demo_orchestrator_main_tiebreak.py`
`demo_orchestrator_query_management.py`
`demo_orchestrator_supplement_diversity.py`
`demo_orchestrator_trigger_boundary.py`
`qwen_runner.py`
`planner_regression_suite.py`
`planner_regression_suite_tasktype.py`
`sample_cases.py`
`sample_cases_tasktype.py`
建议同时保留以下报告文件作为冻结证据：
`optimized_task_seriousness_report_v2.json`
`optimized_task_seriousness_report_v2.txt`
`tasktype_case_report_after_patch2.json`
`tasktype_case_report_after_patch2.txt`
---
8. 推荐运行命令
8.1 扫原始 17-case 基准集
```powershell
python planner_regression_suite.py --orchestrator-module demo_orchestrator_optimized.py --json-output outputs\optimized_task_seriousness_report_v2.json --txt-output outputs\optimized_task_seriousness_report_v2.txt
```
8.2 扫扩展后的 24-case task-type 集
```powershell
python planner_regression_suite_tasktype.py --orchestrator-module demo_orchestrator_optimized.py --sample-cases-module sample_cases_tasktype.py --json-output outputs\tasktype_case_report_after_patch2.json --txt-output outputs\tasktype_case_report_after_patch2.txt
```
8.3 联动 qwen 下游生成
```powershell
python qwen_runner.py --mode multi --case all --rounds 3 --model qwen-plus --orchestrator-module demo_orchestrator_optimized.py --output outputs\qwen_tasktype_run.json
```
---
9. 这一版的意义
和最初只做“谁分数高谁先说”的调度器相比，这一版已经从单纯排序器，升级成了一个更完整的 面向任务类型的多角色调度器。
它的提升主要体现在：
能区分不同任务类型下，谁应该做主答
能区分不同严肃程度下，补充应该继续面向用户还是承接 main
能把“回复对象”和“开头方式”提前写进执行计划
能在不破坏原有 benchmark 的前提下，通过扩展 case 验证新增机制是否真正生效
---
10. 下一步可继续做的方向
虽然这版已经适合冻结，但后续仍可继续扩展：
把 task type 从规则版升级为“规则 + 轻量分类器”混合版
让 `topic_seriousness` 不只依赖 task type，而加入上下文强度修正
增加更多 addressee / opening 风格 case
把 `execution_steps` 再向下游 prompt 模板标准化
增加更多“近分但任务不同”的回归 case，用于进一步验证 tie-break 稳定性
---
11. 版本建议命名
建议将当前版本命名为：
freeze_tasktype_v1
如果需要在 Git 中冻结，推荐提交信息：
```bash
freeze tasktype v1: task-aware tie-break + directed execution steps stable
```
