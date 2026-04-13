from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


CaseDict = Dict[str, Any]
AgentDict = Dict[str, Any]


ROLE_MAIN = "main"
ROLE_SUPPLEMENT = "supplement"
ROLE_CHECK = "check"
ROLE_SUMMARY = "summary"

CHECK_REASON_RELATION = "relation_verify"
CHECK_REASON_PRE_SUMMARY = "pre_summary_verify"
SUMMARY_REASON_REDUNDANCY = "high_redundancy"
SUMMARY_REASON_TOPIC_RETURN = "topic_return"
SUMMARY_REASON_TOPIC_END = "end_conversation"


def get_shared_agents() -> List[AgentDict]:
    """
    公共固定角色：Sheldon / Leonard / Penny。
    这里保留最小但可用的人设字段，直接适配 demo_orchestrator.py。
    """
    return [
        {
            "agent_id": "sheldon",
            "name": "Sheldon",
            "status": "online",
            "keywords": ["逻辑", "定义", "解释", "纠偏", "规则", "分析", "结构化建议"],
            "description": "理性、强解释欲、喜欢把事情讲清楚，擅长逻辑分析、澄清概念和指出漏洞。",
            "persona": "遇到分歧时会优先定义问题、拆分条件、纠正不够准确的说法。",
            "style": "直接、清楚、结构化，情绪支持场景里不适合抢第一个主答。",
        },
        {
            "agent_id": "leonard",
            "name": "Leonard",
            "status": "online",
            "keywords": ["平衡", "缓冲", "总结", "温和建议", "衔接", "关系", "阶段收束"],
            "description": "平衡型、缓冲型角色，能承接不同观点，并把讨论往中间收。",
            "persona": "会先接住别人的话，再给温和、可执行的建议，也适合做阶段总结。",
            "style": "自然、温和、不抢戏，适合做 supplement 和 summary。",
        },
        {
            "agent_id": "penny",
            "name": "Penny",
            "status": "online",
            "keywords": ["生活感", "情绪回应", "现实建议", "体验", "直觉", "人际感受"],
            "description": "生活化、反应快、情绪感强，擅长把抽象问题拉回直观体验和现实感受。",
            "persona": "更容易先回应情绪，再给一个接地气的小建议。",
            "style": "口语化、自然、接地气，知识型话题里更适合补现实视角。",
        },
    ]


def get_functional_agents() -> List[AgentDict]:
    """
    用于路径覆盖测试：
    - Sheldon 显式承担 check
    - Leonard 显式承担 summary
    仍然保持三个人设不变，只是补上功能标签，方便触发 check / summary / return / end 路径。
    """
    agents = copy.deepcopy(get_shared_agents())
    for item in agents:
        if item["agent_id"] == "sheldon":
            item["can_check"] = True
        if item["agent_id"] == "leonard":
            item["can_summarize"] = True
    return agents


def _make_case(
    case_id: str,
    title: str,
    target_behavior: str,
    query: str,
    topic_state: Dict[str, Any],
    expected: Dict[str, Any],
    history_summary: Optional[Dict[str, Any]] = None,
    last_round_outputs: Optional[List[Dict[str, Any]]] = None,
    context_query: str = "",
    agent_list: Optional[List[AgentDict]] = None,
    group: str = "public",
) -> CaseDict:
    return {
        "case_id": case_id,
        "title": title,
        "group": group,
        "target_behavior": target_behavior,
        "query": query,
        "context_query": context_query,
        "agent_list": copy.deepcopy(agent_list if agent_list is not None else get_shared_agents()),
        "topic_state": copy.deepcopy(topic_state),
        "history_summary": copy.deepcopy(history_summary or {"done_points": [], "resolved": False}),
        "last_round_outputs": copy.deepcopy(last_round_outputs or []),
        "expected": copy.deepcopy(expected),
    }


def get_public_cases(agent_list: Optional[List[AgentDict]] = None) -> List[CaseDict]:
    """
    这 6 个是原本公共主线 case，保持不改，只作为公共 benchmark。
    """
    agents = copy.deepcopy(agent_list if agent_list is not None else get_shared_agents())
    cases: List[CaseDict] = []

    cases.append(
        _make_case(
            case_id="pc01",
            title="最近很累，只想躺着，什么都不想做",
            target_behavior="情绪/疲惫场景里，主答优先 Penny 或 Leonard，避免 Sheldon 抢第一个主答。",
            query="我这两天真的特别累，下班以后只想躺着，什么都不想做。",
            context_query="用户在表达疲惫和低能量，当前只讨论怎么调整状态，不要漂到空泛人生鸡汤。",
            topic_state={
                "main_topic": "疲惫时如何调整状态",
                "active_topic": ["疲惫", "下班后恢复", "状态调整", "休息与效率平衡"],
                "topic_summary": "围绕疲惫、恢复和温和建议展开。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "forbid_main": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="public",
        )
    )

    cases.append(
        _make_case(
            case_id="pc02",
            title="我总是拖延，明知道要做还是会刷手机",
            target_behavior="生活习惯场景里，主答优先 Leonard 或 Sheldon，Penny 更适合做现实补充。",
            query="我明知道有很多事要做，但就是会一直刷手机拖着不动。",
            context_query="用户在聊拖延和刷手机，当前只讨论拖延机制、启动成本和现实可做的小办法。",
            topic_state={
                "main_topic": "拖延与刷手机",
                "active_topic": ["拖延", "刷手机", "注意力管理", "行动启动"],
                "topic_summary": "围绕拖延、分心和具体做法展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["leonard", "sheldon"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="public",
        )
    )

    cases.append(
        _make_case(
            case_id="pc03",
            title="和朋友闹别扭，不知道要不要先联系",
            target_behavior="关系/情绪场景里，主答优先 Penny 或 Leonard，Sheldon 更适合分析补充。",
            query="我跟朋友闹得有点僵，现在不知道该不该我先去联系他。",
            context_query="用户在犹豫要不要主动联系朋友，先接住情绪和犹豫，再给关系修复建议。",
            topic_state={
                "main_topic": "朋友关系僵住后是否主动联系",
                "active_topic": ["朋友矛盾", "是否先联系", "关系修复", "表达方式"],
                "topic_summary": "围绕关系修复和主动联系的利弊展开。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "forbid_main": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="public",
        )
    )

    cases.append(
        _make_case(
            case_id="pc04",
            title="努力和天赋哪个更重要",
            target_behavior="观点讨论场景里，主答优先 Sheldon 或 Leonard，Penny 补直观经验视角。",
            query="你们觉得努力和天赋，到底哪个更重要？",
            context_query="围绕努力与天赋的比较展开，允许出现定义澄清和平衡性总结，但不要跑题。",
            topic_state={
                "main_topic": "努力 vs 天赋",
                "active_topic": ["努力", "天赋", "长期积累", "起点差异", "观点比较"],
                "topic_summary": "保持在努力与天赋的比较主线上。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
                "allow_pre_summary": True,
            },
            expected={
                "preferred_main_any": ["sheldon", "leonard"],
                "preferred_supplement_any": ["penny"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [],
                "min_speakers_gte": 2,
                "max_speakers_lte": 3,
            },
            agent_list=agents,
            group="public",
        )
    )

    cases.append(
        _make_case(
            case_id="pc05",
            title="AI 会不会取代很多工作",
            target_behavior="公共讨论里 Sheldon 应该较稳定地主答，Leonard / Penny 作为补充而不是缺席。",
            query="AI 发展这么快，你们觉得以后会不会取代很多人的工作？",
            context_query="围绕 AI 是否改变就业和工作结构展开，不依赖最新新闻细节，只讨论一般趋势和现实影响。",
            topic_state={
                "main_topic": "AI 与工作替代",
                "active_topic": ["AI", "工作替代", "职业变化", "现实选择"],
                "topic_summary": "讨论 AI 对工作和职业结构的影响。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "check_trigger": False,
                "preferred_main_any": ["sheldon"],
                "preferred_supplement_any": ["leonard", "penny", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 2,
            },
            agent_list=agents,
            group="public",
        )
    )

    cases.append(
        _make_case(
            case_id="pc06",
            title="短视频和 AI 工具是不是让人更难专注",
            target_behavior="多角色最好形成体验—分析—收束链路，主答优先 Leonard 或 Sheldon。",
            query="现在短视频加上各种 AI 工具，我感觉大家越来越难静下心来专注了。",
            context_query="围绕专注力变化、技术使用方式和主观体验展开，适合平衡分析，不要漂到行业八卦。",
            topic_state={
                "main_topic": "短视频与 AI 对专注的影响",
                "active_topic": ["短视频", "AI 工具", "专注力", "注意力切换", "生活方式"],
                "topic_summary": "围绕专注力变化和技术使用习惯展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
                "allow_pre_summary": True,
            },
            expected={
                "preferred_main_any": ["leonard", "sheldon"],
                "preferred_supplement_any": ["penny", "sheldon", "leonard"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 3,
            },
            agent_list=agents,
            group="public",
        )
    )

    return cases


def get_variant_cases(agent_list: Optional[List[AgentDict]] = None) -> List[CaseDict]:
    """
    在不改原 6 个 case 的前提下，补充语义近似但表述不同的变体。
    这些主要测主答/补充稳定性、近分翻转和角色分化是否足够稳。
    """
    agents = copy.deepcopy(agent_list if agent_list is not None else get_shared_agents())
    cases: List[CaseDict] = []

    cases.append(
        _make_case(
            case_id="pv01",
            title="累但一休息就有负罪感",
            target_behavior="情绪场景变体里，主答仍应优先 Penny 或 Leonard，而不是 Sheldon 抢主答。",
            query="我其实知道自己该休息，可是一躺下又会觉得今天什么都没做，很有负罪感。",
            context_query="用户在疲惫之外又提到了负罪感，当前仍属于情绪支持和状态调整，不要直接变成纯效率 lecture。",
            topic_state={
                "main_topic": "疲惫与休息负罪感",
                "active_topic": ["疲惫", "休息负罪感", "状态调整", "效率焦虑"],
                "topic_summary": "先接住情绪，再谈怎么调整状态。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "forbid_main": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="variant",
        )
    )

    cases.append(
        _make_case(
            case_id="pv02",
            title="拖延变体：短视频一刷就停不下来",
            target_behavior="拖延变体里，主答仍应优先 Leonard 或 Sheldon，避免 Penny 抢成主答。",
            query="我不是完全不想做事，就是一打开短视频就停不下来，原本只想刷五分钟。",
            context_query="当前还是拖延和注意力管理问题，只是具体诱因变成了短视频连刷。",
            topic_state={
                "main_topic": "短视频导致的拖延",
                "active_topic": ["短视频", "拖延", "注意力滑坡", "启动成本"],
                "topic_summary": "围绕拖延机制和现实干预办法展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="variant",
        )
    )

    cases.append(
        _make_case(
            case_id="pv03",
            title="想和朋友和好，但又怕显得自己先认输",
            target_behavior="关系修复变体里，主答仍应优先 Penny 或 Leonard，Sheldon 更适合分析性补充。",
            query="我其实想跟朋友和好，但又有点别扭，怕我先开口会显得像我认输了。",
            context_query="用户在关系修复里掺杂了面子和别扭，先接住情绪，再谈怎么表达比较自然。",
            topic_state={
                "main_topic": "朋友矛盾中的主动表达",
                "active_topic": ["朋友和好", "先开口", "面子", "关系修复"],
                "topic_summary": "围绕关系修复和怎么主动表达展开。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "forbid_main": ["sheldon"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "must_not_have_roles": [ROLE_CHECK, ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="variant",
        )
    )

    cases.append(
        _make_case(
            case_id="pv04",
            title="如果天赋一般，努力还有没有意义",
            target_behavior="观点讨论变体里，主答仍应优先 Sheldon 或 Leonard，Penny 更适合作为现实经验补充。",
            query="如果一个人天赋真的很一般，那他再怎么努力还有意义吗？",
            context_query="围绕天赋一般时努力是否仍有价值展开，允许定义澄清和平衡型回答。",
            topic_state={
                "main_topic": "天赋一般时努力的意义",
                "active_topic": ["天赋一般", "努力价值", "成长空间", "现实判断"],
                "topic_summary": "围绕努力的意义和天赋边界展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
                "allow_pre_summary": True,
            },
            expected={
                "preferred_main_any": ["sheldon", "leonard"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 3,
            },
            agent_list=agents,
            group="variant",
        )
    )

    cases.append(
        _make_case(
            case_id="pv05",
            title="AI 是先替代重复工作，还是先改变人的工作方式",
            target_behavior="AI 讨论变体里，Sheldon 仍应较稳定地主答，Leonard / Penny 作为补充而不是缺席。",
            query="你们觉得 AI 更可能先替代哪类工作，还是先重构人的工作方式？这两者的区别到底在哪？",
            context_query="围绕 AI 对工作方式和职业结构的影响展开，优先分析“替代”和“重构”的区别与机制，不需要引用最新事实。",
            topic_state={
                "main_topic": "AI 对工作的改变方式",
                "active_topic": ["重复工作", "工作方式重构", "职业结构", "机制分析", "概念区分"],
                "topic_summary": "讨论 AI 到底是替代工作还是重构工作方式，先做概念区分和机制分析。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "check_trigger": False,
                "summary_trigger": False,
                "preferred_main_any": ["sheldon"],
                "preferred_supplement_any": ["leonard", "penny", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 2,
            },
            agent_list=agents,
            group="variant",
        )
    )

    cases.append(
        _make_case(
            case_id="pv06",
            title="AI 工具变多以后，我反而更懒得自己想了",
            target_behavior="专注/工具使用变体里，主答仍应优先 Leonard 或 Sheldon，Penny 提供体验型补充。",
            query="我现在越来越依赖 AI 帮我整理东西，可是久了以后反而更懒得自己想了。",
            context_query="围绕 AI 工具使用、思考惰性和专注方式变化展开，不要漂到行业八卦。",
            topic_state={
                "main_topic": "AI 工具对思考与专注的影响",
                "active_topic": ["AI 工具依赖", "思考惰性", "专注变化", "使用方式"],
                "topic_summary": "讨论 AI 工具是否让人更难保持主动思考和专注。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
                "allow_pre_summary": True,
            },
            expected={
                "preferred_main_any": ["leonard", "sheldon"],
                "preferred_supplement_any": ["penny", "leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 3,
            },
            agent_list=agents,
            group="variant",
        )
    )

    return cases



def get_task_type_cases(agent_list: Optional[List[AgentDict]] = None) -> List[CaseDict]:
    """
    新增 task_type / seriousness / address 行为测试。
    这组 case 不替换原公共 benchmark，而是专门验证：
    1) task_type 是否按规则命中
    2) seriousness 是否稳定输出
    3) main / supplement 的 address_to、address_style、opening_hint 是否符合预期
    """
    agents = copy.deepcopy(agent_list if agent_list is not None else get_shared_agents())
    cases: List[CaseDict] = []

    cases.append(
        _make_case(
            case_id="tt01",
            title="practical_help：执行建议型求助",
            target_behavior="命中 practical_help，严肃度为 medium；主答应直接面向用户给方案，补充者延续主答。",
            query="我最近状态有点乱，很多事情堆着不动，我现在到底该怎么办？",
            context_query="用户在求一个可执行的起步办法，当前重点是先给现实可做的第一步。",
            topic_state={
                "main_topic": "状态乱时如何启动行动",
                "active_topic": ["状态混乱", "行动启动", "第一步怎么做"],
                "topic_summary": "围绕现实可执行的第一步展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "practical_help",
                "expected_main_seriousness": "medium",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "direct_user_solution",
                "expected_main_opening_hint_contains": "可执行方向",
                "expected_supplement_task_type": "practical_help",
                "expected_supplement_seriousness": "medium",
                "expected_supplement_address_style": "continue_from_main",
                "expected_supplement_opening_hint_contains": "先接前一位的话",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt02",
            title="relationship_reading：关系揣测与高严肃度",
            target_behavior="命中 relationship_reading，严肃度为 high；主答应优先 Leonard / Penny 并直接对用户安抚判断。",
            query="他最近突然变得很冷淡，是不是在针对我，还是故意想疏远我？",
            context_query="用户在揣测对方态度，当前先接住感受，再判断这种冷淡可能意味着什么。",
            topic_state={
                "main_topic": "如何理解对方突然冷淡",
                "active_topic": ["冷淡", "是否针对我", "关系判断", "如何理解态度"],
                "topic_summary": "围绕关系揣测与感受安抚展开。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["leonard", "penny"],
                "forbid_main": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "relationship_reading",
                "expected_main_seriousness": "high",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "direct_user_empathy",
                "expected_main_opening_hint_contains": "感受做判断和安抚",
                "expected_supplement_task_type": "relationship_reading",
                "expected_supplement_seriousness": "high",
                "expected_supplement_address_style": "support_user_after_main",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt03",
            title="decision_support：选择支持",
            target_behavior="命中 decision_support，严肃度为 medium；主答先框选择维度，补充者继续接主答。",
            query="我现在拿到两个 offer，到底该选哪个，还是再等等看？",
            context_query="用户在做选择，需要先把判断维度拆出来，再给建议。",
            topic_state={
                "main_topic": "offer 选择",
                "active_topic": ["选哪个", "等待还是决定", "判断维度"],
                "topic_summary": "围绕选择支持和判断维度展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "decision_support",
                "expected_main_seriousness": "medium",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "direct_user_solution",
                "expected_main_opening_hint_contains": "拆成两个判断",
                "expected_supplement_task_type": "decision_support",
                "expected_supplement_seriousness": "medium",
                "expected_supplement_address_style": "continue_from_main",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt04",
            title="emotional_support：心动但不确定",
            target_behavior="命中 emotional_support，严肃度为 high；main 和 supplement 都更像在对用户说话。",
            query="我好像喜欢上了一个人，但是我又不太确定，这种感觉到底算不算喜欢？",
            context_query="用户在表达心动和不确定，当前先接住情绪，再帮助他确认自己的感受。",
            topic_state={
                "main_topic": "确认是不是喜欢",
                "active_topic": ["喜欢一个人", "是否心动", "确认感受"],
                "topic_summary": "围绕心动与不确定展开。",
                "topic_action": "maintain",
                "scene_mode": "emotion_support",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "forbid_main": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "emotional_support",
                "expected_main_seriousness": "high",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "direct_user_empathy",
                "expected_main_opening_hint_contains": "接住用户的情绪",
                "expected_supplement_task_type": "emotional_support",
                "expected_supplement_seriousness": "high",
                "expected_supplement_address_to_all": ["user"],
                "expected_supplement_address_style": "support_user_after_main",
                "expected_supplement_opening_hint_contains": "承接 main 对用户的回应",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt05",
            title="clarification：澄清追问",
            target_behavior="命中 clarification，严肃度为 high；Sheldon 更适合主答，先回应问题核心。",
            query="你是说我其实不该立刻回他吗？为什么会这样判断，不是已经解释过了吗？",
            context_query="用户在追问理由并要求澄清，当前先把判断核心讲清楚。",
            topic_state={
                "main_topic": "澄清是否应立刻回复",
                "active_topic": ["为什么这样判断", "澄清核心", "是否立刻回复"],
                "topic_summary": "围绕澄清理由和判断核心展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "clarification",
                "expected_main_seriousness": "high",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "direct_user_solution",
                "expected_main_opening_hint_contains": "问题核心",
                "expected_supplement_task_type": "clarification",
                "expected_supplement_seriousness": "high",
                "expected_supplement_address_style": "support_user_after_main",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt06",
            title="open_discussion：开放讨论",
            target_behavior="命中 open_discussion，严肃度为 low；主答正常面向用户，补充者接主答继续说。",
            query="你觉得短视频和 AI 工具这件事，到底该怎么看，有什么看法？",
            context_query="当前是开放讨论，不是单点求助，允许平衡观点和不同角度。",
            topic_state={
                "main_topic": "如何看待短视频与 AI 工具",
                "active_topic": ["怎么看", "看法", "技术影响", "生活方式"],
                "topic_summary": "围绕开放性看法展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["leonard", "sheldon"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "min_speakers_gte": 2,
                "max_speakers_lte": 3,
                "expected_main_task_type": "open_discussion",
                "expected_main_seriousness": "low",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "default",
                "expected_main_opening_hint_contains": "正面回应用户当前的问题",
                "expected_supplement_task_type": "open_discussion",
                "expected_supplement_seriousness": "low",
                "expected_supplement_address_style": "continue_from_main",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    cases.append(
        _make_case(
            case_id="tt07",
            title="light_trouble：轻麻烦",
            target_behavior="命中 light_trouble，严肃度为 low；Penny / Leonard 更适合主答，主答开头应轻接当前小麻烦。",
            query="我今天出门忘记带工卡了，现在卡在楼下进不去，真的有点麻烦。",
            context_query="这是一个轻麻烦场景，先快速接住，再给现实的小处理办法。",
            topic_state={
                "main_topic": "忘带工卡的应对",
                "active_topic": ["忘记带工卡", "卡在楼下", "小麻烦", "应对办法"],
                "topic_summary": "围绕一个生活里的小麻烦展开。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            expected={
                "summary_trigger": False,
                "check_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "must_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT],
                "exact_speaker_count": 2,
                "expected_main_task_type": "light_trouble",
                "expected_main_seriousness": "low",
                "expected_main_address_to_all": ["user"],
                "expected_main_address_style": "default",
                "expected_main_opening_hint_contains": "小麻烦",
                "expected_supplement_task_type": "light_trouble",
                "expected_supplement_seriousness": "low",
                "expected_supplement_address_style": "continue_from_main",
            },
            agent_list=agents,
            group="task_type",
        )
    )

    return cases


def get_path_cases() -> List[CaseDict]:
    """
    这组不是替换公共 case，而是额外补的“路径测试 case”。
    重点压 check / pre-summary / redundancy summary / return / end。
    """
    agents = get_functional_agents()
    cases: List[CaseDict] = []

    cases.append(
        _make_case(
            case_id="tc01",
            title="朋友矛盾中的分歧核实",
            target_behavior="上一轮出现不同路线时，应先触发 relation_verify，并显式选出 check。",
            query="你们两个观点好像不一样，先别急着继续说，先核实一下到底是在反对还是只是角度不同。",
            context_query="上一轮 Sheldon 说不要立刻联系，Penny 说先去联系。当前先核实两种说法的关系。",
            topic_state={
                "main_topic": "是否先联系朋友",
                "active_topic": ["先联系朋友", "先梳理情绪", "关系核实"],
                "topic_summary": "上一轮似乎出现不同路线，需要 check 层先核实。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            history_summary={"done_points": ["朋友关系有点僵"], "resolved": False},
            last_round_outputs=[
                {
                    "agent_id": "sheldon",
                    "text": "我不建议现在立刻联系，先把问题拆开看清楚再说。",
                    "key_points": ["不建议立刻联系", "先拆开问题看清楚"],
                },
                {
                    "agent_id": "penny",
                    "text": "我倒觉得别想太多了，如果你在乎这段关系就先发一句消息吧。",
                    "key_points": ["别想太多", "先发一句消息"],
                },
            ],
            expected={
                "check_trigger": True,
                "check_reason": CHECK_REASON_RELATION,
                "summary_trigger": False,
                "preferred_main_any": ["penny", "leonard"],
                "must_have_roles": [ROLE_MAIN, ROLE_CHECK],
                "must_not_have_roles": [ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="path",
        )
    )

    cases.append(
        _make_case(
            case_id="tc02",
            title="努力 vs 天赋：进入总结前的检查",
            target_behavior="覆盖点接近收齐但尚未 resolved 时，应触发 pre_summary_verify，并显式出现 check。",
            query="现在关于努力和天赋已经聊出几条主线了，这一轮先看看是不是可以进入阶段总结。",
            context_query="我们已经聊到起点差异、长期训练和机会结构，这一轮先检查覆盖度，而不是直接继续展开。",
            topic_state={
                "main_topic": "努力 vs 天赋",
                "active_topic": ["起点差异", "长期训练", "机会结构", "阶段总结"],
                "topic_summary": "观点已经接近收齐，先检查是否可总结。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
                "allow_pre_summary": True,
            },
            history_summary={"done_points": ["天赋会影响起点", "努力决定能走多远"], "resolved": False},
            last_round_outputs=[
                {
                    "agent_id": "sheldon",
                    "text": "如果定义得更严格一点，天赋更像初始条件，而努力更像长期变量。",
                    "key_points": ["天赋像初始条件", "努力像长期变量"],
                }
            ],
            expected={
                "check_trigger": True,
                "check_reason": CHECK_REASON_PRE_SUMMARY,
                "summary_trigger": False,
                "must_have_roles": [ROLE_MAIN, ROLE_CHECK],
                "must_not_have_roles": [ROLE_SUMMARY],
                "exact_speaker_count": 2,
            },
            agent_list=agents,
            group="path",
        )
    )

    cases.append(
        _make_case(
            case_id="tc03",
            title="拖延建议已经高度重复，直接收束",
            target_behavior="内容高度重复时，应触发 high_redundancy summary，并显式出现 summary。",
            query="你们已经来回说得差不多了，这一轮就别再重复了，直接收一下现在最有用的结论。",
            context_query="前面已经反复提到先降低启动成本、把手机放远、先做五分钟。这一轮判断是否直接总结。",
            topic_state={
                "main_topic": "拖延刷手机的调整办法",
                "active_topic": ["降低启动成本", "手机放远", "先做五分钟", "阶段收束"],
                "topic_summary": "内容已经比较重复，适合收束。",
                "topic_action": "maintain",
                "scene_mode": "general_chat",
            },
            history_summary={
                "done_points": ["先降低启动成本", "把手机放远一点", "先做五分钟再说", "不要一开始就想做完整件事"],
                "resolved": False,
            },
            last_round_outputs=[
                {
                    "agent_id": "leonard",
                    "text": "我还是觉得先降低启动成本比较重要，先做五分钟就好。",
                    "key_points": ["先降低启动成本", "先做五分钟"],
                },
                {
                    "agent_id": "penny",
                    "text": "对啊，而且手机先放远一点，不然你手一伸又会点开。",
                    "key_points": ["手机放远一点", "又会点开"],
                },
            ],
            expected={
                "check_trigger": False,
                "summary_trigger": True,
                "summary_reason": SUMMARY_REASON_REDUNDANCY,
                "must_have_roles": [ROLE_MAIN, ROLE_SUMMARY],
                "must_not_have_roles": [ROLE_CHECK],
                "exact_speaker_count": 2,
                "expected_redundancy_gte": 0.5,
            },
            agent_list=agents,
            group="path",
        )
    )

    cases.append(
        _make_case(
            case_id="tc04",
            title="从公司八卦拉回 AI 工作替代主线",
            target_behavior="topic_action=return 时，应直接进入 topic_return summary，并切回 previous_main_topic。",
            query="先别聊八卦了，拉回到 AI 会不会影响工作 这个问题本身。",
            context_query="刚才已经从 AI 会不会影响工作 漂到了某家公司裁员八卦，这一轮先把大家拉回原题。",
            topic_state={
                "main_topic": "公司八卦",
                "active_topic": ["裁员八卦", "公司新闻"],
                "topic_summary": "已经偏到具体公司八卦。",
                "topic_action": "return",
                "previous_main_topic": "AI 会不会影响工作",
                "previous_active_topic": ["AI", "工作替代", "职业变化"],
                "previous_topic_summary": "上一稳定话题是 AI 对工作的影响。",
                "scene_mode": "general_chat",
            },
            history_summary={"done_points": ["AI 会改变一部分工作结构"], "resolved": False},
            last_round_outputs=[
                {
                    "agent_id": "penny",
                    "text": "我倒觉得大家最近都在聊那家公司更吓人一点。",
                    "key_points": ["那家公司更吓人"],
                }
            ],
            expected={
                "summary_trigger": True,
                "summary_reason": SUMMARY_REASON_TOPIC_RETURN,
                "must_have_roles": [ROLE_SUMMARY],
                "must_not_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT, ROLE_CHECK],
                "exact_speaker_count": 1,
                "expected_next_main_topic": "AI 会不会影响工作",
                "expected_next_topic_action": "maintain",
                "expected_forced_summary": True,
            },
            agent_list=agents,
            group="path",
        )
    )

    cases.append(
        _make_case(
            case_id="tc05",
            title="用户主动结束对话",
            target_behavior="topic_action=end 时，不再正常调度，直接由总结者礼貌收尾。",
            query="好啦那就先聊到这里，晚安。",
            context_query="用户明确想结束这轮对话。",
            topic_state={
                "main_topic": "结束对话",
                "active_topic": ["结束会话"],
                "topic_summary": "用户希望现在结束。",
                "topic_action": "end",
                "scene_mode": "general_chat",
            },
            history_summary={"done_points": ["已经给过建议"], "resolved": True},
            expected={
                "summary_trigger": True,
                "summary_reason": SUMMARY_REASON_TOPIC_END,
                "must_have_roles": [ROLE_SUMMARY],
                "must_not_have_roles": [ROLE_MAIN, ROLE_SUPPLEMENT, ROLE_CHECK],
                "exact_speaker_count": 1,
                "expected_next_topic_action": "end",
                "expected_forced_summary": True,
            },
            agent_list=agents,
            group="path",
        )
    )

    return cases


def get_cases(agent_list: Optional[List[AgentDict]] = None) -> List[CaseDict]:
    """
    最终测试集 = 原公共 6 case（不改） + 变体 case + 路径测试 case。
    """
    public_cases = get_public_cases(agent_list=agent_list)
    variant_cases = get_variant_cases(agent_list=agent_list)
    task_type_cases = get_task_type_cases(agent_list=agent_list)
    path_cases = get_path_cases()
    return public_cases + variant_cases + task_type_cases + path_cases
