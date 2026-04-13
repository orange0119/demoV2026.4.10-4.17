# Optimized unified orchestrator generated from query_management + main_tiebreak + supplement_diversity

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z0-9_]+")
CONTRAST_CUES = ["但", "不过", "然而", "另一方面", "相比", "反而", "而是", "but", "however", "instead"]

ROLE_MAIN = "main"
ROLE_SUPPLEMENT = "supplement"
ROLE_CHECK = "check"
ROLE_SUMMARY = "summary"

ACTION_SKIP = "skip"
ACTION_MUTE = "mute"

SUMMARY_REASON_ENOUGH = "enough_points_collected"
SUMMARY_REASON_REDUNDANCY = "high_redundancy"
SUMMARY_REASON_RESOLVED = "user_query_resolved"
SUMMARY_REASON_TOPIC_RETURN = "topic_return"
SUMMARY_REASON_TOPIC_END = "end_conversation"

CHECK_REASON_RELATION_VERIFY = "relation_verify"
CHECK_REASON_REDUNDANCY = "redundancy_verify"
CHECK_REASON_PRE_SUMMARY = "pre_summary_verify"

SKIP_REASON_LOW_RELEVANCE = "topic_low_relevance"
SKIP_REASON_REDUNDANCY = "high_redundancy"
SKIP_REASON_QUOTA_LIMIT = "quota_limit"

NEGATION_CUES = [
    "不建议", "不适合", "不是", "不能", "不会", "没", "没有", "无", "别", "避免", "反对",
    "no", "not", "never", "cannot", "can't", "won't", "avoid", "against",
]

CHECK_NAME_HINTS = ("主持", "检查", "审核", "裁判", "moderator", "checker", "review")
SUMMARY_NAME_HINTS = ("主持", "总结", "收束", "summar", "moderator")

TASK_TYPE_RULES = {
    "practical_help": [r"怎么办", r"怎么做", r"如何", r"帮我", r"解决", r"该咋办", r"该怎么办"],
    "relationship_reading": [r"是不是在针对我", r"是不是讨厌我", r"他什么意思", r"她什么意思", r"是不是故意", r"阴阳怪气", r"冷淡"],
    "decision_support": [r"该不该", r"要不要", r"是否要", r"要不要换", r"选哪个", r"该选"],
    "emotional_support": [r"我好", r"我很", r"难受", r"委屈", r"烦", r"怕", r"累", r"崩溃", r"焦虑"],
    "clarification": [r"为什么", r"凭什么", r"怎么会", r"不是.+吗", r"你是说"],
    "open_discussion": [r"怎么看", r"你觉得", r"看法", r"评价", r"有什么看法"],
    "light_trouble": [r"忘记", r"没带", r"找不到", r"迟到", r"卡住", r"坏了", r"麻烦", r"丢了", r"来不及", r"没电", r"断网", r"出问题", r"出故障", r"不见了", r"搞丢"],
}


def tokenize_mixed_text(text: str) -> List[str]:
    if not text:
        return []
    text = text.strip()
    lowered = text.lower()
    tokens: List[str] = []
    tokens.extend(LATIN_RE.findall(lowered))
    cjk_chars = [ch for ch in text if CJK_RE.match(ch)]
    tokens.extend(cjk_chars)
    tokens.extend(["".join(cjk_chars[i:i + 2]) for i in range(len(cjk_chars) - 1)])
    tokens.extend(["".join(cjk_chars[i:i + 3]) for i in range(len(cjk_chars) - 2)])
    return [tok for tok in tokens if not re.fullmatch(r"[a-z]", tok)]


class LightweightHashEmbedder:
    """零外部依赖的最小 embedding。"""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _stable_index(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def _char_ngrams(self, text: str) -> List[str]:
        compact = re.sub(r"\s+", "", text.lower())
        grams: List[str] = []
        for n in (2, 3):
            for i in range(max(len(compact) - n + 1, 0)):
                grams.append(compact[i:i + n])
        return grams

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        features = tokenize_mixed_text(text) + self._char_ngrams(text)
        if not features:
            return vec
        for feat in features:
            vec[self._stable_index(feat)] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


@dataclass
class Agent:
    agent_id: str
    name: str
    status: str = "online"
    keywords: Optional[List[str]] = None
    description: str = ""
    persona: str = ""
    style: str = ""
    can_summarize: bool = False
    can_check: bool = False


@dataclass
class ScoreDetail:
    agent_id: str
    name: str
    relevance: float
    repeat_penalty: float
    final_score: float
    query_focus: float = 0.0
    topic_focus: float = 0.0


@dataclass
class TopicBundle:
    user_query: str
    context_query: str
    effective_query: str
    query_source: str
    main_topic: str
    active_topics: List[str]
    topic_summary: str
    topic_action: str
    topic_transition: str
    allow_pre_summary: bool
    scene_mode: str
    done_points: List[str]
    resolved: bool
    previous_main_topic: str
    previous_active_topics: List[str]
    previous_topic_summary: str


class MinimalOrchestrator:
    """
    最小 MVP 版：
    1) relevance 只看当前轮上下文与主线
    2) 删除 novelty
    3) history_summary 只保留 done_points / resolved
    4) 冲突检测集中放到 check 层，借鉴 DDPE 的 relation 判断 + WHoW 的 moderator act
    5) topic_action 支持 maintain | return | end；return 时主持人先收束再切回上一个话题，end 时直接向用户告别
    """

    def _scene_mode(self, bundle: TopicBundle, payload: Dict[str, Any]) -> str:
        topic_state = payload.get("topic_state", {}) or {}
        return (topic_state.get("scene_mode", "") or "general_chat").strip()

    def _scene_bonus(self, agent: Agent, scene_mode: str) -> float:
        def _agent_key(self, agent: Agent) -> str:
            return (agent.agent_id or agent.name or "").strip().lower()

        def _contains_any(self, text: str, keywords: Sequence[str]) -> bool:
            lowered = (text or "").lower()
            return any(keyword.lower() in lowered for keyword in keywords)

        def _character_role_bonus(self, agent: Agent, bundle: "TopicBundle") -> float:
            key = self._agent_key(agent)
            text = " ".join([
                bundle.user_query or "",
                bundle.context_query or "",
                bundle.main_topic or "",
                " ".join(bundle.active_topics or []),
                bundle.topic_summary or "",
                bundle.scene_mode or "",
            ]).lower()

            emotion_like = self._contains_any(text, ["累", "疲惫", "不想做", "闹矛盾", "朋友", "联系", "关系", "难过",
                                                     "犹豫"])
            habit_like = self._contains_any(text, ["拖延", "刷手机", "注意力", "行动启动"])
            opinion_like = self._contains_any(text, ["努力", "天赋", "哪个更重要"])
            ai_work_like = self._contains_any(text, ["ai", "取代", "工作", "职业变化"])
            focus_like = self._contains_any(text, ["短视频", "专注", "静下心", "注意力切换"])

            bonus = 0.0
            if key == "sheldon":
                if emotion_like or bundle.scene_mode == "emotion_support":
                    bonus -= 0.065
                if habit_like:
                    bonus += 0.085
                if opinion_like:
                    bonus += 0.06
                if ai_work_like:
                    bonus += 0.11
                if focus_like:
                    bonus += 0.05
            elif key == "leonard":
                if emotion_like or bundle.scene_mode == "emotion_support":
                    bonus += 0.05
                if habit_like:
                    bonus += 0.055
                if opinion_like:
                    bonus += 0.04
                if ai_work_like:
                    bonus += 0.0
                if focus_like:
                    bonus += 0.06
            elif key == "penny":
                if emotion_like or bundle.scene_mode == "emotion_support":
                    bonus += 0.075
                if habit_like:
                    bonus -= 0.04
                if opinion_like:
                    bonus -= 0.01
                if ai_work_like:
                    bonus -= 0.03
                if focus_like:
                    bonus += 0.015

            return round(bonus, 4)
        text = " ".join([
            agent.name,
            agent.description,
            agent.persona,
            agent.style,
            " ".join(agent.keywords or []),
        ]).lower()

        if scene_mode == "emotion_support":
            if any(k in text for k in ["情绪", "共鸣", "真诚", "成熟", "总结", "控场", "稳重"]):
                return 0.045
            if any(k in text for k in ["舞台", "炸场", "灯光", "综艺效果"]):
                return -0.01

        if scene_mode == "celeb_work_topic":
            if any(k in text for k in ["舞台", "节奏", "作品", "开场", "编排", "审美"]):
                return 0.04

        if scene_mode == "promo_discussion":
            if any(k in text for k in ["采访", "互动", "控场", "综艺", "表达"]):
                return 0.04

        if scene_mode == "topic_return":
            if agent.can_summarize or agent.can_check:
                return 0.05

        return 0.0

    def _agent_key(self, agent: Agent) -> str:
        return (agent.agent_id or agent.name or "").strip().lower()

    def _contains_any(self, text: str, keywords: Sequence[str]) -> bool:
        lowered = (text or "").lower()
        return any(keyword.lower() in lowered for keyword in keywords)

    def _character_role_bonus(self, agent: Agent, bundle: "TopicBundle") -> float:
        key = self._agent_key(agent)
        text = " ".join([
            bundle.user_query or "",
            bundle.context_query or "",
            bundle.main_topic or "",
            " ".join(bundle.active_topics or []),
            bundle.topic_summary or "",
            bundle.scene_mode or "",
        ]).lower()

        emotion_like = self._contains_any(text,
                                          ["累", "疲惫", "不想做", "闹矛盾", "朋友", "联系", "关系", "难过", "犹豫"])
        habit_like = self._contains_any(text, ["拖延", "刷手机", "注意力", "行动启动"])
        opinion_like = self._contains_any(text, ["努力", "天赋", "哪个更重要"])
        ai_work_like = self._contains_any(text, ["ai", "取代", "工作", "职业变化"])
        focus_like = self._contains_any(text, ["短视频", "专注", "静下心", "注意力切换"])

        bonus = 0.0
        if key == "sheldon":
            if emotion_like or bundle.scene_mode == "emotion_support":
                bonus -= 0.065
            if habit_like:
                bonus += 0.085
            if opinion_like:
                bonus += 0.06
            if ai_work_like:
                bonus += 0.11
            if focus_like:
                bonus += 0.05
        elif key == "leonard":
            if emotion_like or bundle.scene_mode == "emotion_support":
                bonus += 0.05
            if habit_like:
                bonus += 0.055
            if opinion_like:
                bonus += 0.04
            if ai_work_like:
                bonus += 0.0
            if focus_like:
                bonus += 0.06
        elif key == "penny":
            if emotion_like or bundle.scene_mode == "emotion_support":
                bonus += 0.075
            if habit_like:
                bonus -= 0.04
            if opinion_like:
                bonus -= 0.01
            if ai_work_like:
                bonus -= 0.03
            if focus_like:
                bonus += 0.015

        return round(bonus, 4)

    def _main_scene_tiebreak_prior(self, agent: Agent, bundle: "TopicBundle") -> float:
        key = self._agent_key(agent)
        text = " ".join([
            bundle.effective_query or bundle.user_query or "",
            bundle.context_query or "",
            bundle.main_topic or "",
            " ".join(bundle.active_topics or []),
            bundle.topic_summary or "",
            bundle.scene_mode or "",
        ]).lower()

        emotion_like = self._contains_any(text, ["累", "疲惫", "不想做", "闹矛盾", "朋友", "联系", "关系", "难过", "犹豫"])
        habit_like = self._contains_any(text, ["拖延", "刷手机", "短视频停不下来", "行动启动", "注意力管理"])
        opinion_like = self._contains_any(text, ["努力", "天赋", "哪个更重要", "意义"])
        ai_like = self._contains_any(text, ["ai", "取代", "工作", "职业变化", "岗位"])
        focus_like = self._contains_any(text, ["短视频", "专注", "静下心", "注意力切换", "ai 工具"])

        if emotion_like or bundle.scene_mode == "emotion_support":
            priors = {"penny": 1.0, "leonard": 0.9, "sheldon": 0.25}
        elif habit_like:
            priors = {"sheldon": 1.0, "leonard": 0.9, "penny": 0.35}
        elif opinion_like:
            priors = {"sheldon": 1.0, "leonard": 0.95, "penny": 0.45}
        elif ai_like or focus_like:
            priors = {"sheldon": 1.0, "leonard": 0.85, "penny": 0.4}
        else:
            priors = {"leonard": 0.9, "sheldon": 0.85, "penny": 0.6}
        return float(priors.get(key, 0.5))

    def _supplement_functional_fit(self, candidate: Agent, main_agent: Optional[Agent], bundle: "TopicBundle") -> float:
        if main_agent is None:
            return 0.0

        main_key = self._agent_key(main_agent)
        cand_key = self._agent_key(candidate)
        text = " ".join([
            bundle.effective_query or bundle.user_query or "",
            bundle.context_query or "",
            bundle.main_topic or "",
            " ".join(bundle.active_topics or []),
            bundle.topic_summary or "",
            bundle.scene_mode or "",
        ]).lower()

        emotion_like = self._contains_any(text, ["累", "疲惫", "闹矛盾", "朋友", "联系", "关系", "难过", "犹豫"])
        habit_like = self._contains_any(text, ["拖延", "刷手机", "行动启动", "注意力管理"])
        opinion_like = self._contains_any(text, ["努力", "天赋", "意义"])
        ai_like = self._contains_any(text, ["ai", "取代", "工作", "职业变化"])
        focus_like = self._contains_any(text, ["短视频", "专注", "ai 工具"])

        fit = 0.0
        if main_key == "sheldon":
            if emotion_like:
                fit = {"leonard": 0.75, "penny": 1.0}.get(cand_key, 0.0)
            elif habit_like or ai_like or focus_like:
                fit = {"leonard": 1.0, "penny": 0.45}.get(cand_key, 0.0)
            elif opinion_like:
                fit = {"leonard": 0.85, "penny": 0.8}.get(cand_key, 0.0)
            else:
                fit = {"leonard": 0.9, "penny": 0.7}.get(cand_key, 0.0)
        elif main_key == "leonard":
            if emotion_like:
                fit = {"penny": 1.0, "sheldon": 0.45}.get(cand_key, 0.0)
            elif habit_like or ai_like or opinion_like or focus_like:
                fit = {"sheldon": 1.0, "penny": 0.7}.get(cand_key, 0.0)
            else:
                fit = {"penny": 0.85, "sheldon": 0.75}.get(cand_key, 0.0)
        elif main_key == "penny":
            if emotion_like:
                fit = {"leonard": 1.0, "sheldon": 0.35}.get(cand_key, 0.0)
            elif habit_like or ai_like or opinion_like or focus_like:
                fit = {"leonard": 0.8, "sheldon": 1.0}.get(cand_key, 0.0)
            else:
                fit = {"leonard": 1.0, "sheldon": 0.65}.get(cand_key, 0.0)
        return round(fit, 4)

    def __init__(self, embedder: Optional[LightweightHashEmbedder] = None) -> None:
        self.embedder = embedder or LightweightHashEmbedder(dim=384)

        self.min_main_relevance = 0.18
        self.min_main_score = 0.18
        self.min_supplement_relevance = 0.14
        self.min_supplement_score = 0.14

        self.medium_redundancy_threshold = 0.45
        self.high_redundancy_threshold = 0.52
        self.conflict_similarity_threshold = 0.22
        self.pre_summary_points_threshold = 3
        self.summary_points_threshold = 4

        self.repeat_penalty_value = 0.08
        self.tie_margin_main = 0.02
        self.tie_margin_supplement = 0.02

        # 近似同分冲突裁决阈值
        self.close_score_delta_main = 0.02
        self.close_score_delta_supplement = 0.02
        self.close_query_focus_delta = 0.015
        self.close_topic_focus_delta = 0.015
        self.close_diversity_delta = 0.05

        # 功能型 agent（check / summarize）默认不抢 main，
        # 只有普通内容型候选明显过弱，才允许它们回到 main 兜底。
        self.functional_main_fallback_floor = 0.14
        self.functional_main_fallback_delta = 0.05

    # =========================
    # 输入解析
    # =========================

    def _build_context_query(self, payload: Dict[str, Any]) -> str:
        explicit_context = (payload.get("context_query", "") or "").strip()
        if explicit_context:
            return explicit_context

        last_round_outputs = payload.get("last_round_outputs", []) or []
        context_lines: List[str] = []
        for item in last_round_outputs:
            speaker = item.get("agent_name") or item.get("name") or item.get("agent_id") or "某人"
            role = item.get("role", "") or ""
            text = (item.get("text", "") or "").strip()
            if not text:
                continue
            role_suffix = f"（{role}）" if role else ""
            context_lines.append(f"{speaker}{role_suffix}：{text}")
        if context_lines:
            return "\n".join(context_lines)

        return (payload.get("query", "") or "").strip()

    def _is_current_user_turn(self, payload: Dict[str, Any]) -> bool:
        explicit_flag = payload.get("user_current_turn")
        if isinstance(explicit_flag, bool):
            return explicit_flag

        explicit_flag = payload.get("has_user_query")
        if isinstance(explicit_flag, bool):
            return explicit_flag

        topic_state = payload.get("topic_state", {}) or {}
        if bool(topic_state.get("user_silent", False)) or bool(topic_state.get("inherit_context_query", False)):
            return False

        user_text = (payload.get("user_text", "") or "").strip()
        query = (payload.get("query", "") or "").strip()
        if user_text and query and user_text == query:
            return True

        if not query:
            return False

        continuation_cues = ["继续", "推进", "展开", "这一轮", "再说", "补充", "继续推进", "接着", "延续"]
        if len(query) <= 10 and any(cue in query for cue in continuation_cues):
            return False
        return True

    def _resolve_effective_query(self, payload: Dict[str, Any], context_query: str) -> Tuple[str, str]:
        user_query = (payload.get("query", "") or "").strip()
        if self._is_current_user_turn(payload) and user_query:
            return user_query, "user_query"

        inherited_context = (context_query or "").strip()
        if inherited_context:
            return inherited_context, "inherited_context_query"

        return user_query, "fallback_query"

    def _normalize_topic_action(self, action: str) -> str:
        action = (action or "").strip().lower()
        if action in {"return", "maintain", "end"}:
            return action
        if action in {"stay", "keep", ""}:
            return "maintain"
        if action in {"stop", "finish", "close", "bye"}:
            return "end"
        return "maintain"

    def _normalize_topic_transition(self, transition: str) -> str:
        transition = (transition or "").strip().lower()
        if transition in {"maintain", "shift", "return", "end"}:
            return transition
        if transition in {"new_topic", "switch", "topic_shift"}:
            return "shift"
        return "maintain"

    def _sanitize_history_summary(self, history_summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(history_summary, dict):
            return {"done_points": [], "resolved": False}
        return {
            "done_points": list(history_summary.get("done_points", []) or []),
            "resolved": bool(history_summary.get("resolved", False)),
        }

    def _parse_topic_bundle(self, payload: Dict[str, Any]) -> TopicBundle:
        topic_state = payload.get("topic_state", {}) or {}
        history_summary = self._sanitize_history_summary(payload.get("history_summary", {}) or {})

        active_topics = topic_state.get("active_topic", []) or []
        if isinstance(active_topics, str):
            active_topics = [active_topics] if active_topics.strip() else []

        previous_active = topic_state.get("previous_active_topic", []) or []
        if isinstance(previous_active, str):
            previous_active = [previous_active] if previous_active.strip() else []

        user_query = (payload.get("query", "") or "").strip()
        context_query = self._build_context_query(payload)
        effective_query, query_source = self._resolve_effective_query(payload, context_query)

        return TopicBundle(
            user_query=user_query,
            context_query=context_query,
            effective_query=effective_query,
            query_source=query_source,
            main_topic=topic_state.get("main_topic", "") or "",
            active_topics=active_topics,
            topic_summary=topic_state.get("topic_summary", "") or "",
            topic_action=self._normalize_topic_action(topic_state.get("topic_action", "")),
            topic_transition=self._normalize_topic_transition(topic_state.get("topic_transition", "")),
            allow_pre_summary=bool(topic_state.get("allow_pre_summary", False)),
            scene_mode=(topic_state.get("scene_mode", "") or "general_chat").strip(),
            done_points=history_summary.get("done_points", []) or [],
            resolved=bool(topic_state.get("resolved", False) or history_summary.get("resolved", False)),
            previous_main_topic=(
                topic_state.get("previous_main_topic")
                or topic_state.get("return_to_topic")
                or topic_state.get("last_stable_topic")
                or topic_state.get("main_topic", "")
                or ""
            ),
            previous_active_topics=previous_active,
            previous_topic_summary=topic_state.get("previous_topic_summary", "") or "",
        )

    def _topic_text(self, bundle: TopicBundle) -> str:
        parts = [
            bundle.user_query,
            bundle.effective_query,
            bundle.context_query,
            bundle.main_topic,
            " ".join(bundle.active_topics),
            bundle.topic_summary,
            bundle.topic_action,
            " ".join(bundle.done_points),
        ]
        return "\n".join(p for p in parts if p)

    def _parse_agents(self, agent_list: List[Dict[str, Any]]) -> List[Agent]:
        agents: List[Agent] = []
        for raw in agent_list:
            name = raw.get("name", "")
            agents.append(
                Agent(
                    agent_id=raw.get("agent_id", name),
                    name=name,
                    status=raw.get("status", "online"),
                    keywords=raw.get("keywords", []) or [],
                    description=raw.get("description", "") or "",
                    persona=raw.get("persona", "") or "",
                    style=raw.get("style", "") or "",
                    can_summarize=raw.get("can_summarize", False),
                    can_check=raw.get("can_check", False),
                )
            )
        return agents

    def _get_recent_speakers(self, last_round_outputs: List[Dict[str, Any]]) -> set[str]:
        return {item.get("agent_id") for item in last_round_outputs if item.get("agent_id")}

    def _collect_last_round_key_points(self, last_round_outputs: List[Dict[str, Any]]) -> List[str]:
        points: List[str] = []
        for item in last_round_outputs:
            for p in item.get("key_points", []) or []:
                if p:
                    points.append(p)
            if item.get("text") and not item.get("key_points"):
                points.append(item["text"])
        return points

    # =========================
    # 相似度工具
    # =========================

    def _encode(self, text: str) -> np.ndarray:
        return self.embedder.encode(text or "")

    def _sim(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        return round(cosine_similarity(self._encode(text_a), self._encode(text_b)), 4)

    def _avg_topk_sim(self, text: str, candidates: Sequence[str], k: int = 2) -> float:
        vals = [self._sim(text, c) for c in candidates if c]
        if not vals:
            return 0.0
        vals.sort(reverse=True)
        top = vals[: max(k, 1)]
        return round(sum(top) / len(top), 4)

    def _max_sim_to_list(self, text: str, candidates: Sequence[str]) -> float:
        vals = [self._sim(text, c) for c in candidates if c]
        return max(vals) if vals else 0.0

    def _topic_anchor(self, bundle: TopicBundle) -> str:
        return "\n".join([
            bundle.user_query,
            bundle.main_topic,
            " ".join(bundle.active_topics),
            bundle.topic_summary,
        ])

    def _topic_reference_text(self, bundle: TopicBundle) -> str:
        return "\n".join([
            bundle.previous_main_topic,
            " ".join(bundle.previous_active_topics),
            bundle.previous_topic_summary,
        ])

    def _lexical_overlap_ratio(self, text: str, reference: str) -> float:
        text_tokens = set(tokenize_mixed_text(text or ""))
        ref_tokens = set(tokenize_mixed_text(reference or ""))
        if not text_tokens or not ref_tokens:
            return 0.0
        return round(len(text_tokens & ref_tokens) / max(len(text_tokens), 1), 4)

    def _point_topic_relevance(self, point: str, bundle: TopicBundle) -> float:
        current_anchor = self._topic_anchor(bundle)
        current_sim = self._sim(point, current_anchor)
        current_lex = self._lexical_overlap_ratio(point, current_anchor)
        current_rel = max(current_sim, current_lex)

        if bundle.topic_transition != "shift":
            return round(current_rel, 4)

        previous_anchor = self._topic_reference_text(bundle)
        previous_sim = self._sim(point, previous_anchor)
        previous_lex = self._lexical_overlap_ratio(point, previous_anchor)
        previous_rel = max(previous_sim, previous_lex)

        if previous_rel > current_rel + 0.03:
            return 0.0
        return round(current_rel, 4)

    def _filter_points_for_current_topic(
        self,
        points: Sequence[str],
        bundle: TopicBundle,
        min_relevance: float = 0.18,
    ) -> List[str]:
        filtered: List[str] = []
        for point in points:
            point = (point or "").strip()
            if not point:
                continue
            if self._point_topic_relevance(point, bundle) >= min_relevance:
                filtered.append(point)
        return filtered

    # =========================
    # 相关度与得分
    # =========================

    def _agent_profile_text(self, agent: Agent) -> str:
        return " ".join(
            p for p in [
                agent.name,
                agent.description,
                agent.persona,
                agent.style,
                " ".join(agent.keywords or []),
            ] if p
        ).strip()

    def _agent_relevance(self, agent: Agent, bundle: TopicBundle) -> float:
        profile_text = self._agent_profile_text(agent)
        if not profile_text:
            return 0.0

        weak_query_cues = ["继续", "推进", "展开", "这一轮", "再说", "补充", "继续推进"]
        weak_user_query = bundle.query_source == "user_query" and bool(bundle.user_query) and (
            len(bundle.user_query.strip()) <= 10
            or any(cue in bundle.user_query for cue in weak_query_cues)
        )

        if bundle.query_source == "user_query" and not weak_user_query:
            weighted_parts: List[Tuple[str, float]] = [
                (bundle.effective_query, 0.60),
                (bundle.main_topic, 0.20),
                (" ".join(bundle.active_topics), 0.10),
                (bundle.topic_summary, 0.05),
                (bundle.context_query, 0.05),
            ]
        elif bundle.query_source == "inherited_context_query":
            weighted_parts = [
                (bundle.effective_query, 0.60),
                (bundle.main_topic, 0.20),
                (" ".join(bundle.active_topics), 0.10),
                (bundle.topic_summary, 0.05),
                (bundle.user_query, 0.05),
            ]
        else:
            weighted_parts = [
                (bundle.effective_query, 0.25),
                (bundle.main_topic, 0.25),
                (" ".join(bundle.active_topics), 0.15),
                (bundle.topic_summary, 0.15),
                (bundle.context_query, 0.20),
            ]

        profile_score = 0.0
        for text_part, weight in weighted_parts:
            if text_part:
                profile_score += weight * self._sim(profile_text, text_part)

        keyword_score = 0.0
        keywords = agent.keywords or []
        if keywords:
            keyword_text = " ".join(keywords)
            keyword_targets = [
                bundle.effective_query,
                bundle.main_topic,
                " ".join(bundle.active_topics),
                bundle.topic_summary,
            ]
            keyword_score = self._avg_topk_sim(keyword_text, keyword_targets, k=2)

        relevance = 0.80 * profile_score + 0.20 * keyword_score
        return round(min(max(relevance, 0.0), 1.0), 4)

    def _agent_query_focus(self, agent: Agent, bundle: TopicBundle) -> float:
        profile_text = self._agent_profile_text(agent)
        target = bundle.effective_query or bundle.user_query or bundle.main_topic or bundle.topic_summary
        if not profile_text or not target:
            return 0.0
        return round(self._sim(profile_text, target), 4)

    def _agent_topic_focus(self, agent: Agent, bundle: TopicBundle) -> float:
        profile_text = self._agent_profile_text(agent)
        targets = [bundle.main_topic, " ".join(bundle.active_topics), bundle.topic_summary]
        if not profile_text:
            return 0.0
        return round(self._avg_topk_sim(profile_text, [t for t in targets if t], k=2), 4)

    def _tasktype_text(self, bundle: TopicBundle) -> str:
        return " ".join([
            bundle.user_query or "",
            bundle.effective_query or "",
            bundle.context_query or "",
            bundle.main_topic or "",
            " ".join(bundle.active_topics or []),
            bundle.topic_summary or "",
            bundle.scene_mode or "",
        ]).strip()

    def _match_any_rule(self, text: str, patterns: Sequence[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _classify_task_type(self, bundle: TopicBundle) -> str:
        text = self._tasktype_text(bundle)
        if not text:
            return "open_discussion"

        relationship_patterns = TASK_TYPE_RULES["relationship_reading"]
        decision_patterns = TASK_TYPE_RULES["decision_support"]
        clarification_patterns = TASK_TYPE_RULES["clarification"]
        light_patterns = TASK_TYPE_RULES["light_trouble"]
        emotion_patterns = TASK_TYPE_RULES["emotional_support"]
        practical_patterns = TASK_TYPE_RULES["practical_help"]
        open_patterns = TASK_TYPE_RULES["open_discussion"]

        if self._match_any_rule(text, relationship_patterns):
            return "relationship_reading"
        if self._match_any_rule(text, decision_patterns):
            return "decision_support"
        if self._match_any_rule(text, clarification_patterns):
            return "clarification"

        high_emotion_markers = ["崩溃", "焦虑", "委屈", "难受", "害怕", "怕", "讨厌我", "针对我"]
        if self._match_any_rule(text, light_patterns) and not any(marker in text for marker in high_emotion_markers):
            return "light_trouble"

        if self._match_any_rule(text, emotion_patterns) or bundle.scene_mode == "emotion_support":
            return "emotional_support"

        open_hit = self._match_any_rule(text, open_patterns)
        practical_hit = self._match_any_rule(text, practical_patterns)

        if open_hit:
            return "open_discussion"
        if practical_hit:
            return "practical_help"

        return "open_discussion"

    def _infer_topic_seriousness(self, bundle: TopicBundle, task_type: str) -> str:
        if task_type in {"relationship_reading", "emotional_support", "clarification"}:
            return "high"
        if task_type in {"practical_help", "decision_support"}:
            return "medium"
        return "low"

    def _main_task_fit(self, agent: Agent, task_type: str, seriousness: str) -> float:
        key = self._agent_key(agent)
        priors = {
            "practical_help": {"sheldon": 0.95, "leonard": 1.00, "penny": 0.70},
            "relationship_reading": {"sheldon": 0.35, "leonard": 1.00, "penny": 0.95},
            "decision_support": {"sheldon": 1.05, "leonard": 1.10, "penny": 0.60},
            "emotional_support": {"sheldon": 0.25, "leonard": 0.98, "penny": 1.05},
            "clarification": {"sheldon": 1.05, "leonard": 0.90, "penny": 0.45},
            "open_discussion": {"sheldon": 0.95, "leonard": 1.00, "penny": 0.75},
            "light_trouble": {"sheldon": 0.45, "leonard": 0.95, "penny": 1.05},
        }
        base = priors.get(task_type, {}).get(key, 0.6)
        if seriousness == "high" and key == "sheldon" and task_type in {"relationship_reading", "emotional_support"}:
            base -= 0.10
        return round(max(0.0, min(base, 1.2)), 4)

    def _supplement_task_fit(self, candidate: Agent, main_agent: Optional[Agent], task_type: str, seriousness: str) -> float:
        if main_agent is None:
            return 0.0
        main_key = self._agent_key(main_agent)
        cand_key = self._agent_key(candidate)
        pair_map = {
            "emotional_support": {
                "penny": {"leonard": 1.0, "sheldon": 0.35},
                "leonard": {"penny": 1.0, "sheldon": 0.40},
                "sheldon": {"leonard": 0.85, "penny": 1.0},
            },
            "relationship_reading": {
                "leonard": {"penny": 1.0, "sheldon": 0.45},
                "penny": {"leonard": 1.0, "sheldon": 0.40},
                "sheldon": {"leonard": 0.9, "penny": 0.8},
            },
            "clarification": {
                "sheldon": {"leonard": 1.0, "penny": 0.55},
                "leonard": {"sheldon": 1.0, "penny": 0.70},
            },
            "practical_help": {
                "leonard": {"penny": 0.75, "sheldon": 1.0},
                "sheldon": {"leonard": 1.0, "penny": 0.65},
                "penny": {"leonard": 0.9, "sheldon": 0.75},
            },
            "decision_support": {
                "leonard": {"penny": 0.75, "sheldon": 1.0},
                "sheldon": {"leonard": 1.0, "penny": 0.65},
                "penny": {"leonard": 0.9, "sheldon": 0.85},
            },
            "open_discussion": {
                "leonard": {"penny": 1.0, "sheldon": 0.95},
                "sheldon": {"leonard": 1.0, "penny": 0.8},
                "penny": {"leonard": 0.95, "sheldon": 0.75},
            },
            "light_trouble": {
                "penny": {"leonard": 1.0, "sheldon": 0.55},
                "leonard": {"penny": 0.95, "sheldon": 0.75},
                "sheldon": {"leonard": 1.0, "penny": 0.70},
            },
        }
        fit = pair_map.get(task_type, {}).get(main_key, {}).get(cand_key, 0.5)
        if seriousness == "high" and cand_key == "sheldon" and task_type in {"emotional_support", "relationship_reading"}:
            fit -= 0.10
        return round(max(0.0, min(fit, 1.2)), 4)

    def _score_agents(
        self,
        agents: List[Agent],
        bundle: TopicBundle,
        last_round_outputs: List[Dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> List[ScoreDetail]:
        recent_speakers = self._get_recent_speakers(last_round_outputs)
        details: List[ScoreDetail] = []
        topic_state = (payload or {}).get("topic_state", {}) or {}
        scene_mode = (topic_state.get("scene_mode", "") or "general_chat").strip()

        for agent in agents:
            if agent.status != "online":
                details.append(
                    ScoreDetail(
                        agent_id=agent.agent_id,
                        name=agent.name,
                        relevance=0.0,
                        repeat_penalty=0.0,
                        final_score=0.0,
                        query_focus=0.0,
                        topic_focus=0.0,
                    )
                )
                continue

            relevance = self._agent_relevance(agent, bundle)
            query_focus = self._agent_query_focus(agent, bundle)
            topic_focus = self._agent_topic_focus(agent, bundle)
            repeat_penalty = self.repeat_penalty_value if agent.agent_id in recent_speakers else 0.0
            final_score = relevance + self._scene_bonus(agent, scene_mode)+ self._character_role_bonus(agent, bundle) - repeat_penalty

            details.append(
                ScoreDetail(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    relevance=round(relevance, 4),
                    repeat_penalty=round(repeat_penalty, 4),
                    final_score=round(max(0.0, min(final_score, 1.0)), 4),
                    query_focus=round(query_focus, 4),
                    topic_focus=round(topic_focus, 4),
                )
            )

        details.sort(key=lambda x: (x.final_score, x.query_focus, x.topic_focus), reverse=True)
        return details

    def _score_bucket(self, score: float, margin: float) -> int:
        if margin <= 1e-9:
            return int(score * 1000)
        return int(round(score / margin))

    def _is_functional_agent(self, agent: Agent) -> bool:
        return bool(agent.can_check or agent.can_summarize)

    def _agent_profile_similarity(self, agent_a: Agent, agent_b: Agent) -> float:
        return round(self._sim(self._agent_profile_text(agent_a), self._agent_profile_text(agent_b)), 4)

    def _main_rank_key(self, score: ScoreDetail, order_idx: int) -> Tuple[float, ...]:
        return (
            self._score_bucket(score.final_score, self.tie_margin_main),
            score.final_score,
            score.query_focus,
            score.topic_focus,
            -score.repeat_penalty,
            -order_idx,
        )

    def _supplement_rank_key(
        self,
        score: ScoreDetail,
        order_idx: int,
        main_agent: Optional[Agent],
        current_agent: Agent,
        bundle: TopicBundle,
    ) -> Tuple[float, ...]:
        diversity_from_main = 0.0
        if main_agent is not None:
            diversity_from_main = round(1.0 - self._agent_profile_similarity(current_agent, main_agent), 4)
        functional_fit = self._supplement_functional_fit(current_agent, main_agent, bundle)
        return (
            self._score_bucket(score.final_score, self.tie_margin_supplement),
            score.final_score,
            functional_fit,
            diversity_from_main,
            score.topic_focus,
            -score.repeat_penalty,
            -order_idx,
        )

    def _diversity_from_main(self, agent: Agent, main_agent: Optional[Agent]) -> float:
        if main_agent is None:
            return 0.0
        return round(1.0 - self._agent_profile_similarity(agent, main_agent), 4)

    def _is_close_score(self, score_a: float, score_b: float, margin: float) -> bool:
        return abs(score_a - score_b) < margin

    def _resolve_main_close_conflict(
        self,
        ranked_candidates: Sequence[Agent],
        score_map: Dict[str, ScoreDetail],
        bundle: TopicBundle,
    ) -> Tuple[Optional[Agent], Dict[str, Any]]:
        if not ranked_candidates:
            return None, {"decision_reason": "no_candidate", "close_score_conflict": False}

        if len(ranked_candidates) == 1:
            only_agent = ranked_candidates[0]
            only_score = score_map[only_agent.agent_id]
            return only_agent, {
                "decision_reason": "single_candidate",
                "close_score_conflict": False,
                "winner": only_agent.agent_id,
                "winner_score": only_score.final_score,
            }

        cand_a, cand_b = ranked_candidates[0], ranked_candidates[1]
        score_a = score_map[cand_a.agent_id]
        score_b = score_map[cand_b.agent_id]
        scene_prior_a = self._main_scene_tiebreak_prior(cand_a, bundle)
        scene_prior_b = self._main_scene_tiebreak_prior(cand_b, bundle)
        task_type = self._classify_task_type(bundle)
        seriousness = self._infer_topic_seriousness(bundle, task_type)
        task_fit_a = self._main_task_fit(cand_a, task_type, seriousness)
        task_fit_b = self._main_task_fit(cand_b, task_type, seriousness)

        info: Dict[str, Any] = {
            "task_type": task_type,
            "task_type_label": task_type,
            "topic_seriousness": seriousness,
            "close_score_conflict": False,
            "thresholds": {
                "close_score_delta_main": self.close_score_delta_main,
                "close_query_focus_delta": self.close_query_focus_delta,
                "close_topic_focus_delta": self.close_topic_focus_delta,
            },
            "top2_candidates": [
                {
                    "agent_id": cand_a.agent_id,
                    "name": cand_a.name,
                    "final_score": score_a.final_score,
                    "query_focus": score_a.query_focus,
                    "topic_focus": score_a.topic_focus,
                    "repeat_penalty": score_a.repeat_penalty,
                    "scene_prior": scene_prior_a,
                    "task_fit": task_fit_a,
                },
                {
                    "agent_id": cand_b.agent_id,
                    "name": cand_b.name,
                    "final_score": score_b.final_score,
                    "query_focus": score_b.query_focus,
                    "topic_focus": score_b.topic_focus,
                    "repeat_penalty": score_b.repeat_penalty,
                    "scene_prior": scene_prior_b,
                    "task_fit": task_fit_b,
                },
            ],
        }

        forced_task_fit_types = {"decision_support", "clarification"}
        if task_type in forced_task_fit_types and abs(task_fit_a - task_fit_b) >= 0.25:
            winner = cand_a if task_fit_a >= task_fit_b else cand_b
            info["decision_reason"] = "task_fit_override"
            info["winner"] = winner.agent_id
            return winner, info

        if not self._is_close_score(score_a.final_score, score_b.final_score, self.close_score_delta_main):
            winner = cand_a if score_a.final_score >= score_b.final_score else cand_b
            info["decision_reason"] = "higher_final_score"
            info["winner"] = winner.agent_id
            return winner, info

        info["close_score_conflict"] = True

        task_fit_margin = 0.05 if task_type in {"decision_support", "clarification", "relationship_reading"} else 0.08
        if abs(task_fit_a - task_fit_b) >= task_fit_margin:
            winner = cand_a if task_fit_a >= task_fit_b else cand_b
            info["decision_reason"] = "close_score_then_task_seriousness_fit"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(scene_prior_a - scene_prior_b) >= 0.08:
            winner = cand_a if scene_prior_a >= scene_prior_b else cand_b
            info["decision_reason"] = "close_score_then_scene_role_prior"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(score_a.query_focus - score_b.query_focus) >= self.close_query_focus_delta:
            winner = cand_a if score_a.query_focus >= score_b.query_focus else cand_b
            info["decision_reason"] = "close_score_then_query_focus"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(score_a.topic_focus - score_b.topic_focus) >= self.close_topic_focus_delta:
            winner = cand_a if score_a.topic_focus >= score_b.topic_focus else cand_b
            info["decision_reason"] = "close_score_then_topic_focus"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(score_a.repeat_penalty - score_b.repeat_penalty) > 1e-9:
            winner = cand_a if score_a.repeat_penalty <= score_b.repeat_penalty else cand_b
            info["decision_reason"] = "close_score_then_less_recent"
            info["winner"] = winner.agent_id
            return winner, info

        winner = ranked_candidates[0]
        info["decision_reason"] = "close_score_then_agent_list_order"
        info["winner"] = winner.agent_id
        return winner, info

    def _resolve_supplement_close_conflict(
        self,
        ranked_candidates: Sequence[Agent],
        score_map: Dict[str, ScoreDetail],
        main_agent: Optional[Agent],
        bundle: TopicBundle,
    ) -> Tuple[Optional[Agent], Dict[str, Any]]:
        if not ranked_candidates:
            return None, {"decision_reason": "no_candidate", "close_score_conflict": False}

        if len(ranked_candidates) == 1:
            only_agent = ranked_candidates[0]
            only_score = score_map[only_agent.agent_id]
            return only_agent, {
                "decision_reason": "single_candidate",
                "close_score_conflict": False,
                "winner": only_agent.agent_id,
                "winner_score": only_score.final_score,
            }

        cand_a, cand_b = ranked_candidates[0], ranked_candidates[1]
        score_a = score_map[cand_a.agent_id]
        score_b = score_map[cand_b.agent_id]
        diversity_a = self._diversity_from_main(cand_a, main_agent)
        diversity_b = self._diversity_from_main(cand_b, main_agent)
        functional_fit_a = self._supplement_functional_fit(cand_a, main_agent, bundle)
        functional_fit_b = self._supplement_functional_fit(cand_b, main_agent, bundle)
        task_type = self._classify_task_type(bundle)
        seriousness = self._infer_topic_seriousness(bundle, task_type)
        task_fit_a = self._supplement_task_fit(cand_a, main_agent, task_type, seriousness)
        task_fit_b = self._supplement_task_fit(cand_b, main_agent, task_type, seriousness)

        info: Dict[str, Any] = {
            "task_type": task_type,
            "task_type_label": task_type,
            "topic_seriousness": seriousness,
            "close_score_conflict": False,
            "thresholds": {
                "close_score_delta_supplement": self.close_score_delta_supplement,
                "close_diversity_delta": self.close_diversity_delta,
                "close_topic_focus_delta": self.close_topic_focus_delta,
            },
            "top2_candidates": [
                {
                    "agent_id": cand_a.agent_id,
                    "name": cand_a.name,
                    "final_score": score_a.final_score,
                    "topic_focus": score_a.topic_focus,
                    "diversity_from_main": diversity_a,
                    "functional_fit": functional_fit_a,
                    "task_fit": task_fit_a,
                    "repeat_penalty": score_a.repeat_penalty,
                },
                {
                    "agent_id": cand_b.agent_id,
                    "name": cand_b.name,
                    "final_score": score_b.final_score,
                    "topic_focus": score_b.topic_focus,
                    "diversity_from_main": diversity_b,
                    "functional_fit": functional_fit_b,
                    "task_fit": task_fit_b,
                    "repeat_penalty": score_b.repeat_penalty,
                },
            ],
        }

        if not self._is_close_score(score_a.final_score, score_b.final_score, self.close_score_delta_supplement):
            winner = cand_a if score_a.final_score >= score_b.final_score else cand_b
            info["decision_reason"] = "higher_final_score"
            info["winner"] = winner.agent_id
            return winner, info

        info["close_score_conflict"] = True

        if abs(task_fit_a - task_fit_b) >= 0.08:
            winner = cand_a if task_fit_a >= task_fit_b else cand_b
            info["decision_reason"] = "close_score_then_task_seriousness_fit"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(functional_fit_a - functional_fit_b) >= 0.10:
            winner = cand_a if functional_fit_a >= functional_fit_b else cand_b
            info["decision_reason"] = "close_score_then_functional_fit"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(diversity_a - diversity_b) >= self.close_diversity_delta:
            winner = cand_a if diversity_a >= diversity_b else cand_b
            info["decision_reason"] = "close_score_then_diversity_from_main"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(score_a.topic_focus - score_b.topic_focus) >= self.close_topic_focus_delta:
            winner = cand_a if score_a.topic_focus >= score_b.topic_focus else cand_b
            info["decision_reason"] = "close_score_then_topic_focus"
            info["winner"] = winner.agent_id
            return winner, info

        if abs(score_a.repeat_penalty - score_b.repeat_penalty) > 1e-9:
            winner = cand_a if score_a.repeat_penalty <= score_b.repeat_penalty else cand_b
            info["decision_reason"] = "close_score_then_less_recent"
            info["winner"] = winner.agent_id
            return winner, info

        winner = ranked_candidates[0]
        info["decision_reason"] = "close_score_then_agent_list_order"
        info["winner"] = winner.agent_id
        return winner, info

    def _sorted_main_candidates(
        self,
        agents: List[Agent],
        score_map: Dict[str, ScoreDetail],
        strict_only: bool = True,
    ) -> List[Agent]:
        order_idx = {agent.agent_id: idx for idx, agent in enumerate(agents)}
        candidates: List[Agent] = []
        for agent in agents:
            score = score_map.get(agent.agent_id)
            if agent.status != "online" or score is None:
                continue
            if strict_only and (agent.can_check or agent.can_summarize):
                continue
            if score.relevance < self.min_main_relevance or score.final_score < self.min_main_score:
                continue
            candidates.append(agent)
        return sorted(
            candidates,
            key=lambda a: self._main_rank_key(score_map[a.agent_id], order_idx[a.agent_id]),
            reverse=True,
        )

    def _sorted_any_main_candidates(
        self,
        agents: List[Agent],
        score_map: Dict[str, ScoreDetail],
        strict_only: bool = True,
    ) -> List[Agent]:
        order_idx = {agent.agent_id: idx for idx, agent in enumerate(agents)}
        candidates: List[Agent] = []
        for agent in agents:
            score = score_map.get(agent.agent_id)
            if agent.status != "online" or score is None:
                continue
            if strict_only and (agent.can_check or agent.can_summarize):
                continue
            candidates.append(agent)
        return sorted(
            candidates,
            key=lambda a: self._main_rank_key(score_map[a.agent_id], order_idx[a.agent_id]),
            reverse=True,
        )

    def _sorted_supplement_candidates(
        self,
        agents: List[Agent],
        score_map: Dict[str, ScoreDetail],
        selected_agent_ids: Sequence[str],
        main_agent: Optional[Agent],
        bundle: TopicBundle,
    ) -> List[Agent]:
        order_idx = {agent.agent_id: idx for idx, agent in enumerate(agents)}
        blocked = set(selected_agent_ids)
        candidates: List[Agent] = []
        for agent in agents:
            score = score_map.get(agent.agent_id)
            if agent.status != "online" or score is None:
                continue
            if agent.agent_id in blocked:
                continue
            if agent.can_check or agent.can_summarize:
                continue
            if score.relevance < self.min_supplement_relevance or score.final_score < self.min_supplement_score:
                continue
            candidates.append(agent)
        return sorted(
            candidates,
            key=lambda a: self._supplement_rank_key(score_map[a.agent_id], order_idx[a.agent_id], main_agent, a, bundle),
            reverse=True,
        )

    def _ranking_trace(
        self,
        ranked_agents: Sequence[Agent],
        score_map: Dict[str, ScoreDetail],
        role: str,
        main_agent: Optional[Agent] = None,
    ) -> List[Dict[str, Any]]:
        order_idx = {agent.agent_id: idx for idx, agent in enumerate(ranked_agents)}
        trace: List[Dict[str, Any]] = []
        for agent in ranked_agents:
            score = score_map.get(agent.agent_id)
            if score is None:
                continue
            item = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "role_candidate": role,
                "final_score": score.final_score,
                "relevance": score.relevance,
                "query_focus": score.query_focus,
                "topic_focus": score.topic_focus,
                "repeat_penalty": score.repeat_penalty,
                "can_check": agent.can_check,
                "can_summarize": agent.can_summarize,
                "rank_position": order_idx.get(agent.agent_id, 0) + 1,
            }
            if role == ROLE_SUPPLEMENT and main_agent is not None:
                item["diversity_from_main"] = round(1.0 - self._agent_profile_similarity(agent, main_agent), 4)
            trace.append(item)
        return trace

    # =========================
    # check 层：DDPE 风格 relation + WHoW 风格 moderator act
    # =========================

    def _negation_flag(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(cue in lowered for cue in NEGATION_CUES)

    def _has_contrast_cue(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(cue in lowered for cue in CONTRAST_CUES)

    def _best_topic_match(self, text: str, active_topics: Sequence[str]) -> Tuple[Optional[str], float]:
        if not text or not active_topics:
            return None, 0.0
        scores = [(topic, self._sim(text, topic)) for topic in active_topics if topic]
        if not scores:
            return None, 0.0
        topic, score = max(scores, key=lambda x: x[1])
        return topic, round(score, 4)

    def _relation_explanation(
        self,
        relation: str,
        topic_a: Optional[str],
        topic_b: Optional[str],
        text_a: str,
        text_b: str,
    ) -> str:
        if relation == "topic_shift":
            return f"两条发言分别落在『{topic_a or '话题A'}』和『{topic_b or '话题B'}』，属于子话题分裂。"
        if relation == "correction":
            return "两条发言围绕同一子话题展开，但极性相反，更像直接纠正或反对。"
        if relation == "contrast":
            return "两条发言讨论同一方向，但切入点不同，像对比而不是直接互斥。"
        if relation == "supplement":
            return "两条发言在同一子话题上相近，更像补充或顺承。"
        short_a = text_a[:24]
        short_b = text_b[:24]
        return f"『{short_a}』与『{short_b}』之间的关系需要进一步核实。"

    def _moderator_act(self, relation: str) -> str:
        if relation == "topic_shift":
            return "instruction"
        if relation == "correction":
            return "confronting"
        if relation == "contrast":
            return "probing"
        return "interpretation"

    def _relation_context_flag(self, bundle: TopicBundle) -> bool:
        text = " ".join([
            bundle.user_query or "",
            bundle.context_query or "",
            bundle.topic_summary or "",
            " ".join(bundle.active_topics or []),
        ])
        relation_cues = ["分歧", "核实", "contrast", "correction", "topic_shift", "关系核实", "反对", "纠正", "不同路线"]
        return any(cue in text for cue in relation_cues)

    def _relation_label(
        self,
        pair_sim: float,
        same_topic: bool,
        negation_flip: bool,
        contrast_cue: bool,
        topic_split: bool,
        relation_context: bool = False,
    ) -> Optional[str]:
        if topic_split and relation_context and (contrast_cue or negation_flip or pair_sim >= 0.08):
            return "contrast"
        if topic_split:
            return "topic_shift"
        if (same_topic or relation_context) and pair_sim >= self.conflict_similarity_threshold and negation_flip:
            return "correction"
        if (same_topic or relation_context) and pair_sim >= 0.10 and (contrast_cue or negation_flip or relation_context):
            return "contrast"
        if same_topic and pair_sim >= 0.14 and not negation_flip:
            return "supplement"
        return None

    def _detect_check_relations(
        self,
        bundle: TopicBundle,
        last_round_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if len(last_round_outputs) < 2:
            return {
                "need_check": False,
                "pair_relations": [],
                "relation_counter": {},
                "highest_risk": "low",
            }

        topic_counter: Counter[str] = Counter()
        relation_context = self._relation_context_flag(bundle)
        enriched: List[Dict[str, Any]] = []
        for item in last_round_outputs:
            combined_text = " ".join([
                item.get("text", "") or "",
                " ".join(item.get("key_points", []) or []),
            ]).strip()
            top_topic, top_score = self._best_topic_match(combined_text, bundle.active_topics)
            lexical_hit = self._lexical_overlap_ratio(combined_text, " ".join(bundle.active_topics))
            if top_topic and max(top_score, lexical_hit) >= 0.12:
                topic_counter[top_topic] += 1
            enriched.append(
                {
                    "agent_id": item.get("agent_id"),
                    "text": combined_text,
                    "top_topic": top_topic,
                    "top_topic_score": max(top_score, lexical_hit),
                    "negation": self._negation_flag(combined_text),
                    "contrast_cue": self._has_contrast_cue(combined_text),
                }
            )

        pair_relations: List[Dict[str, Any]] = []
        relation_counter: Counter[str] = Counter()
        highest_risk = "low"

        for i in range(len(enriched)):
            for j in range(i + 1, len(enriched)):
                a = enriched[i]
                b = enriched[j]
                pair_sim = self._sim(a["text"], b["text"])
                same_topic = bool(a["top_topic"] and a["top_topic"] == b["top_topic"])
                topic_split = bool(
                    a["top_topic"]
                    and b["top_topic"]
                    and a["top_topic"] != b["top_topic"]
                    and a["top_topic_score"] >= 0.14
                    and b["top_topic_score"] >= 0.14
                )
                relation = self._relation_label(
                    pair_sim=pair_sim,
                    same_topic=same_topic,
                    negation_flip=a["negation"] != b["negation"],
                    contrast_cue=a["contrast_cue"] or b["contrast_cue"],
                    topic_split=topic_split,
                    relation_context=relation_context,
                )
                if relation is None:
                    continue

                if relation == "correction":
                    risk = "high"
                elif relation in {"contrast", "topic_shift"}:
                    risk = "medium"
                else:
                    risk = "low"

                if risk == "high":
                    highest_risk = "high"
                elif risk == "medium" and highest_risk == "low":
                    highest_risk = "medium"

                relation_counter[relation] += 1
                pair_relations.append(
                    {
                        "agent_a": a["agent_id"],
                        "agent_b": b["agent_id"],
                        "pair_similarity": pair_sim,
                        "relation": relation,
                        "risk": risk,
                        "topic_a": a["top_topic"],
                        "topic_b": b["top_topic"],
                        "explanation": self._relation_explanation(
                            relation=relation,
                            topic_a=a["top_topic"],
                            topic_b=b["top_topic"],
                            text_a=a["text"],
                            text_b=b["text"],
                        ),
                        "moderator_act": self._moderator_act(relation),
                    }
                )

        need_check = any(rel["risk"] in {"medium", "high"} for rel in pair_relations)
        return {
            "need_check": need_check,
            "pair_relations": pair_relations,
            "relation_counter": dict(relation_counter),
            "topic_counter": dict(topic_counter),
            "highest_risk": highest_risk,
        }

    # =========================
    # 冗余与收束
    # =========================

    def _redundancy_ratio(
        self,
        bundle: TopicBundle,
        history_summary: Dict[str, Any],
        last_round_outputs: List[Dict[str, Any]],
    ) -> float:
        done_points = self._filter_points_for_current_topic(history_summary.get("done_points", []) or [], bundle, min_relevance=0.18)
        last_points = self._filter_points_for_current_topic(self._collect_last_round_key_points(last_round_outputs), bundle, min_relevance=0.18)

        if not last_points or not done_points:
            return 0.0

        overlaps = [self._max_sim_to_list(point, done_points) for point in last_points if point]
        if not overlaps:
            return 0.0
        return round(sum(overlaps) / len(overlaps), 4)

    def _coverage_state(
        self,
        bundle: TopicBundle,
        history_summary: Dict[str, Any],
        last_round_outputs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        done_points = self._filter_points_for_current_topic(history_summary.get("done_points", []) or [], bundle, min_relevance=0.10)
        last_points = self._filter_points_for_current_topic(self._collect_last_round_key_points(last_round_outputs), bundle, min_relevance=0.10)

        total_points = len(done_points) + len(last_points)
        return {
            "total_points": total_points,
            "current_topic_done_points": done_points,
            "current_topic_last_points": last_points,
            "enough_points_collected": total_points >= self.summary_points_threshold,
            "pre_summary_ready": bundle.allow_pre_summary and total_points >= self.pre_summary_points_threshold,
            "allow_pre_summary": bundle.allow_pre_summary,
            "user_query_resolved": bundle.resolved,
        }

    def _check_trigger_reason(
        self,
        relation_info: Dict[str, Any],
        coverage_state: Dict[str, Any],
        redundancy_ratio: float,
    ) -> Tuple[bool, Optional[str]]:
        if relation_info["need_check"]:
            return True, CHECK_REASON_RELATION_VERIFY
        if self.medium_redundancy_threshold <= redundancy_ratio < self.high_redundancy_threshold:
            return True, CHECK_REASON_REDUNDANCY
        if coverage_state["pre_summary_ready"] and not coverage_state["user_query_resolved"]:
            return True, CHECK_REASON_PRE_SUMMARY
        return False, None

    def _summary_trigger_reason(
        self,
        coverage_state: Dict[str, Any],
        redundancy_ratio: float,
    ) -> Tuple[bool, Optional[str]]:
        reason: Optional[str] = None
        if redundancy_ratio >= self.high_redundancy_threshold:
            reason = SUMMARY_REASON_REDUNDANCY
        elif coverage_state["user_query_resolved"]:
            reason = SUMMARY_REASON_RESOLVED
        return reason is not None, reason

    # =========================
    # 角色挑选
    # =========================

    def _pick_summary_agent(self, agents: List[Agent], selected_agent_ids: List[str]) -> Optional[Agent]:
        for agent in agents:
            if agent.status == "online" and agent.can_summarize and agent.agent_id not in selected_agent_ids:
                return agent
        for agent in agents:
            if agent.status == "online" and agent.agent_id not in selected_agent_ids and any(k in agent.name.lower() for k in SUMMARY_NAME_HINTS):
                return agent
        return None

    def _pick_check_agent(self, agents: List[Agent], selected_agent_ids: List[str]) -> Optional[Agent]:
        for agent in agents:
            if agent.status == "online" and agent.can_check and agent.agent_id not in selected_agent_ids:
                return agent
        for agent in agents:
            if agent.status == "online" and agent.agent_id not in selected_agent_ids and any(k in agent.name.lower() for k in CHECK_NAME_HINTS):
                return agent
        for agent in agents:
            if agent.status == "online" and agent.can_summarize and agent.agent_id not in selected_agent_ids:
                return agent
        return None

    def _return_target_text(self, bundle: TopicBundle) -> str:
        return bundle.previous_main_topic or bundle.main_topic or "上一个稳定话题"

    def _next_topic_state_after_return(self, bundle: TopicBundle) -> Dict[str, Any]:
        return {
            "main_topic": self._return_target_text(bundle),
            "active_topic": bundle.previous_active_topics or bundle.active_topics,
            "topic_summary": bundle.previous_topic_summary or bundle.topic_summary,
            "topic_action": "maintain",
        }

    def _next_topic_state_after_end(self, bundle: TopicBundle) -> Dict[str, Any]:
        return {
            "main_topic": bundle.main_topic,
            "active_topic": bundle.active_topics,
            "topic_summary": bundle.topic_summary,
            "topic_action": "end",
            "resolved": True,
            "session_status": "closed",
        }

    def _next_topic_state_after_maintain(self, bundle: TopicBundle) -> Dict[str, Any]:
        return {
            "main_topic": bundle.main_topic,
            "active_topic": bundle.active_topics,
            "topic_summary": bundle.topic_summary,
            "topic_action": "maintain",
            "topic_transition": "maintain",
            "scene_mode": bundle.scene_mode,
            "previous_main_topic": bundle.previous_main_topic,
            "previous_active_topic": bundle.previous_active_topics,
            "previous_topic_summary": bundle.previous_topic_summary,
        }

    def _instruction_for_role(
        self,
        role: str,
        agent: Agent,
        payload: Dict[str, Any],
        check_reason: Optional[str],
        summary_reason: Optional[str],
        relation_info: Optional[Dict[str, Any]] = None,
        bundle: Optional[TopicBundle] = None,
    ) -> str:
        query = payload.get("query", "") or ""
        topic_state = payload.get("topic_state", {}) or {}

        if role == ROLE_MAIN:
            return (
                f"{agent.name} 围绕『{query}』给出主答，优先回应主话题『{topic_state.get('main_topic', '')}』，"
                "只输出 1~2 个最关键观点，不展开无关支线。"
            )

        if role == ROLE_SUPPLEMENT:
            return f"{agent.name} 只补充 1 个主答未覆盖的新点，避免重复上一轮说过的内容。"

        if role == ROLE_CHECK:
            if check_reason == CHECK_REASON_RELATION_VERIFY:
                relation_hint = ""
                pair_relations = (relation_info or {}).get("pair_relations", [])[:2]
                if pair_relations:
                    parts = [
                        f"{item['agent_a']} vs {item['agent_b']} = {item['relation']}（{item['moderator_act']}）"
                        for item in pair_relations
                    ]
                    relation_hint = "；".join(parts)
                return (
                    f"{agent.name} 负责冲突核实：先按 DDPE 风格判断前文关系属于 supplement / contrast / correction / topic_shift 中哪一类，"
                    f"再给出一句 explanation；随后按 WHoW 风格选择更合适的主持动作（probing / confronting / interpretation / instruction）。"
                    f"{(' 当前候选关系：' + relation_hint) if relation_hint else ''}"
                )
            if check_reason == CHECK_REASON_REDUNDANCY:
                return f"{agent.name} 负责检查重复内容，标出 keep / drop / merge 的观点。"
            return f"{agent.name} 负责检查当前覆盖度：已回答什么、还缺什么、是否已可进入总结。"

        if role == ROLE_SUMMARY:
            if summary_reason == SUMMARY_REASON_TOPIC_END:
                return (
                    f"{agent.name} 用 1~2 句自然、礼貌地向用户告别，明确这轮对话到这里结束。"
                    "不要继续分析，不要开启新话题，不要再展开建议。"
                )
            if summary_reason == SUMMARY_REASON_TOPIC_RETURN:
                target_text = self._return_target_text(bundle or TopicBundle("", "", "", [], "", "maintain", "maintain", False, "general_chat", [], False, "", [], ""))
                return (
                    f"{agent.name} 先指出当前发言已经偏题，再用 2~3 句把大家拉回『{target_text}』，"
                    "并给出一个继续往下聊的切入点，不要结束整个对话。"
                )
            if summary_reason == SUMMARY_REASON_REDUNDANCY:
                return f"{agent.name} 负责压缩重复内容，只保留最有价值的结论。"
            if summary_reason == SUMMARY_REASON_RESOLVED:
                return f"{agent.name} 负责总结最终答案，并明确当前话题可以暂时收束。"
            return f"{agent.name} 负责阶段总结：提炼已经达成的关键结论，并指出下一步继续聊什么。"

        return f"{agent.name} 发言。"

    def _address_display_names(self, agent_ids: Sequence[str], agent_map: Dict[str, Agent]) -> List[str]:
        names: List[str] = []
        for agent_id in agent_ids:
            if agent_id == "user":
                names.append("用户")
            elif agent_id == "all":
                names.append("全体")
            else:
                agent = agent_map.get(agent_id)
                names.append(agent.name if agent else agent_id)
        return names

    def _opening_hint_for_step(self, role: str, task_type: str, seriousness: str, address_to_names: Sequence[str]) -> str:
        target = "用户" if (not address_to_names or "用户" in address_to_names) else "、".join(address_to_names)
        if role == ROLE_MAIN:
            if task_type == "emotional_support":
                return "先直接接住用户的情绪，例如：‘我知道，你现在是真的很累。’"
            if task_type == "relationship_reading":
                return "先对用户的感受做判断和安抚，例如：‘我能理解你为什么会这么想。’"
            if task_type == "decision_support":
                return "先替用户把选择问题框清楚，例如：‘这件事其实可以先拆成两个判断。’"
            if task_type == "clarification":
                return "先直接回应用户的问题核心，例如：‘这里的关键不是A，而是B。’"
            if task_type == "practical_help":
                return "先给用户一个可执行方向，例如：‘这件事你可以先做第一步。’"
            if task_type == "light_trouble":
                return "先轻接用户当前的小麻烦，例如：‘这确实是个小麻烦，但先别慌。’"
            return "先正面回应用户当前的问题。"
        if role == ROLE_SUPPLEMENT:
            if task_type in {"emotional_support", "relationship_reading", "clarification"}:
                return "先承接 main 对用户的回应，再补一个新增点，不要自己重新开主线。"
            return f"先接前一位的话，再给{target}补一个不同角度。"
        return ""

    def _address_style_for_step(self, role: str, task_type: str, seriousness: str) -> str:
        if role == ROLE_MAIN:
            if task_type in {"emotional_support", "relationship_reading"}:
                return "direct_user_empathy"
            if task_type in {"practical_help", "decision_support", "clarification"}:
                return "direct_user_solution"
            return "default"
        if role == ROLE_SUPPLEMENT:
            if task_type in {"emotional_support", "relationship_reading", "clarification"}:
                return "support_user_after_main"
            return "continue_from_main"
        return "default"

    def _build_step_address_spec(
        self,
        role: str,
        agent_id: str,
        speaker_order: Sequence[str],
        agent_map: Dict[str, Agent],
        relation_info: Optional[Dict[str, Any]] = None,
        summary_reason: Optional[str] = None,
        bundle: Optional[TopicBundle] = None,
    ) -> Dict[str, Any]:
        previous_speakers = [x for x in speaker_order if x != agent_id]
        main_agent_id = speaker_order[0] if speaker_order else None
        task_type = self._classify_task_type(bundle) if bundle is not None else "open_discussion"
        seriousness = self._infer_topic_seriousness(bundle, task_type) if bundle is not None else "low"

        if role == ROLE_MAIN:
            address_to = ["user"]
            note = "主答优先直接回应用户。"
        elif role == ROLE_SUPPLEMENT:
            if task_type in {"emotional_support", "relationship_reading", "clarification"}:
                address_to = ["user"]
                note = "高严肃度场景下，补充者继续面向用户说话。"
            else:
                address_to = [main_agent_id] if main_agent_id and main_agent_id != agent_id else ["user"]
                note = "补充者优先接主答继续说。"
        elif role == ROLE_CHECK:
            pair_relations = (relation_info or {}).get("pair_relations", [])
            conflict_targets: List[str] = []
            for item in pair_relations[:2]:
                for one_id in (item.get("agent_a"), item.get("agent_b")):
                    if one_id and one_id != agent_id and one_id not in conflict_targets:
                        conflict_targets.append(one_id)
            address_to = conflict_targets or previous_speakers or ["all"]
            note = "检查者优先对存在关系风险的发言人说。"
        elif role == ROLE_SUMMARY:
            if summary_reason == SUMMARY_REASON_TOPIC_END:
                address_to = ["user"]
                note = "结束场景下，总结者直接对用户礼貌告别。"
            elif summary_reason in {SUMMARY_REASON_TOPIC_RETURN, SUMMARY_REASON_RESOLVED}:
                address_to = ["all", "user"]
                note = "总结者面向全体和用户做收束。"
            else:
                address_to = ["all"]
                note = "总结者面向全体做阶段收束。"
        else:
            address_to = ["all"]
            note = "默认面向全体发言。"

        address_names = self._address_display_names(address_to, agent_map)
        return {
            "task_type": task_type,
            "topic_seriousness": seriousness,
            "address_to": address_to,
            "address_to_names": address_names,
            "address_note": note,
            "address_style": self._address_style_for_step(role, task_type, seriousness),
            "opening_hint": self._opening_hint_for_step(role, task_type, seriousness, address_names),
        }

    def _build_skip_mute(
        self,
        agents: List[Agent],
        selected_agent_ids: List[str],
        score_map: Dict[str, ScoreDetail],
        recent_speakers: set[str],
    ) -> List[Dict[str, str]]:
        skip_mute: List[Dict[str, str]] = []
        for agent in agents:
            if agent.agent_id in selected_agent_ids:
                continue

            score = score_map.get(agent.agent_id)
            if agent.status == "muted":
                skip_mute.append({"agent_id": agent.agent_id, "action": ACTION_MUTE, "reason": SKIP_REASON_QUOTA_LIMIT})
                continue

            if agent.status != "online":
                skip_mute.append({"agent_id": agent.agent_id, "action": ACTION_SKIP, "reason": SKIP_REASON_QUOTA_LIMIT})
                continue

            low_relevance_floor = max(self.min_supplement_relevance + 0.04, 0.18)
            if score is None or score.relevance < low_relevance_floor:
                reason = SKIP_REASON_LOW_RELEVANCE
            elif agent.agent_id in recent_speakers:
                reason = SKIP_REASON_REDUNDANCY
            else:
                reason = SKIP_REASON_QUOTA_LIMIT

            skip_mute.append({"agent_id": agent.agent_id, "action": ACTION_SKIP, "reason": reason})
        return skip_mute

    def _direct_summary_return_plan(
        self,
        agents: List[Agent],
        payload: Dict[str, Any],
        bundle: TopicBundle,
    ) -> Dict[str, Any]:
        summary_agent = self._pick_summary_agent(agents, selected_agent_ids=[]) or self._pick_check_agent(agents, selected_agent_ids=[])
        if summary_agent is None:
            online_agents = [agent for agent in agents if agent.status == "online"]
            if not online_agents:
                return self._empty_plan()
            summary_agent = online_agents[0]

        speaker_order = [summary_agent.agent_id]
        role_assignment = [{"agent_id": summary_agent.agent_id, "role": ROLE_SUMMARY}]
        agent_map = {agent.agent_id: agent for agent in agents}
        step_address = self._build_step_address_spec(
            role=ROLE_SUMMARY,
            agent_id=summary_agent.agent_id,
            speaker_order=speaker_order,
            agent_map=agent_map,
            relation_info=None,
            summary_reason=SUMMARY_REASON_TOPIC_RETURN,
            bundle=bundle,
        )
        execution_steps = [
            {
                "step_id": 1,
                "agent_id": summary_agent.agent_id,
                "role": ROLE_SUMMARY,
                "instruction": self._instruction_for_role(
                    role=ROLE_SUMMARY,
                    agent=summary_agent,
                    payload=payload,
                    check_reason=None,
                    summary_reason=SUMMARY_REASON_TOPIC_RETURN,
                    relation_info=None,
                    bundle=bundle,
                ),
                **step_address,
            }
        ]

        skip_mute = self._build_skip_mute(agents, speaker_order, {}, set())
        next_topic_state = self._next_topic_state_after_return(bundle)

        return {
            "quota": {"max_speakers": 1, "per_agent_max_turn": 1},
            "role_assignment": role_assignment,
            "speaker_order": speaker_order,
            "skip_mute": skip_mute,
            "check_trigger": {"triggered": False, "reason": None},
            "summary_trigger": {"triggered": True, "reason": SUMMARY_REASON_TOPIC_RETURN},
            "final_plan": {
                "selected_agents": speaker_order,
                "execution_steps": execution_steps,
                "next_topic_state": next_topic_state,
                "planner_meta": {
                    "topic_action": bundle.topic_action,
                    "forced_summary": True,
                    "summary_reason": SUMMARY_REASON_TOPIC_RETURN,
                    "check_triggered": False,
                },
            },
            "ranking": {
                "main": [],
                "supplement": [],
                "decision": {"mode": "force_return_summary", "selected_summary_agent": summary_agent.agent_id},
            },
            "diagnostics": {
                "topic_text": self._topic_text(bundle),
                "topic_action": bundle.topic_action,
                "direct_return_to_topic": True,
                "return_target_topic": next_topic_state["main_topic"],
                "score_details": [],
            },
        }

    def _direct_end_plan(
        self,
        agents: List[Agent],
        payload: Dict[str, Any],
        bundle: TopicBundle,
    ) -> Dict[str, Any]:
        summary_agent = self._pick_summary_agent(agents, selected_agent_ids=[]) or self._pick_check_agent(agents, selected_agent_ids=[])
        if summary_agent is None:
            online_agents = [agent for agent in agents if agent.status == "online"]
            if not online_agents:
                return self._empty_plan()
            summary_agent = online_agents[0]

        speaker_order = [summary_agent.agent_id]
        role_assignment = [{"agent_id": summary_agent.agent_id, "role": ROLE_SUMMARY}]
        agent_map = {agent.agent_id: agent for agent in agents}
        step_address = self._build_step_address_spec(
            role=ROLE_SUMMARY,
            agent_id=summary_agent.agent_id,
            speaker_order=speaker_order,
            agent_map=agent_map,
            relation_info=None,
            summary_reason=SUMMARY_REASON_TOPIC_END,
            bundle=bundle,
        )
        execution_steps = [
            {
                "step_id": 1,
                "agent_id": summary_agent.agent_id,
                "role": ROLE_SUMMARY,
                "instruction": self._instruction_for_role(
                    role=ROLE_SUMMARY,
                    agent=summary_agent,
                    payload=payload,
                    check_reason=None,
                    summary_reason=SUMMARY_REASON_TOPIC_END,
                    relation_info=None,
                    bundle=bundle,
                ),
                **step_address,
            }
        ]

        skip_mute = self._build_skip_mute(agents, speaker_order, {}, set())
        next_topic_state = self._next_topic_state_after_end(bundle)

        return {
            "quota": {"max_speakers": 1, "per_agent_max_turn": 1},
            "role_assignment": role_assignment,
            "speaker_order": speaker_order,
            "skip_mute": skip_mute,
            "check_trigger": {"triggered": False, "reason": None},
            "summary_trigger": {"triggered": True, "reason": SUMMARY_REASON_TOPIC_END},
            "final_plan": {
                "selected_agents": speaker_order,
                "execution_steps": execution_steps,
                "next_topic_state": next_topic_state,
                "planner_meta": {
                    "topic_action": bundle.topic_action,
                    "forced_summary": True,
                    "summary_reason": SUMMARY_REASON_TOPIC_END,
                    "check_triggered": False,
                },
            },
            "ranking": {
                "main": [],
                "supplement": [],
                "decision": {"mode": "force_end_summary", "selected_summary_agent": summary_agent.agent_id},
            },
            "diagnostics": {
                "topic_text": self._topic_text(bundle),
                "topic_action": bundle.topic_action,
                "direct_end_conversation": True,
                "score_details": [],
            },
        }

    def _empty_plan(self) -> Dict[str, Any]:
        return {
            "quota": {"max_speakers": 0, "per_agent_max_turn": 0},
            "role_assignment": [],
            "speaker_order": [],
            "skip_mute": [],
            "check_trigger": {"triggered": False, "reason": None},
            "summary_trigger": {"triggered": False, "reason": None},
            "final_plan": {"selected_agents": [], "execution_steps": []},
            "diagnostics": {},
        }

    # =========================
    # 主入口
    # =========================

    def plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        agents = self._parse_agents(payload.get("agent_list", []))
        if not agents:
            return self._empty_plan()

        history_summary = payload.get("history_summary", {}) or {}
        last_round_outputs = payload.get("last_round_outputs", []) or []
        bundle = self._parse_topic_bundle(payload)

        online_agents = [a for a in agents if a.status == "online"]
        if not online_agents:
            result = self._empty_plan()
            result["skip_mute"] = [
                {
                    "agent_id": agent.agent_id,
                    "action": ACTION_MUTE if agent.status == "muted" else ACTION_SKIP,
                    "reason": SKIP_REASON_QUOTA_LIMIT,
                }
                for agent in agents
            ]
            return result

        if bundle.topic_action == "return":
            return self._direct_summary_return_plan(agents=agents, payload=payload, bundle=bundle)

        if bundle.topic_action == "end":
            return self._direct_end_plan(agents=agents, payload=payload, bundle=bundle)

        redundancy_ratio = self._redundancy_ratio(bundle, history_summary, last_round_outputs)
        relation_info = self._detect_check_relations(bundle, last_round_outputs)
        coverage_state = self._coverage_state(bundle, history_summary, last_round_outputs)

        summary_triggered, summary_reason = self._summary_trigger_reason(
            coverage_state=coverage_state,
            redundancy_ratio=redundancy_ratio,
        )
        check_triggered, check_reason = (False, None)
        if not summary_triggered:
            check_triggered, check_reason = self._check_trigger_reason(
                relation_info=relation_info,
                coverage_state=coverage_state,
                redundancy_ratio=redundancy_ratio,
            )

        max_speakers = min(3 if (summary_triggered or check_triggered) else 2, len(online_agents))
        scores = self._score_agents(online_agents, bundle, last_round_outputs, payload=payload)
        score_map = {s.agent_id: s for s in scores}
        recent_speakers = self._get_recent_speakers(last_round_outputs)
        agent_map = {a.agent_id: a for a in agents}

        main_agent: Optional[Agent] = None
        supplements: List[Agent] = []
        check_agent: Optional[Agent] = None
        summary_agent: Optional[Agent] = None

        # 主答优先从普通内容型 agent 中选
        main_candidates = self._sorted_main_candidates(agents, score_map, strict_only=True)
        if not main_candidates:
            main_candidates = self._sorted_any_main_candidates(agents, score_map, strict_only=True)

        # 功能型 agent 默认不参与 main。
        # 只有普通内容型候选明显过弱，才允许 check / summary 型 agent 回到 main 兜底。
        fallback_main_floor = self.functional_main_fallback_floor
        fallback_main_delta = self.functional_main_fallback_delta

        best_regular_main_score = 0.0
        if main_candidates:
            best_regular_main_score = score_map[main_candidates[0].agent_id].final_score

        relaxed_main_candidates = self._sorted_any_main_candidates(agents, score_map, strict_only=False)
        best_relaxed_main_score = 0.0
        best_relaxed_is_functional = False
        if relaxed_main_candidates:
            best_relaxed_main_score = score_map[relaxed_main_candidates[0].agent_id].final_score
            best_relaxed_is_functional = self._is_functional_agent(relaxed_main_candidates[0])

        should_relax_main = False
        if not main_candidates and relaxed_main_candidates and best_relaxed_is_functional:
            should_relax_main = True
        elif relaxed_main_candidates and best_relaxed_is_functional:
            should_relax_main = (
                best_regular_main_score < fallback_main_floor
                and (best_relaxed_main_score - best_regular_main_score) >= fallback_main_delta
            )

        if should_relax_main and relaxed_main_candidates:
            main_candidates = relaxed_main_candidates

        main_conflict_resolution: Dict[str, Any] = {"decision_reason": "no_candidate", "close_score_conflict": False}
        if main_candidates:
            main_agent, main_conflict_resolution = self._resolve_main_close_conflict(main_candidates, score_map, bundle)

        selected_agent_ids: List[str] = []
        if main_agent:
            selected_agent_ids.append(main_agent.agent_id)

        reserved_for_check = 1 if check_triggered else 0
        reserved_for_summary = 1 if summary_triggered else 0
        remaining_slots = max(max_speakers - len(selected_agent_ids) - reserved_for_check - reserved_for_summary, 0)

        supplement_candidates = self._sorted_supplement_candidates(
            agents=agents,
            score_map=score_map,
            selected_agent_ids=selected_agent_ids,
            main_agent=main_agent,
            bundle=bundle,
        )
        supplement_resolution_steps: List[Dict[str, Any]] = []
        remaining_supplement_candidates = list(supplement_candidates)
        while remaining_slots > 0 and remaining_supplement_candidates:
            chosen_supplement, supplement_resolution = self._resolve_supplement_close_conflict(
                remaining_supplement_candidates,
                score_map=score_map,
                main_agent=main_agent,
                bundle=bundle,
            )
            if chosen_supplement is None:
                break
            supplements.append(chosen_supplement)
            supplement_resolution_steps.append(supplement_resolution)
            selected_agent_ids.append(chosen_supplement.agent_id)
            remaining_slots -= 1
            remaining_supplement_candidates = [
                agent for agent in remaining_supplement_candidates if agent.agent_id != chosen_supplement.agent_id
            ]

        if check_triggered:
            check_agent = self._pick_check_agent(agents, selected_agent_ids)
            if check_agent is not None:
                selected_agent_ids.append(check_agent.agent_id)

        if summary_triggered:
            summary_agent = self._pick_summary_agent(agents, selected_agent_ids)
            if summary_agent is not None:
                selected_agent_ids.append(summary_agent.agent_id)

        role_assignment: List[Dict[str, str]] = []
        speaker_order: List[str] = []

        if main_agent is not None:
            role_assignment.append({"agent_id": main_agent.agent_id, "role": ROLE_MAIN})
            speaker_order.append(main_agent.agent_id)

        for agent in supplements:
            role_assignment.append({"agent_id": agent.agent_id, "role": ROLE_SUPPLEMENT})
            speaker_order.append(agent.agent_id)

        if check_agent is not None:
            role_assignment.append({"agent_id": check_agent.agent_id, "role": ROLE_CHECK})
            speaker_order.append(check_agent.agent_id)

        if summary_agent is not None:
            role_assignment.append({"agent_id": summary_agent.agent_id, "role": ROLE_SUMMARY})
            speaker_order.append(summary_agent.agent_id)

        skip_mute = self._build_skip_mute(
            agents=agents,
            selected_agent_ids=speaker_order,
            score_map=score_map,
            recent_speakers=recent_speakers,
        )

        role_map = {item["agent_id"]: item["role"] for item in role_assignment}
        execution_steps: List[Dict[str, Any]] = []
        for idx, agent_id in enumerate(speaker_order, start=1):
            agent = agent_map[agent_id]
            role = role_map[agent_id]
            step_address = self._build_step_address_spec(
                role=role,
                agent_id=agent_id,
                speaker_order=speaker_order,
                agent_map=agent_map,
                relation_info=relation_info,
                summary_reason=summary_reason,
                bundle=bundle,
            )
            execution_steps.append(
                {
                    "step_id": idx,
                    "agent_id": agent_id,
                    "role": role,
                    "instruction": self._instruction_for_role(
                        role=role,
                        agent=agent,
                        payload=payload,
                        check_reason=check_reason,
                        summary_reason=summary_reason,
                        relation_info=relation_info,
                        bundle=bundle,
                    ),
                    **step_address,
                }
            )

        main_ranking = self._ranking_trace(main_candidates, score_map=score_map, role=ROLE_MAIN, main_agent=None)
        supplement_ranking = self._ranking_trace(
            supplement_candidates,
            score_map=score_map,
            role=ROLE_SUPPLEMENT,
            main_agent=main_agent,
        )

        next_topic_state = self._next_topic_state_after_maintain(bundle)
        task_type = self._classify_task_type(bundle)
        topic_seriousness = self._infer_topic_seriousness(bundle, task_type)

        return {
            "quota": {"max_speakers": max_speakers, "per_agent_max_turn": 1},
            "role_assignment": role_assignment,
            "speaker_order": speaker_order,
            "skip_mute": skip_mute,
            "check_trigger": {"triggered": check_triggered, "reason": check_reason},
            "summary_trigger": {"triggered": summary_triggered, "reason": summary_reason},
            "final_plan": {
                "selected_agents": speaker_order,
                "execution_steps": execution_steps,
                "next_topic_state": next_topic_state,
                "planner_meta": {
                    "topic_action": bundle.topic_action,
                    "forced_summary": False,
                    "summary_reason": summary_reason,
                    "check_triggered": check_triggered,
                },
            },
            "ranking": {
                "main": main_ranking,
                "supplement": supplement_ranking,
                "decision": {
                    "main_selected": main_agent.agent_id if main_agent else None,
                    "supplements_selected": [agent.agent_id for agent in supplements],
                    "tie_margin_main": self.tie_margin_main,
                    "tie_margin_supplement": self.tie_margin_supplement,
                    "task_type": task_type,
                    "topic_seriousness": topic_seriousness,
                    "explicit_close_score_thresholds": {
                        "close_score_delta_main": self.close_score_delta_main,
                        "close_score_delta_supplement": self.close_score_delta_supplement,
                        "close_query_focus_delta": self.close_query_focus_delta,
                        "close_topic_focus_delta": self.close_topic_focus_delta,
                        "close_diversity_delta": self.close_diversity_delta,
                    },
                    "main_tiebreak_rule": [
                        "if abs(final_score) >= close_score_delta_main -> higher_final_score",
                        "else -> task_seriousness_fit",
                        "else -> query_focus",
                        "else -> topic_focus",
                        "else -> less_recent",
                        "else -> agent_list_order",
                    ],
                    "supplement_tiebreak_rule": [
                        "if abs(final_score) >= close_score_delta_supplement -> higher_final_score",
                        "else -> task_seriousness_fit",
                        "else -> diversity_from_main",
                        "else -> topic_focus",
                        "else -> less_recent",
                        "else -> agent_list_order",
                    ],
                    "main_conflict_resolution": main_conflict_resolution,
                    "supplement_conflict_resolutions": supplement_resolution_steps,
                },
            },
            "diagnostics": {
                "topic_text": self._topic_text(bundle),
                "topic_action": bundle.topic_action,
                "redundancy_ratio": redundancy_ratio,
                "relation_info": relation_info,
                "coverage_state": coverage_state,
                "score_details": [asdict(item) for item in scores],
                "close_score_conflicts": {
                    "main": main_conflict_resolution,
                    "supplements": supplement_resolution_steps,
                },
            },
        }


def pretty_print(title: str, data: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_payload = {
        "query": "明星群聊里下一轮谁来主答，怎么避免重复并在 check 层核实冲突？",
        "agent_list": [
            {"agent_id": "a1", "name": "演员", "keywords": ["剧情", "情绪表达", "互动"], "description": "擅长从剧情推进和观众体验角度发言"},
            {"agent_id": "a2", "name": "歌手", "keywords": ["情绪", "节奏", "舞台"], "description": "擅长补充情绪和舞台节奏"},
            {"agent_id": "a3", "name": "主持人", "keywords": ["总结", "控场", "收束"], "description": "擅长检查关系并总结", "can_summarize": True, "can_check": True},
        ],
        "topic_state": {
            "main_topic": "多智能体群聊调度",
            "active_topic": ["主答选择", "重复抑制", "check 层冲突核实"],
            "topic_summary": "当前在讨论最小调度规则。",
            "topic_action": "maintain",
        },
        "history_summary": {
            "done_points": ["每轮先选一个主答", "补充者只说一个新点"],
        },
        "last_round_outputs": [
            {"agent_id": "a1", "text": "我建议先由最相关的 agent 主答，再让其他人补充。", "key_points": ["主答优先", "再补充"]},
            {"agent_id": "a2", "text": "我不建议所有人都补充，否则很容易重复。", "key_points": ["避免所有人都补充", "减少重复"]},
        ],
    }

    orchestrator = MinimalOrchestrator()
    pretty_print("Demo Plan", orchestrator.plan(demo_payload))
