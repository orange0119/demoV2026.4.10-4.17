from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sample_cases import get_cases, get_shared_agents

MinimalOrchestrator = None
LightweightHashEmbedder = None
cosine_similarity = None
tokenize_mixed_text = None
ROLE_MAIN = "main"
ROLE_SUPPLEMENT = "supplement"
ROLE_CHECK = "check"
ROLE_SUMMARY = "summary"


def _import_orchestrator_module(orchestrator_module: str = "demo_orchestrator"):
    module_ref = (orchestrator_module or "demo_orchestrator").strip()
    if not module_ref:
        module_ref = "demo_orchestrator"

    if module_ref.endswith(".py") or any(sep in module_ref for sep in ("/", "\\")):
        module_path = Path(module_ref)
        if not module_path.exists():
            raise FileNotFoundError(f"找不到 orchestrator 文件：{module_path}")
        module_name = f"{module_path.stem}_qwen_runtime"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从文件加载 orchestrator：{module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    module_name = module_ref[:-3] if module_ref.endswith(".py") else module_ref
    return importlib.import_module(module_name)


def configure_orchestrator_runtime(orchestrator_module: str = "demo_orchestrator") -> None:
    global MinimalOrchestrator, LightweightHashEmbedder, cosine_similarity, tokenize_mixed_text
    global ROLE_MAIN, ROLE_SUPPLEMENT, ROLE_CHECK, ROLE_SUMMARY

    mod = _import_orchestrator_module(orchestrator_module)
    MinimalOrchestrator = getattr(mod, "MinimalOrchestrator")
    LightweightHashEmbedder = getattr(mod, "LightweightHashEmbedder")
    cosine_similarity = getattr(mod, "cosine_similarity")
    tokenize_mixed_text = getattr(mod, "tokenize_mixed_text")
    ROLE_MAIN = getattr(mod, "ROLE_MAIN", "main")
    ROLE_SUPPLEMENT = getattr(mod, "ROLE_SUPPLEMENT", "supplement")
    ROLE_CHECK = getattr(mod, "ROLE_CHECK", "check")
    ROLE_SUMMARY = getattr(mod, "ROLE_SUMMARY", "summary")


CLOSING_CUES = [
    "就这么定",
    "那就这么定",
    "先按这个走",
    "先这么定",
    "可以直接拍",
    "开始准备",
    "今天先到这",
    "结束这个话题",
    "这个方案先定了",
    "当前结论先按这个执行",
    "先收在这里",
]

QUIT_COMMANDS = {"quit", "exit", "/q", "/quit", "/exit"}
RESET_COMMANDS = {"/reset", "reset", "/clear"}
HELP_COMMANDS = {"/help", "help"}
RETURN_COMMANDS = {"/return", "return"}
MAINTAIN_COMMANDS = {"/maintain", "maintain"}
STATE_COMMANDS = {"/state", "state"}

CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def estimate_token_count(text: str) -> int:
    """
    粗略 token 估计：
    - 中文按字近似
    - 非中文部分按每 4 个非空字符约 1 token
    """
    text = text or ""
    if not text:
        return 0
    cjk_count = len(CJK_CHAR_RE.findall(text))
    non_cjk = CJK_CHAR_RE.sub("", text)
    compact = re.sub(r"\s+", "", non_cjk)
    other_count = math.ceil(len(compact) / 4) if compact else 0
    return cjk_count + other_count


def build_planner_input_snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    topic_state = payload.get("topic_state", {}) or {}
    history_summary = sanitize_history_summary(payload.get("history_summary", {}) or {})
    last_round_outputs = payload.get("last_round_outputs", []) or []

    last_round_lines: List[str] = []
    for item in last_round_outputs:
        speaker = item.get("agent_name") or item.get("name") or item.get("agent_id") or "某人"
        role = item.get("role", "") or ""
        text = (item.get("text", "") or "").strip()
        if not text:
            continue
        role_suffix = f"（{role}）" if role else ""
        last_round_lines.append(f"{speaker}{role_suffix}：{text}")

    pieces = {
        "query": payload.get("query", "") or "",
        "context_query": payload.get("context_query", "") or "",
        "user_text": payload.get("user_text", "") or "",
        "topic_state": json.dumps(topic_state, ensure_ascii=False),
        "history_summary": json.dumps(history_summary, ensure_ascii=False),
        "last_round_outputs": "\n".join(last_round_lines),
    }
    token_breakdown = {key: estimate_token_count(value) for key, value in pieces.items() if value}
    joined_text = "\n".join([value for value in pieces.values() if value])
    return {
        "text": joined_text,
        "estimated_total_tokens": estimate_token_count(joined_text),
        "estimated_breakdown": token_breakdown,
    }


def _normalize_last_round_outputs(last_round_outputs: Sequence[Dict[str, Any]], agent_name_map: Dict[str, str]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in last_round_outputs or []:
        agent_id = item.get("agent_id") or ""
        normalized.append({
            "agent_id": agent_id,
            "agent_name": item.get("agent_name") or item.get("name") or agent_name_map.get(agent_id, agent_id),
            "role": item.get("role", "") or "",
            "text": (item.get("text", "") or "").strip(),
            "key_points": list(item.get("key_points", []) or []),
            "stance": item.get("stance"),
        })
    return normalized


def build_downstream_input(
    payload: Dict[str, Any],
    planner_result: Dict[str, Any],
    planner_snapshot: Dict[str, Any],
    planner_elapsed_ms: float,
) -> Dict[str, Any]:
    agent_list = payload.get("agent_list", []) or []
    agent_map_by_id = {item.get("agent_id"): item for item in agent_list if item.get("agent_id")}
    agent_name_map = {item.get("agent_id"): item.get("name", item.get("agent_id", "")) for item in agent_list if item.get("agent_id")}

    final_plan = planner_result.get("final_plan", {}) or {}
    execution_steps = final_plan.get("execution_steps", []) or []
    selected_agents = final_plan.get("selected_agents", []) or []

    selected_agent_profiles: List[Dict[str, Any]] = []
    for agent_id in selected_agents:
        raw = agent_map_by_id.get(agent_id, {})
        selected_agent_profiles.append({
            "agent_id": agent_id,
            "agent_name": raw.get("name", agent_id),
            "status": raw.get("status", "online"),
            "keywords": list(raw.get("keywords", []) or []),
            "description": raw.get("description", "") or "",
            "persona": raw.get("persona", "") or "",
            "style": raw.get("style", "") or "",
            "can_summarize": bool(raw.get("can_summarize", False)),
            "can_check": bool(raw.get("can_check", False)),
        })

    normalized_steps: List[Dict[str, Any]] = []
    for step in execution_steps:
        agent_id = step.get("agent_id", "")
        normalized_steps.append({
            "step_id": step.get("step_id"),
            "agent_id": agent_id,
            "agent_name": agent_name_map.get(agent_id, agent_id),
            "role": step.get("role", ""),
            "instruction": step.get("instruction", "") or "",
            "address_to": list(step.get("address_to", []) or []),
            "address_to_names": list(step.get("address_to_names", []) or []),
            "address_note": step.get("address_note", "") or "",
        })

    return {
        "schema_version": "downstream_input.v1",
        "task": "multi_agent_groupchat_generation",
        "input_meta": {
            "source": "minimal_orchestrator",
            "query": payload.get("query", "") or "",
            "context_query": payload.get("context_query", "") or "",
            "user_text": payload.get("user_text", payload.get("query", "") or ""),
        },
        "topic_context": {
            "topic_state": dict(payload.get("topic_state", {}) or {}),
            "history_summary": sanitize_history_summary(payload.get("history_summary", {}) or {}),
            "last_round_outputs": _normalize_last_round_outputs(payload.get("last_round_outputs", []) or [], agent_name_map),
        },
        "planner_runtime": {
            "planner_elapsed_ms": planner_elapsed_ms,
            "planner_input_estimated_tokens": planner_snapshot.get("estimated_total_tokens", 0),
            "planner_input_token_breakdown": planner_snapshot.get("estimated_breakdown", {}),
        },
        "planner_decision": {
            "quota": planner_result.get("quota", {}) or {},
            "role_assignment": list(planner_result.get("role_assignment", []) or []),
            "speaker_order": list(planner_result.get("speaker_order", []) or []),
            "skip_mute": list(planner_result.get("skip_mute", []) or []),
            "check_trigger": dict(planner_result.get("check_trigger", {}) or {}),
            "summary_trigger": dict(planner_result.get("summary_trigger", {}) or {}),
            "ranking": dict(planner_result.get("ranking", {}) or {}),
            "diagnostics": dict(planner_result.get("diagnostics", {}) or {}),
        },
        "generation_plan": {
            "selected_agents": list(selected_agents),
            "selected_agent_profiles": selected_agent_profiles,
            "execution_steps": normalized_steps,
            "next_topic_state": dict(final_plan.get("next_topic_state", {}) or {}),
            "planner_meta": dict(final_plan.get("planner_meta", {}) or {}),
        },
    }


class SemanticHelper:
    """供调用层复用的最小 embedding + cosine 语义工具。"""

    def __init__(self, embedder: Optional[LightweightHashEmbedder] = None) -> None:
        self.embedder = embedder or LightweightHashEmbedder(dim=384)

    def encode(self, text: str):
        return self.embedder.encode(text or "")

    def similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return cosine_similarity(self.encode(a), self.encode(b))

    def max_similarity(self, text: str, candidates: Sequence[str]) -> float:
        if not text or not candidates:
            return 0.0
        query_vec = self.encode(text)
        best = 0.0
        for item in candidates:
            if not item:
                continue
            best = max(best, cosine_similarity(query_vec, self.encode(item)))
        return best


class QwenClient:
    def __init__(
        self,
        model: str = "qwen-plus",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量。请先设置 API Key，或在命令行中通过 --api-key 传入。")

        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "缺少 openai 依赖。请先执行 `pip install openai`，再运行 qwen_runner.py。"
            ) from exc

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            return "\n".join([x.strip() for x in parts if str(x).strip()]).strip()
        return str(content).strip()

    @staticmethod
    def _extract_usage(resp: Any) -> Dict[str, Optional[int]]:
        usage = getattr(resp, "usage", None)
        if usage is None and isinstance(resp, dict):
            usage = resp.get("usage")

        def _pick(obj: Any, key: str) -> Optional[int]:
            if obj is None:
                return None
            if isinstance(obj, dict):
                value = obj.get(key)
            else:
                value = getattr(obj, key, None)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return {
            "prompt_tokens": _pick(usage, "prompt_tokens"),
            "completion_tokens": _pick(usage, "completion_tokens"),
            "total_tokens": _pick(usage, "total_tokens"),
        }

    def chat_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": self._extract_text_content(resp.choices[0].message.content),
            "usage": self._extract_usage(resp),
            "model": self.model,
        }

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 400,
    ) -> str:
        return self.chat_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )["content"]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", "", text or "")
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    return text.lower()


def unique_keep_order_semantic(
    items: Sequence[str],
    semantic: SemanticHelper,
    threshold: float = 0.82,
) -> List[str]:
    result: List[str] = []
    for item in items:
        item = (item or "").strip()
        if not item:
            continue
        if semantic.max_similarity(item, result) >= threshold:
            continue
        result.append(item)
    return result


def extract_key_points(text: str) -> List[str]:
    sentences = re.split(r"[。！？!?；;\n]", text or "")
    points = [s.strip() for s in sentences if s.strip()]
    return points[:2]


def infer_stance(text: str) -> Optional[str]:
    if "温情" in text or "感动" in text or "温暖" in text:
        return "warm_style"
    if "搞笑" in text or "好笑" in text or "包袱" in text or "幽默" in text:
        return "funny_style"
    if "折中" in text or "结合" in text or "都可以" in text:
        return "hybrid_style"
    if "反转" in text:
        return "funny_twist"
    return None


def format_history_summary(history_summary: Dict[str, Any]) -> str:
    done_points = history_summary.get("done_points", []) or []
    done_text = "；".join(done_points[:8]) if done_points else "暂无"
    return f"已形成的关键点：{done_text}"


def sanitize_history_summary(history_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(history_summary, dict):
        return {"done_points": [], "resolved": False}
    return {
        "done_points": list(history_summary.get("done_points", []) or []),
        "resolved": bool(history_summary.get("resolved", False)),
    }


def build_role_requirement(role: str) -> str:
    if role == ROLE_MAIN:
        return (
            "- 你是主答：优先正面回应当前主题，推进主线\n"
            "- 给出 1~2 个最关键的判断或方案，不要发散\n"
            "- 不要做总结收尾，也不要复述上一轮原话"
        )
    if role == ROLE_SUPPLEMENT:
        return (
            "- 你是补充：只补 1 个真正新增的点\n"
            "- 优先补细节、风险、例子或可执行建议\n"
            "- 若你的想法和前文相似，就换一个更具体的新角度"
        )
    if role == ROLE_CHECK:
        return (
            "- 你是检查：先判断前文关系更像 supplement / contrast / correction / topic_shift 中哪一类\n"
            "- 再给一句 explanation，最后再决定主持动作更适合 probing / confronting / interpretation / instruction\n"
            "- 不要抢主答，不要展开成长篇总结"
        )
    if role == ROLE_SUMMARY:
        return (
            "- 你是总结：整合已有信息，压缩重复，并给出阶段性结论\n"
            "- 当 topic_action=return 时，你要先把大家拉回上一个稳定话题，再给一个继续往下聊的切口\n"
            "- 不要机械结束对话，除非已经明确收束"
        )
    return "- 按调度器角色要求发言"


def format_address_to_names(address_names: Sequence[str]) -> str:
    names = [name for name in address_names if name]
    return "、".join(names) if names else "全体"


def build_round_context_query(payload: Dict[str, Any], agent_name_map: Optional[Dict[str, str]] = None) -> str:
    explicit_context = (payload.get("context_query", "") or "").strip()
    if explicit_context:
        return explicit_context

    last_round_outputs = payload.get("last_round_outputs", []) or []
    context_lines: List[str] = []
    for item in last_round_outputs:
        agent_id = item.get("agent_id", "")
        name = agent_name_map.get(agent_id, "") if agent_name_map and agent_id else ""
        name = name or item.get("agent_name") or item.get("name") or agent_id or "某人"
        role = item.get("role", "") or ""
        text = (item.get("text", "") or "").strip()
        if not text:
            continue
        role_suffix = f"（{role}）" if role else ""
        context_lines.append(f"{name}{role_suffix}：{text}")

    if context_lines:
        return "\n".join(context_lines)
    return (payload.get("query", "") or "").strip()


def build_system_prompt(agent: Dict[str, Any], role: str) -> str:
    role_requirement = build_role_requirement(role)

    if role == ROLE_MAIN:
        role_extra = "- 第一句必须直接回应用户当前的问题、情绪或困惑"
    elif role == ROLE_SUPPLEMENT:
        role_extra = "- 先承接主答或上一位发言，再补 1 个新增点，不要把自己写成新的主答"
    elif role == ROLE_CHECK:
        role_extra = "- 先判断前文关系，再给主持动作；你不是来重新回答用户问题的"
    elif role == ROLE_SUMMARY:
        role_extra = "- 先做群聊收束，再视需要补一句面向用户的话，不要重新展开主线"
    else:
        role_extra = "- 发言要符合当前角色分工"

    return f"""
你在一个明星群聊系统中扮演：{agent['name']}

你的身份设定：
- 身份：{agent['name']}
- 描述：{agent.get('description', '')}
- persona：{agent.get('persona', '')}
- style：{agent.get('style', '')}

你的任务要求：
- 当前角色：{role}
{role_requirement}
{role_extra}
- 必须保持该明星身份视角说话，但不要只营造氛围
- 如果用户在聊情感、困惑、关系、心情，不要把回答大段写成舞台、歌词、灯光、采样这类隐喻
- 优先输出：判断 / 共情 / 一个小建议；再保留一点明星气质
- 语气自然，像群聊发言，不要像论文或报告
- 输出控制在 2~3 句
- 尽量具体，不空泛
- 不要使用项目符号或编号
""".strip()


def build_user_prompt(
    query: str,
    instruction: str,
    history_summary: Dict[str, Any],
    last_round_outputs: List[Dict[str, Any]],
    agent_name_map: Dict[str, str],
    planner_result: Optional[Dict[str, Any]] = None,
    address_to_names: Optional[Sequence[str]] = None,
    user_text: Optional[str] = None,
    role: Optional[str] = None,
    address_note: Optional[str] = None,
    opening_hint: Optional[str] = None,
) -> str:
    recent_history_lines: List[str] = []
    for x in last_round_outputs[-2:]:
        agent_id = x.get("agent_id", "")
        agent_name = agent_name_map.get(agent_id, agent_id or "某人")
        role = x.get("role")
        text = (x.get("text", "") or "").strip()
        if not text:
            continue
        role_suffix = f"（{role}）" if role else ""
        recent_history_lines.append(f"{agent_name}{role_suffix}：{text}")

    history_text = "\n".join(recent_history_lines) if recent_history_lines else "暂无上一轮发言"
    addressee_text = format_address_to_names(address_to_names or [])
    history_summary_text = format_history_summary(history_summary)

    current_user_text = user_text or query
    opening_text = opening_hint or "先自然接住当前对象，再进入你的角色任务。"
    address_note_text = address_note or ""

    if role == ROLE_MAIN:
        closing_hint = "记住：先回应用户，再保持明星身份感。"
    elif role == ROLE_SUPPLEMENT:
        closing_hint = "记住：先接主答或上一位发言，再补 1 个真正新增的点。"
    elif role == ROLE_CHECK:
        closing_hint = "记住：你在做关系核实与主持动作判断，不要把自己写成新的主答。"
    elif role == ROLE_SUMMARY:
        closing_hint = "记住：先收束群聊，再视情况对用户落一句，不要重新发散。"
    else:
        closing_hint = "记住：按当前角色分工自然发言。"

    return f"""
用户刚刚说的话：
{current_user_text}

你这轮的任务：
{instruction}

你这轮主要在对谁说：
{addressee_text}

关于回复对象的补充说明：
{address_note_text or '无'}

建议开头方式：
{opening_text}

可参考的最近上下文：
{history_text}

已有简短摘要：
{history_summary_text}

请直接给出这轮群聊发言。
{closing_hint}
""".strip()


def summarize_round_outputs(
    previous_summary: Dict[str, Any],
    round_outputs: List[Dict[str, Any]],
    semantic: SemanticHelper,
) -> Dict[str, Any]:
    done_points = list(previous_summary.get("done_points", []) or [])
    new_points: List[str] = []

    for item in round_outputs:
        role = item.get("role")
        if role == ROLE_CHECK:
            continue
        key_points = item.get("key_points", []) or []
        if key_points:
            new_points.extend(key_points)
        else:
            text = (item.get("text", "") or "").strip()
            if text:
                new_points.append(text[:60])

    done_points = unique_keep_order_semantic(done_points + new_points, semantic=semantic, threshold=0.82)
    return {
        "done_points": done_points[:12],
        "resolved": bool(previous_summary.get("resolved", False)),
    }


def detect_summarizer_stop(round_outputs: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    for item in round_outputs:
        if item.get("role") != ROLE_SUMMARY:
            continue
        content = item.get("text", "") or ""
        for cue in CLOSING_CUES:
            if cue in content:
                return True, cue
    return False, None


def detect_repeat_stop(
    current_round_outputs: List[Dict[str, Any]],
    previous_round_outputs: List[Dict[str, Any]],
    semantic: SemanticHelper,
    threshold: float = 0.86,
) -> bool:
    if not current_round_outputs or not previous_round_outputs:
        return False

    current_text = " ".join([x.get("text", "") for x in current_round_outputs if x.get("text")])
    prev_text = " ".join([x.get("text", "") for x in previous_round_outputs if x.get("text")])
    if not current_text or not prev_text:
        return False

    role_signature_now = [x.get("role") for x in current_round_outputs]
    role_signature_prev = [x.get("role") for x in previous_round_outputs]
    sim = semantic.similarity(current_text, prev_text)
    return sim >= threshold and role_signature_now == role_signature_prev


def apply_next_topic_state(payload: Dict[str, Any], planner_result: Dict[str, Any]) -> None:
    next_topic_state = (planner_result.get("final_plan", {}) or {}).get("next_topic_state")
    if not next_topic_state:
        return
    payload.setdefault("topic_state", {})
    payload["topic_state"].update(next_topic_state)
    payload["topic_state"]["topic_action"] = "maintain"


import re
from typing import List, Dict, Any, Optional, Sequence

EMOTION_CUES = [
    "喜欢", "心动", "想念", "暧昧", "纠结", "难过", "分手", "失恋", "告白", "暗恋",
    "不知道怎么办", "不确定", "是不是", "要不要在一起", "想他", "想她",
]

WORK_STAGE_CUES = [
    "舞台", "首唱", "演唱会", "彩排", "开场", "灯光", "编排", "节奏", "唱跳",
]

PROMO_CUES = [
    "宣发", "宣传", "采访", "红毯", "综艺", "营业", "路透", "话题度",
]

RETURN_CUES = [
    "跑题", "偏题", "拉回来", "回到刚才", "先别聊那个",
]

CASUAL_CUES = [
    "哈哈", "笑死", "好玩", "无语", "离谱", "尴尬", "今天", "刚刚",
]

CHINESE_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")

TOPIC_STOPWORDS = {
    "我觉得", "感觉", "就是", "然后", "这个", "那个", "真的", "好像", "有点",
    "不知道", "怎么说", "是不是", "一个人", "东西", "事情",
}


def classify_scene_mode(user_text: str, last_round_outputs: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    text = (user_text or "").strip().lower()

    if any(cue in text for cue in RETURN_CUES):
        return "topic_return"
    if any(cue in text for cue in EMOTION_CUES):
        return "emotion_support"
    if any(cue in text for cue in WORK_STAGE_CUES):
        return "celeb_work_topic"
    if any(cue in text for cue in PROMO_CUES):
        return "promo_discussion"
    if any(cue in text for cue in CASUAL_CUES):
        return "casual_groupchat"

    # 如果上一轮已经是强工作场景，可以继承一点上下文
    joined_last_round = " ".join((x.get("text", "") or "") for x in (last_round_outputs or []))
    if any(cue in joined_last_round for cue in WORK_STAGE_CUES + PROMO_CUES):
        return "celeb_work_topic"

    return "general_chat"


def _dedup_keep_order(items: Sequence[str], max_items: int = 6) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        item = (item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= max_items:
            break
    return result


def _phrase_candidates(text: str) -> List[str]:
    candidates = CHINESE_PHRASE_RE.findall(text or "")
    cleaned: List[str] = []
    for c in candidates:
        if c in TOPIC_STOPWORDS:
            continue
        if len(c) < 2:
            continue
        cleaned.append(c)
    return _dedup_keep_order(cleaned, max_items=8)


def extract_active_topics_from_user_text(
    user_text: str,
    max_topics: int = 6,
    scene_mode: Optional[str] = None,
) -> List[str]:
    text = (user_text or "").strip()
    if not text:
        return ["当前话题"]

    mode = scene_mode or classify_scene_mode(text)

    if mode == "emotion_support":
        topics = ["情感困惑", "是否心动", "确认自己的感受"]
        phrases = _phrase_candidates(text)
        if "喜欢" in text:
            topics.insert(0, "喜欢一个人")
        return _dedup_keep_order(topics + phrases, max_items=max_topics)

    if mode == "celeb_work_topic":
        topics = ["舞台设计", "表达方式", "现场效果"]
        phrases = _phrase_candidates(text)
        return _dedup_keep_order(phrases + topics, max_items=max_topics)

    if mode == "promo_discussion":
        topics = ["宣传风格", "对外表达", "互动效果"]
        phrases = _phrase_candidates(text)
        return _dedup_keep_order(phrases + topics, max_items=max_topics)

    if mode == "topic_return":
        topics = ["回到主线", "纠偏", "当前分支收束"]
        phrases = _phrase_candidates(text)
        return _dedup_keep_order(topics + phrases, max_items=max_topics)

    phrases = _phrase_candidates(text)
    if phrases:
        return _dedup_keep_order(phrases, max_items=max_topics)

    return [text[: min(len(text), 12)]]


def trim_text(text: str, max_len: int = 28) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


CONTINUATION_CUES = ["继续", "再说", "展开", "补充", "这一轮", "接着", "继续推进"]
END_SESSION_CUES = ["先这样", "晚安", "下次聊", "结束吧", "今天先到这", "拜拜", "再见"]


def detect_topic_transition(user_text: str, previous_topic_state: Dict[str, Any]) -> str:
    text = (user_text or "").strip()
    lowered = text.lower()
    if any(cue in text for cue in END_SESSION_CUES):
        return "end"
    if any(cue in lowered for cue in RETURN_COMMANDS) or any(cue in text for cue in RETURN_CUES):
        return "return"
    if any(cue in text for cue in CONTINUATION_CUES):
        return "maintain"

    previous_text = " ".join([
        previous_topic_state.get("main_topic", "") or "",
        " ".join(previous_topic_state.get("active_topic", []) or []),
        previous_topic_state.get("topic_summary", "") or "",
    ])
    if not previous_text:
        return "maintain"

    current_tokens = set(tokenize_mixed_text(text))
    previous_tokens = set(tokenize_mixed_text(previous_text))
    if not current_tokens or not previous_tokens:
        return "maintain"

    overlap = len(current_tokens & previous_tokens) / max(len(current_tokens | previous_tokens), 1)
    return "shift" if overlap < 0.12 else "maintain"


def build_interactive_context_query(user_text: str, last_round_outputs: List[Dict[str, Any]], agent_name_map: Dict[str, str]) -> str:
    lines = [f"用户：{user_text}"]
    for item in last_round_outputs[-3:]:
        agent_id = item.get("agent_id", "")
        agent_name = agent_name_map.get(agent_id, agent_id or "某人")
        role = item.get("role")
        text = (item.get("text", "") or "").strip()
        if not text:
            continue
        role_suffix = f"（{role}）" if role else ""
        lines.append(f"{agent_name}{role_suffix}：{text}")
    return "\n".join(lines)


def build_interactive_payload(
    user_text: str,
    shared_state: Dict[str, Any],
    turn_index: int,
) -> Dict[str, Any]:
    agent_list = shared_state["agent_list"]
    history_summary = sanitize_history_summary(shared_state.get("history_summary", {"done_points": []}))
    last_round_outputs = list(shared_state.get("last_round_outputs", []) or [])
    previous_topic_state = dict(shared_state.get("topic_state", {}) or {})
    agent_name_map = {a["agent_id"]: a["name"] for a in agent_list}

    scene_mode = classify_scene_mode(user_text, last_round_outputs)
    active_topics = extract_active_topics_from_user_text(user_text, scene_mode=scene_mode)

    previous_main_topic = previous_topic_state.get("main_topic", "") or trim_text(user_text)
    previous_active_topic = previous_topic_state.get("active_topic", []) or active_topics
    previous_topic_summary = previous_topic_state.get("topic_summary", "") or f"上一轮围绕『{previous_main_topic}』继续展开。"

    pending_topic_action = (shared_state.get("pending_topic_action", "maintain") or "maintain").strip().lower()
    if pending_topic_action not in {"maintain", "return"}:
        pending_topic_action = "maintain"

    topic_transition = detect_topic_transition(user_text, previous_topic_state)
    if scene_mode == "topic_return":
        topic_transition = "return"
    if topic_transition == "end":
        pending_topic_action = "end"
    elif topic_transition == "return":
        pending_topic_action = "return"

    topic_state = {
        "main_topic": trim_text(user_text),
        "active_topic": active_topics,
        "topic_summary": f"用户当前想聊：{trim_text(user_text, max_len=40)}",
        "topic_action": "return" if scene_mode == "topic_return" else pending_topic_action,
        "topic_transition": topic_transition,
        "allow_pre_summary": False,
        "scene_mode": scene_mode,
        "round_id": turn_index,
        "previous_main_topic": previous_main_topic,
        "previous_active_topic": previous_active_topic,
        "previous_topic_summary": previous_topic_summary,
    }

    payload = {
        "query": user_text,
        "context_query": build_interactive_context_query(user_text, last_round_outputs, agent_name_map),
        "user_text": user_text,
        "agent_list": agent_list,
        "topic_state": topic_state,
        "history_summary": history_summary,
        "last_round_outputs": last_round_outputs,
    }
    return payload


def generate_one_round(
    llm: QwenClient,
    payload: Dict[str, Any],
    semantic: Optional[SemanticHelper] = None,
    temperature: float = 0.8,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    if MinimalOrchestrator is None:
        raise RuntimeError("MinimalOrchestrator 尚未配置，请先调用 configure_orchestrator_runtime()。")
    semantic = semantic or SemanticHelper()
    orchestrator = MinimalOrchestrator(embedder=semantic.embedder)

    round_start = time.perf_counter()
    planner_snapshot = build_planner_input_snapshot(payload)
    planner_start = time.perf_counter()
    planner_result = orchestrator.plan(payload)
    planner_elapsed_ms = round((time.perf_counter() - planner_start) * 1000, 2)

    agent_map_by_id = {a["agent_id"]: a for a in payload["agent_list"]}
    agent_name_map = {a["agent_id"]: a["name"] for a in payload["agent_list"]}
    round_context_query = build_round_context_query(payload, agent_name_map)

    rolling_history = [dict(item) for item in payload.get("last_round_outputs", [])]
    generated_outputs: List[Dict[str, Any]] = []
    step_timings: List[Dict[str, Any]] = []
    step_token_usage: List[Dict[str, Any]] = []

    execution_steps = planner_result.get("final_plan", {}).get("execution_steps", [])
    for step in execution_steps:
        agent_id = step["agent_id"]
        role = step["role"]
        instruction = step["instruction"]
        address_to = step.get("address_to", []) or []
        address_to_names = step.get("address_to_names", []) or []
        agent_info = agent_map_by_id[agent_id]

        system_prompt = build_system_prompt(agent_info, role)
        user_prompt = build_user_prompt(
            query=round_context_query,
            instruction=instruction,
            history_summary=sanitize_history_summary(payload.get("history_summary", {"done_points": []})),
            last_round_outputs=rolling_history,
            agent_name_map=agent_name_map,
            planner_result=planner_result,
            address_to_names=address_to_names,
            user_text=payload.get("user_text"),
            role=role,
            address_note=step.get("address_note", ""),
            opening_hint=step.get("opening_hint", ""),
        )

        estimated_system_tokens = estimate_token_count(system_prompt)
        estimated_user_tokens = estimate_token_count(user_prompt)

        llm_start = time.perf_counter()
        llm_resp = llm.chat_with_metadata(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_elapsed_ms = round((time.perf_counter() - llm_start) * 1000, 2)

        content = llm_resp["content"]
        usage = llm_resp.get("usage", {}) or {}
        token_source = "api_usage" if any(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")) else "estimated"
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        if prompt_tokens is None:
            prompt_tokens = estimated_system_tokens + estimated_user_tokens
        if completion_tokens is None:
            completion_tokens = estimate_token_count(content)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        one_output = {
            "agent_id": agent_id,
            "agent_name": agent_info["name"],
            "role": role,
            "text": content,
            "key_points": extract_key_points(content),
            "stance": infer_stance(content),
            "address_to": address_to,
            "address_to_names": address_to_names,
            "address_note": step.get("address_note", ""),
        }
        generated_outputs.append(one_output)
        rolling_history.append({
            "agent_id": agent_id,
            "role": role,
            "text": content,
            "key_points": one_output["key_points"],
        })
        step_timings.append({
            "step_id": step.get("step_id"),
            "agent_id": agent_id,
            "agent_name": agent_info["name"],
            "role": role,
            "address_to": address_to,
            "address_to_names": address_to_names,
            "llm_elapsed_ms": llm_elapsed_ms,
        })
        step_token_usage.append({
            "step_id": step.get("step_id"),
            "agent_id": agent_id,
            "agent_name": agent_info["name"],
            "role": role,
            "token_source": token_source,
            "estimated_system_tokens": estimated_system_tokens,
            "estimated_user_tokens": estimated_user_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "address_to": address_to,
            "address_to_names": address_to_names,
        })

    total_elapsed_ms = round((time.perf_counter() - round_start) * 1000, 2)
    prompt_tokens_total = sum(int(item.get("prompt_tokens") or 0) for item in step_token_usage)
    completion_tokens_total = sum(int(item.get("completion_tokens") or 0) for item in step_token_usage)
    total_tokens_total = sum(int(item.get("total_tokens") or 0) for item in step_token_usage)
    final_plan = (planner_result.get("final_plan", {}) or {})
    downstream_input = build_downstream_input(
        payload=payload,
        planner_result=planner_result,
        planner_snapshot=planner_snapshot,
        planner_elapsed_ms=planner_elapsed_ms,
    )

    return {
        "planner_result": planner_result,
        "final_plan": final_plan,
        "downstream_input": downstream_input,
        "generated_outputs": generated_outputs,
        "module_timings": {
            "planner_elapsed_ms": planner_elapsed_ms,
            "generation_steps": step_timings,
            "total_round_elapsed_ms": total_elapsed_ms,
        },
        "token_usage": {
            "planner_input_estimated_tokens": planner_snapshot.get("estimated_total_tokens", 0),
            "planner_input_token_breakdown": planner_snapshot.get("estimated_breakdown", {}),
            "generation_prompt_tokens": prompt_tokens_total,
            "generation_completion_tokens": completion_tokens_total,
            "generation_total_tokens": total_tokens_total,
            "steps": step_token_usage,
        },
    }


def run_single_round_case(
    llm: QwenClient,
    case: Dict[str, Any],
    semantic: Optional[SemanticHelper] = None,
    temperature: float = 0.8,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    result = generate_one_round(llm, case, semantic=semantic, temperature=temperature, max_tokens=max_tokens)
    result["query"] = build_round_context_query(case, {a["agent_id"]: a["name"] for a in case["agent_list"]})
    return result


def run_multi_round_case(
    llm: QwenClient,
    case: Dict[str, Any],
    max_rounds: int,
    semantic: Optional[SemanticHelper] = None,
    temperature: float = 0.8,
    max_tokens: int = 400,
) -> Dict[str, Any]:
    semantic = semantic or SemanticHelper()
    current_payload = json.loads(json.dumps(case))
    current_payload["history_summary"] = sanitize_history_summary(current_payload.get("history_summary", {"done_points": []}))
    current_payload.setdefault("last_round_outputs", [])
    current_payload.setdefault("topic_state", {})
    current_round_id = current_payload.get("topic_state", {}).get("round_id", 1)

    all_rounds: List[Dict[str, Any]] = []
    previous_generated_outputs: List[Dict[str, Any]] = []
    stop_signal = {"stop": False, "reason": None, "detail": None}

    for local_round_idx in range(1, max_rounds + 1):
        current_payload["topic_state"]["round_id"] = current_round_id
        one_round_result = generate_one_round(
            llm=llm,
            payload=current_payload,
            semantic=semantic,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        generated_outputs = one_round_result["generated_outputs"]
        planner_result = one_round_result["planner_result"]
        updated_summary = summarize_round_outputs(
            sanitize_history_summary(current_payload.get("history_summary", {"done_points": []})),
            generated_outputs,
            semantic=semantic,
        )

        round_record = {
            "local_round": local_round_idx,
            "round_id": current_round_id,
            "planner_result": planner_result,
            "final_plan": one_round_result.get("final_plan", planner_result.get("final_plan", {})),
            "downstream_input": one_round_result.get("downstream_input", planner_result.get("final_plan", {})),
            "generated_outputs": generated_outputs,
            "module_timings": one_round_result.get("module_timings", {}),
            "token_usage": one_round_result.get("token_usage", {}),
            "history_summary_after": updated_summary,
        }
        all_rounds.append(round_record)

        stop, cue = detect_summarizer_stop(generated_outputs)
        if stop:
            stop_signal = {"stop": True, "reason": "summarizer_closed", "detail": {"cue": cue}}
            break

        if detect_repeat_stop(generated_outputs, previous_generated_outputs, semantic=semantic):
            stop_signal = {"stop": True, "reason": "repeated_round", "detail": {"message": "相邻两轮内容语义上过于相似"}}
            break

        current_payload["last_round_outputs"] = [
            {
                "agent_id": item["agent_id"],
                "role": item["role"],
                "text": item["text"],
                "key_points": item["key_points"],
            }
            for item in generated_outputs
        ]
        current_payload["history_summary"] = updated_summary
        apply_next_topic_state(current_payload, planner_result)

        previous_generated_outputs = generated_outputs
        current_round_id += 1

    return {
        "query": build_round_context_query(current_payload, {a["agent_id"]: a["name"] for a in current_payload["agent_list"]}),
        "executed_rounds": len(all_rounds),
        "stop_signal": stop_signal,
        "rounds": all_rounds,
    }


def print_token_usage(token_usage: Dict[str, Any]) -> None:
    if not token_usage:
        return
    planner_tokens = token_usage.get("planner_input_estimated_tokens")
    planner_breakdown = token_usage.get("planner_input_token_breakdown", {}) or {}
    if planner_tokens is not None:
        print(f"planner_input_estimated_tokens: {planner_tokens}")
    if planner_breakdown:
        print("planner_input_token_breakdown:", json.dumps(planner_breakdown, ensure_ascii=False))

    print(
        "generation_tokens:",
        json.dumps(
            {
                "prompt_tokens": token_usage.get("generation_prompt_tokens", 0),
                "completion_tokens": token_usage.get("generation_completion_tokens", 0),
                "total_tokens": token_usage.get("generation_total_tokens", 0),
            },
            ensure_ascii=False,
        ),
    )
    for step in token_usage.get("steps", []) or []:
        address = format_address_to_names(step.get("address_to_names", []) or [])
        print(
            f"  - step {step.get('step_id')} | {step.get('agent_name')} ({step.get('role')}) | "
            f"to={address} | token_source={step.get('token_source')} | "
            f"prompt={step.get('prompt_tokens')} | completion={step.get('completion_tokens')} | total={step.get('total_tokens')}"
        )


def print_timings(module_timings: Dict[str, Any]) -> None:
    if not module_timings:
        return
    planner_ms = module_timings.get("planner_elapsed_ms")
    total_ms = module_timings.get("total_round_elapsed_ms")
    if planner_ms is not None:
        print(f"planner_elapsed_ms: {planner_ms}")
    for step in module_timings.get("generation_steps", []) or []:
        address = format_address_to_names(step.get("address_to_names", []) or [])
        print(
            f"  - step {step.get('step_id')} | {step.get('agent_name')} ({step.get('role')}) | "
            f"to={address} | llm_elapsed_ms={step.get('llm_elapsed_ms')}"
        )
    if total_ms is not None:
        print(f"total_round_elapsed_ms: {total_ms}")


def print_downstream_input(downstream_input: Dict[str, Any]) -> None:
    if not downstream_input:
        return
    print(json.dumps(downstream_input, ensure_ascii=False, indent=2))


def print_single_round_case(case_label: str, result: Dict[str, Any]) -> None:
    print(f"\n===== {case_label} | 单轮群聊 =====")
    print("query:", result["query"])
    print("speaker_order:", result["planner_result"]["speaker_order"])
    print("role_assignment:", result["planner_result"]["role_assignment"])
    print("check_trigger:", result["planner_result"].get("check_trigger"))
    print("summary_trigger:", result["planner_result"]["summary_trigger"])
    print("execution_steps:")
    for step in result["planner_result"].get("final_plan", {}).get("execution_steps", []):
        print(
            f"  - step {step.get('step_id')}: {step.get('agent_id')} ({step.get('role')}) "
            f"-> {format_address_to_names(step.get('address_to_names', []))}"
        )
    print("final_plan:")
    print(json.dumps(result.get("final_plan", result["planner_result"].get("final_plan", {})), ensure_ascii=False, indent=2))
    print("downstream_input:")
    print_downstream_input(result.get("downstream_input", {}))
    print("token_usage:")
    print_token_usage(result.get("token_usage", {}))
    print("module_timings:")
    print_timings(result.get("module_timings", {}))
    for item in result["generated_outputs"]:
        print(
            f"- {item['agent_name']} ({item['role']}) -> {format_address_to_names(item.get('address_to_names', []))}: {item['text']}"
        )


def print_multi_round_case(case_label: str, result: Dict[str, Any]) -> None:
    print(f"\n===== {case_label} | 多轮群聊 =====")
    print("query:", result["query"])
    print("executed_rounds:", result["executed_rounds"])
    print("stop_signal:", result["stop_signal"])
    for local_idx, round_item in enumerate(result["rounds"], start=1):
        planner_result = round_item["planner_result"]
        print(f"\n--- Local Round {local_idx} | Global Round {round_item['round_id']} ---")
        print("speaker_order:", planner_result["speaker_order"])
        print("check_trigger:", planner_result.get("check_trigger"))
        print("summary_trigger:", planner_result.get("summary_trigger"))
        print("execution_steps:")
        for step in planner_result.get("final_plan", {}).get("execution_steps", []):
            print(
                f"  - step {step.get('step_id')}: {step.get('agent_id')} ({step.get('role')}) "
                f"-> {format_address_to_names(step.get('address_to_names', []))}"
            )
        print("final_plan:")
        print(json.dumps(round_item.get("final_plan", planner_result.get("final_plan", {})), ensure_ascii=False, indent=2))
        print("downstream_input:")
        print_downstream_input(round_item.get("downstream_input", {}))
        print("token_usage:")
        print_token_usage(round_item.get("token_usage", {}))
        print("module_timings:")
        print_timings(round_item.get("module_timings", {}))
        for item in round_item["generated_outputs"]:
            print(
                f"- {item['agent_name']} ({item['role']}) -> {format_address_to_names(item.get('address_to_names', []))}: {item['text']}"
            )
        print("history_summary_after:", json.dumps(round_item["history_summary_after"], ensure_ascii=False))


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_results(output_path: str, results: Dict[str, Any]) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        ensure_output_dir(parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_interactive_help() -> None:
    print("\n可用命令：")
    print("  /help      查看帮助")
    print("  /reset     清空历史，重新开始一个新话题")
    print("  /return    下一轮强制 topic_action=return")
    print("  /maintain  下一轮恢复 topic_action=maintain")
    print("  /state     查看当前共享状态")
    print("  /exit      退出交互模式")


def print_ranking_summary(planner_result: Dict[str, Any]) -> None:
    ranking = planner_result.get("ranking", {}) or {}
    decision = ranking.get("decision", {}) or {}
    main_rank = ranking.get("main", []) or []
    supplement_rank = ranking.get("supplement", []) or []

    if decision:
        print("ranking_decision:", json.dumps(decision, ensure_ascii=False))

    if main_rank:
        print("main_ranking:")
        for item in main_rank[:5]:
            print(
                f"  - #{item.get('rank_position')} {item.get('name')} | score={item.get('final_score')} | "
                f"query_focus={item.get('query_focus')} | topic_focus={item.get('topic_focus')} | "
                f"repeat_penalty={item.get('repeat_penalty')}"
            )

    if supplement_rank:
        print("supplement_ranking:")
        for item in supplement_rank[:5]:
            extra = f" | diversity_from_main={item.get('diversity_from_main')}" if item.get('diversity_from_main') is not None else ""
            print(
                f"  - #{item.get('rank_position')} {item.get('name')} | score={item.get('final_score')} | "
                f"topic_focus={item.get('topic_focus')} | repeat_penalty={item.get('repeat_penalty')}{extra}"
            )


def print_interactive_round(turn_index: int, user_text: str, result: Dict[str, Any]) -> None:
    planner_result = result["planner_result"]
    print(f"\n{'=' * 96}")
    print(f"Round {turn_index}")
    print(f"用户: {user_text}")
    print(f"{'=' * 96}")
    print("speaker_order:", planner_result.get("speaker_order", []))
    print("role_assignment:", planner_result.get("role_assignment", []))
    print("check_trigger:", planner_result.get("check_trigger"))
    print("summary_trigger:", planner_result.get("summary_trigger"))
    print_ranking_summary(planner_result)
    print("execution_steps:")
    for step in planner_result.get("final_plan", {}).get("execution_steps", []):
        address = format_address_to_names(step.get("address_to_names", []) or [])
        print(f"  - step {step.get('step_id')}: {step.get('agent_id')} ({step.get('role')}) -> {address}")
        if step.get("address_note"):
            print(f"      note: {step.get('address_note')}")
    print("final_plan:")
    print(json.dumps(result.get("final_plan", planner_result.get("final_plan", {})), ensure_ascii=False, indent=2))
    print("downstream_input:")
    print_downstream_input(result.get("downstream_input", {}))
    print("token_usage:")
    print_token_usage(result.get("token_usage", {}))
    print("module_timings:")
    print_timings(result.get("module_timings", {}))
    print("generated_outputs:")
    for item in result.get("generated_outputs", []):
        address = format_address_to_names(item.get("address_to_names", []) or [])
        print(f"- {item['agent_name']} ({item['role']}) -> {address}: {item['text']}")


def run_interactive_session(
    llm: QwenClient,
    semantic: Optional[SemanticHelper] = None,
    temperature: float = 0.8,
    max_tokens: int = 400,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    semantic = semantic or SemanticHelper()
    shared_state: Dict[str, Any] = {
        "agent_list": get_shared_agents(),
        "history_summary": {"done_points": [], "resolved": False},
        "last_round_outputs": [],
        "topic_state": {},
        "pending_topic_action": "maintain",
    }
    transcript: List[Dict[str, Any]] = []
    turn_index = 1

    print("进入 interactive 模式。你在终端输入一句话，系统会模拟一轮明星群聊。")
    print_interactive_help()

    while True:
        try:
            user_text = input("\n你：").strip()
        except EOFError:
            print("\n检测到 EOF，结束交互。")
            break
        except KeyboardInterrupt:
            print("\n用户中断，结束交互。")
            break

        if not user_text:
            continue
        lowered = user_text.lower()

        if lowered in QUIT_COMMANDS:
            print("已退出 interactive 模式。")
            break
        if lowered in HELP_COMMANDS:
            print_interactive_help()
            continue
        if lowered in RESET_COMMANDS:
            shared_state["history_summary"] = {"done_points": [], "resolved": False}
            shared_state["last_round_outputs"] = []
            shared_state["topic_state"] = {}
            shared_state["pending_topic_action"] = "maintain"
            transcript.append({"turn_index": turn_index, "event": "reset"})
            print("历史已清空。")
            continue
        if lowered in RETURN_COMMANDS:
            shared_state["pending_topic_action"] = "return"
            print("下一轮已设为 topic_action=return。你接下来输入的话会触发强制收束。")
            continue
        if lowered in MAINTAIN_COMMANDS:
            shared_state["pending_topic_action"] = "maintain"
            print("下一轮已设为 topic_action=maintain。")
            continue
        if lowered in STATE_COMMANDS:
            state_view = {
                "pending_topic_action": shared_state.get("pending_topic_action", "maintain"),
                "history_done_points": (shared_state.get("history_summary", {}) or {}).get("done_points", []),
                "last_topic_state": shared_state.get("topic_state", {}),
            }
            print(json.dumps(state_view, ensure_ascii=False, indent=2))
            continue

        payload = build_interactive_payload(user_text=user_text, shared_state=shared_state, turn_index=turn_index)
        result = generate_one_round(
            llm=llm,
            payload=payload,
            semantic=semantic,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print_interactive_round(turn_index=turn_index, user_text=user_text, result=result)

        updated_summary = summarize_round_outputs(
            sanitize_history_summary(shared_state.get("history_summary", {"done_points": []})),
            result.get("generated_outputs", []),
            semantic=semantic,
        )
        shared_state["history_summary"] = updated_summary
        shared_state["last_round_outputs"] = [
            {
                "agent_id": item["agent_id"],
                "role": item["role"],
                "text": item["text"],
                "key_points": item["key_points"],
            }
            for item in result.get("generated_outputs", [])
        ]
        shared_state["topic_state"] = dict(payload.get("topic_state", {}))
        apply_next_topic_state(shared_state, result.get("planner_result", {}))
        shared_state["pending_topic_action"] = "maintain"

        transcript.append(
            {
                "turn_index": turn_index,
                "user_text": user_text,
                "payload": payload,
                "result": result,
                "history_summary_after": updated_summary,
            }
        )
        turn_index += 1

    final_results = {
        "mode": "interactive",
        "turns": transcript,
        "history_summary": shared_state.get("history_summary", {}),
        "last_round_outputs": shared_state.get("last_round_outputs", []),
        "pending_topic_action": shared_state.get("pending_topic_action", "maintain"),
    }
    if output_path:
        save_results(output_path, final_results)
        print(f"\n交互记录已保存到：{output_path}")
    return final_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi", "interactive"], default="multi")
    parser.add_argument("--case", default="all", help="all 或 case 编号，例如 3")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--output", default="outputs/qwen_star_group_chat_check_layer_results.json")
    parser.add_argument("--orchestrator-module", default="demo_orchestrator", help="调度器模块名、.py 文件名或相对路径")
    parser.add_argument("--case-id", action="append", default=[],help="只跑指定 case_id，可重复，例如 --case-id pc01 --case-id pc02")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_orchestrator_runtime(args.orchestrator_module)

    llm = QwenClient(model=args.model, base_url=args.base_url, api_key=args.api_key)
    semantic = SemanticHelper()

    if args.mode == "interactive":
        run_interactive_session(
            llm=llm,
            semantic=semantic,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_path=args.output,
        )
        return

    agents = get_shared_agents()
    cases = get_cases(agents)

    if args.case_id:
        keep = set(args.case_id)
        indexed_cases = list(enumerate(cases, start=1))
        filtered = [(idx, case) for idx, case in indexed_cases if case.get("case_id") in keep]
        if not filtered:
            raise ValueError(f"没有找到指定的 case_id：{args.case_id}")
        selected_indices = [idx for idx, _ in filtered]
        chosen_cases = [case for _, case in filtered]
    elif args.case == "all":
        chosen_cases = cases
        selected_indices = list(range(1, len(cases) + 1))
    else:
        case_idx = int(args.case)
        if case_idx < 1 or case_idx > len(cases):
            raise ValueError(f"case 编号越界，当前可选范围为 1~{len(cases)}")
        chosen_cases = [cases[case_idx - 1]]
        selected_indices = [case_idx]

    final_results: Dict[str, Any] = {"mode": args.mode, "model": args.model, "orchestrator_module": args.orchestrator_module, "cases": []}

    for display_idx, case in zip(selected_indices, chosen_cases):
        case_label = f"CASE {display_idx}"
        if args.mode == "single":
            result = run_single_round_case(llm=llm, case=case, semantic=semantic, temperature=args.temperature, max_tokens=args.max_tokens)
            print_single_round_case(case_label, result)
        else:
            result = run_multi_round_case(llm=llm, case=case, max_rounds=args.rounds, semantic=semantic, temperature=args.temperature, max_tokens=args.max_tokens)
            print_multi_round_case(case_label, result)

        final_results["cases"].append({"case_index": display_idx, "case_query": case["query"], "result": result})

    save_results(args.output, final_results)
    print(f"\n结果已保存到：{args.output}")


if __name__ == "__main__":
    main()
