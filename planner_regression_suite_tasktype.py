from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import importlib
import importlib.util
import sys


def _import_sample_cases_module(module_ref: str = "sample_cases"):
    module_name = (module_ref or "sample_cases").strip() or "sample_cases"
    if module_name.endswith(".py") or any(sep in module_name for sep in ("/", "\\")):
        module_path = Path(module_name)
        if not module_path.exists():
            raise FileNotFoundError(f"找不到 sample cases 文件：{module_path}")
        runtime_name = f"{module_path.stem}_sample_cases_runtime"
        spec = importlib.util.spec_from_file_location(runtime_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从文件加载 sample cases：{module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[runtime_name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(module_name[:-3] if module_name.endswith(".py") else module_name)


def _import_orchestrator_module(orchestrator_module: str = "demo_orchestrator"):
    module_ref = (orchestrator_module or "demo_orchestrator").strip() or "demo_orchestrator"
    if module_ref.endswith(".py") or any(sep in module_ref for sep in ("/", "\\")):
        module_path = Path(module_ref)
        if not module_path.exists():
            raise FileNotFoundError(f"找不到 orchestrator 文件：{module_path}")
        runtime_name = f"{module_path.stem}_orchestrator_runtime"
        spec = importlib.util.spec_from_file_location(runtime_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法从文件加载 orchestrator：{module_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[runtime_name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(module_ref[:-3] if module_ref.endswith(".py") else module_ref)

CaseDict = Dict[str, Any]
ResultDict = Dict[str, Any]

ROLE_MAIN = "main"
ROLE_SUPPLEMENT = "supplement"
ROLE_CHECK = "check"
ROLE_SUMMARY = "summary"
SUMMARY_REASON_TOPIC_RETURN = "topic_return"


@dataclass
class CheckItem:
    ok: bool
    name: str
    actual: Any = None
    expected: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "name": self.name,
            "actual": self.actual,
            "expected": self.expected,
        }


def _context_from_last_round(last_round_outputs: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in last_round_outputs or []:
        speaker = item.get("agent_name") or item.get("name") or item.get("agent_id") or "某人"
        role = item.get("role", "") or ""
        text = (item.get("text", "") or "").strip()
        if not text:
            continue
        role_suffix = f"（{role}）" if role else ""
        lines.append(f"{speaker}{role_suffix}：{text}")
    return "\n".join(lines)


def detect_context_source(case: CaseDict) -> Tuple[str, str]:
    if case.get("context_query"):
        return "context_query", str(case.get("context_query") or "")
    if case.get("last_round_outputs"):
        return "last_round_outputs", _context_from_last_round(case.get("last_round_outputs", []) or [])
    return "query", str(case.get("query") or "")


def apply_threshold_overrides(orchestrator: Any, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(orchestrator, key):
            raise AttributeError(f"MinimalOrchestrator 没有阈值字段：{key}")
        setattr(orchestrator, key, value)


def _safe_get(result: ResultDict, *keys: str, default: Any = None) -> Any:
    current: Any = result
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def infer_path_signature(result: ResultDict) -> Dict[str, Any]:
    role_assignment = result.get("role_assignment", []) or []
    roles = [x.get("role") for x in role_assignment]
    skip_mute = result.get("skip_mute", []) or []
    summary_reason = _safe_get(result, "summary_trigger", "reason")
    planner_meta = _safe_get(result, "final_plan", "planner_meta", default={}) or {}

    return {
        "has_main": ROLE_MAIN in roles,
        "has_supplement": ROLE_SUPPLEMENT in roles,
        "has_check": ROLE_CHECK in roles,
        "has_summary": ROLE_SUMMARY in roles,
        "has_skip": any((x.get("action") == "skip") for x in skip_mute),
        "has_mute": any((x.get("action") == "mute") for x in skip_mute),
        "has_return": bool(
            summary_reason == SUMMARY_REASON_TOPIC_RETURN
            or planner_meta.get("forced_summary")
            or _safe_get(result, "diagnostics", "direct_return_to_topic", default=False)
        ),
        "speaker_count": len(result.get("speaker_order", []) or []),
        "roles": roles,
    }


def _check_equals(name: str, actual: Any, expected: Any) -> CheckItem:
    return CheckItem(ok=(actual == expected), name=name, actual=actual, expected=expected)


def _check_contains(name: str, actual_items: Sequence[Any], expected_item: Any) -> CheckItem:
    return CheckItem(ok=(expected_item in actual_items), name=name, actual=list(actual_items), expected=expected_item)


def _check_not_contains(name: str, actual_items: Sequence[Any], expected_item: Any) -> CheckItem:
    return CheckItem(ok=(expected_item not in actual_items), name=name, actual=list(actual_items), expected=expected_item)


def validate_result(case: CaseDict, result: ResultDict, orchestrator: MinimalOrchestrator) -> Dict[str, Any]:
    expected = case.get("expected", {}) or {}
    checks: List[CheckItem] = []

    summary_trigger = _safe_get(result, "summary_trigger", "triggered", default=False)
    summary_reason = _safe_get(result, "summary_trigger", "reason")
    check_trigger = _safe_get(result, "check_trigger", "triggered", default=False)
    check_reason = _safe_get(result, "check_trigger", "reason")
    role_assignment = result.get("role_assignment", []) or []
    roles = [x.get("role") for x in role_assignment]
    speaker_order = result.get("speaker_order", []) or []
    skip_reasons = [x.get("reason") for x in (result.get("skip_mute", []) or [])]
    final_plan = result.get("final_plan", {}) or {}
    next_topic_state = final_plan.get("next_topic_state", {}) or {}
    planner_meta = final_plan.get("planner_meta", {}) or {}
    execution_steps = final_plan.get("execution_steps", []) or []
    diagnostics = result.get("diagnostics", {}) or {}

    actual_query = orchestrator._build_context_query(case)
    actual_context_source, expected_query_text = detect_context_source(case)

    if "expected_context_source" in expected:
        checks.append(_check_equals("context_source", actual_context_source, expected["expected_context_source"]))
        checks.append(_check_equals("effective_query_text", actual_query, expected_query_text))

    if "summary_trigger" in expected:
        checks.append(_check_equals("summary_trigger", summary_trigger, expected["summary_trigger"]))

    role_to_agents: Dict[str, List[str]] = {}
    for item in role_assignment:
        role = item.get("role")
        agent_id = item.get("agent_id")
        if role and agent_id:
            role_to_agents.setdefault(role, []).append(agent_id)

    main_agents = role_to_agents.get(ROLE_MAIN, [])
    supplement_agents = role_to_agents.get(ROLE_SUPPLEMENT, [])

    role_to_steps: Dict[str, List[Dict[str, Any]]] = {}
    for step in execution_steps:
        role = step.get("role")
        if role:
            role_to_steps.setdefault(role, []).append(step)
    if "summary_reason" in expected:
        checks.append(_check_equals("summary_reason", summary_reason, expected["summary_reason"]))
    if "check_trigger" in expected:
        checks.append(_check_equals("check_trigger", check_trigger, expected["check_trigger"]))
    if "check_reason" in expected:
        checks.append(_check_equals("check_reason", check_reason, expected["check_reason"]))

    for role in expected.get("must_have_roles", []) or []:
        checks.append(_check_contains(f"must_have_role:{role}", roles, role))
    for role in expected.get("must_not_have_roles", []) or []:
        checks.append(_check_not_contains(f"must_not_have_role:{role}", roles, role))

    preferred_main_any = expected.get("preferred_main_any", []) or []
    if preferred_main_any:
        checks.append(CheckItem(
            ok=bool(main_agents) and any(agent_id in preferred_main_any for agent_id in main_agents),
            name="preferred_main_any",
            actual=main_agents,
            expected=preferred_main_any,
        ))

    preferred_supplement_any = expected.get("preferred_supplement_any", []) or []
    if preferred_supplement_any:
        checks.append(CheckItem(
            ok=bool(supplement_agents) and any(agent_id in preferred_supplement_any for agent_id in supplement_agents),
            name="preferred_supplement_any",
            actual=supplement_agents,
            expected=preferred_supplement_any,
        ))

    forbid_main = expected.get("forbid_main", []) or []
    if forbid_main:
        checks.append(CheckItem(
            ok=all(agent_id not in forbid_main for agent_id in main_agents),
            name="forbid_main",
            actual=main_agents,
            expected=forbid_main,
        ))


    role_key_map = {
        "main": ROLE_MAIN,
        "supplement": ROLE_SUPPLEMENT,
        "check": ROLE_CHECK,
        "summary": ROLE_SUMMARY,
    }
    for role_prefix, role_name in role_key_map.items():
        role_steps = role_to_steps.get(role_name, []) or []
        first_step = role_steps[0] if role_steps else {}

        expected_task_type = expected.get(f"expected_{role_prefix}_task_type")
        if expected_task_type is not None:
            checks.append(_check_equals(
                f"{role_prefix}_task_type",
                first_step.get("task_type"),
                expected_task_type,
            ))

        expected_seriousness = expected.get(f"expected_{role_prefix}_seriousness")
        if expected_seriousness is not None:
            checks.append(_check_equals(
                f"{role_prefix}_seriousness",
                first_step.get("topic_seriousness"),
                expected_seriousness,
            ))

        expected_address_style = expected.get(f"expected_{role_prefix}_address_style")
        if expected_address_style is not None:
            checks.append(_check_equals(
                f"{role_prefix}_address_style",
                first_step.get("address_style"),
                expected_address_style,
            ))

        expected_opening_contains = expected.get(f"expected_{role_prefix}_opening_hint_contains")
        if expected_opening_contains:
            actual_hint = str(first_step.get("opening_hint", "") or "")
            checks.append(CheckItem(
                ok=expected_opening_contains in actual_hint,
                name=f"{role_prefix}_opening_hint_contains",
                actual=actual_hint,
                expected=expected_opening_contains,
            ))

        expected_address_any = expected.get(f"expected_{role_prefix}_address_to_any", []) or []
        if expected_address_any:
            actual_address = list(first_step.get("address_to", []) or [])
            checks.append(CheckItem(
                ok=bool(actual_address) and any(x in actual_address for x in expected_address_any),
                name=f"{role_prefix}_address_to_any",
                actual=actual_address,
                expected=expected_address_any,
            ))

        expected_address_all = expected.get(f"expected_{role_prefix}_address_to_all", []) or []
        if expected_address_all:
            actual_address = list(first_step.get("address_to", []) or [])
            checks.append(CheckItem(
                ok=all(x in actual_address for x in expected_address_all),
                name=f"{role_prefix}_address_to_all",
                actual=actual_address,
                expected=expected_address_all,
            ))

    if "exact_speaker_count" in expected:
        checks.append(_check_equals("speaker_count", len(speaker_order), expected["exact_speaker_count"]))
    if "max_speakers_lte" in expected:
        checks.append(CheckItem(ok=(len(speaker_order) <= expected["max_speakers_lte"]), name="max_speakers_lte", actual=len(speaker_order), expected=expected["max_speakers_lte"]))
    if "min_speakers_gte" in expected:
        checks.append(CheckItem(ok=(len(speaker_order) >= expected["min_speakers_gte"]), name="min_speakers_gte", actual=len(speaker_order), expected=expected["min_speakers_gte"]))
    if "expected_first_speaker" in expected:
        first = speaker_order[0] if speaker_order else None
        checks.append(_check_equals("expected_first_speaker", first, expected["expected_first_speaker"]))
    if "expected_speaker_order" in expected:
        checks.append(_check_equals("expected_speaker_order", speaker_order, expected["expected_speaker_order"]))

    expect_skip_reasons_any = expected.get("expect_skip_reasons_any", []) or []
    if expect_skip_reasons_any:
        checks.append(CheckItem(ok=any(x in skip_reasons for x in expect_skip_reasons_any), name="expect_skip_reasons_any", actual=skip_reasons, expected=expect_skip_reasons_any))

    expect_skip_reasons_all = expected.get("expect_skip_reasons_all", []) or []
    if expect_skip_reasons_all:
        checks.append(CheckItem(ok=all(x in skip_reasons for x in expect_skip_reasons_all), name="expect_skip_reasons_all", actual=skip_reasons, expected=expect_skip_reasons_all))

    if "expected_next_main_topic" in expected:
        checks.append(_check_equals("expected_next_main_topic", next_topic_state.get("main_topic"), expected["expected_next_main_topic"]))
    if "expected_next_topic_action" in expected:
        checks.append(_check_equals("expected_next_topic_action", next_topic_state.get("topic_action"), expected["expected_next_topic_action"]))
    if "expected_forced_summary" in expected:
        checks.append(_check_equals("expected_forced_summary", planner_meta.get("forced_summary"), expected["expected_forced_summary"]))

    if "expected_redundancy_gte" in expected:
        actual_red = float(diagnostics.get("redundancy_ratio", 0.0) or 0.0)
        checks.append(CheckItem(ok=(actual_red >= expected["expected_redundancy_gte"]), name="expected_redundancy_gte", actual=actual_red, expected=expected["expected_redundancy_gte"]))
    if "expected_redundancy_lt" in expected:
        actual_red = float(diagnostics.get("redundancy_ratio", 0.0) or 0.0)
        checks.append(CheckItem(ok=(actual_red < expected["expected_redundancy_lt"]), name="expected_redundancy_lt", actual=actual_red, expected=expected["expected_redundancy_lt"]))

    signature = infer_path_signature(result)
    case_pass = all(item.ok for item in checks)
    return {
        "case_id": case.get("case_id", ""),
        "title": case.get("title", ""),
        "target_behavior": case.get("target_behavior", ""),
        "passed": case_pass,
        "checks": [item.to_dict() for item in checks],
        "path_signature": signature,
        "summary_trigger": summary_trigger,
        "summary_reason": summary_reason,
        "check_trigger": check_trigger,
        "check_reason": check_reason,
        "speaker_order": speaker_order,
        "role_assignment": role_assignment,
        "skip_mute": result.get("skip_mute", []) or [],
        "diagnostics_brief": {
            "redundancy_ratio": diagnostics.get("redundancy_ratio"),
            "coverage_state": diagnostics.get("coverage_state"),
            "relation_counter": _safe_get(diagnostics, "relation_info", "relation_counter", default={}),
            "highest_risk": _safe_get(diagnostics, "relation_info", "highest_risk"),
            "ranking": result.get("ranking", {}) or {},
            "score_details": (result.get("diagnostics", {}) or {}).get("score_details", []) or [],
            "execution_steps": [
                {
                    "agent_id": step.get("agent_id"),
                    "role": step.get("role"),
                    "task_type": step.get("task_type"),
                    "topic_seriousness": step.get("topic_seriousness"),
                    "address_to": step.get("address_to", []),
                    "address_style": step.get("address_style"),
                    "opening_hint": step.get("opening_hint"),
                }
                for step in execution_steps
            ],
        },
    }


def summarize_coverage(case_reports: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    counter: Counter[str] = Counter()

    if case_reports is None:
        return {
            "counts": {},
            "missing_required_paths": [
                "has_main", "has_supplement", "has_check", "has_summary", "has_skip", "has_return"
            ],
            "required_paths_all_hit": False,
            "error": "case_reports is None",
        }

    for report in case_reports:
        signature = report.get("path_signature", {}) or {}
        for key in ["has_main", "has_supplement", "has_check", "has_summary", "has_skip", "has_mute", "has_return"]:
            if signature.get(key):
                counter[key] += 1

    required = ["has_main", "has_supplement", "has_check", "has_summary", "has_skip", "has_return"]
    missing = [key for key in required if counter.get(key, 0) == 0]
    return {
        "counts": dict(counter),
        "missing_required_paths": missing,
        "required_paths_all_hit": not missing,
    }


def run_regression_suite(
    overrides: Optional[Dict[str, Any]] = None,
    case_ids: Optional[Sequence[str]] = None,
    orchestrator_module: str = "demo_orchestrator",
    sample_cases_module: str = "sample_cases_tasktype.py",
) -> Dict[str, Any]:
    sample_cases_mod = _import_sample_cases_module(sample_cases_module)
    agents = sample_cases_mod.get_shared_agents()
    cases = sample_cases_mod.get_cases(agents)
    if case_ids:
        keep = set(case_ids)
        cases = [case for case in cases if case.get("case_id") in keep]

    orchestrator_mod = _import_orchestrator_module(orchestrator_module)
    MinimalOrchestratorCls = getattr(orchestrator_mod, "MinimalOrchestrator")
    orchestrator = MinimalOrchestratorCls()
    overrides = overrides or {}
    if overrides:
        apply_threshold_overrides(orchestrator, overrides)

    case_reports: List[Dict[str, Any]] = []
    total_checks = 0
    passed_checks = 0
    passed_cases = 0

    for case in cases:
        payload = copy.deepcopy(case)
        result = orchestrator.plan(payload)
        report = validate_result(case=payload, result=result, orchestrator=orchestrator)

        # 注意：这里只能 append，不能写成 case_reports = case_reports.append(report)
        case_reports.append(report)

        checks = report["checks"]
        total_checks += len(checks)
        passed_checks += sum(1 for item in checks if item["ok"])
        passed_cases += int(report["passed"])

    coverage = summarize_coverage(case_reports)
    return {
        "orchestrator_module": orchestrator_module,
        "threshold_overrides": overrides,
        "summary": {
            "case_count": len(case_reports),
            "passed_cases": passed_cases,
            "case_pass_rate": round((passed_cases / len(case_reports)) if case_reports else 0.0, 4),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "check_pass_rate": round((passed_checks / total_checks) if total_checks else 0.0, 4),
        },
        "coverage": coverage,
        "case_reports": case_reports,
    }


def _parse_number(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_key_value_list(items: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"参数格式错误，应为 key=value：{item}")
        key, value = item.split("=", 1)
        result[key.strip()] = _parse_number(value)
    return result


def parse_sweep_items(items: Sequence[str]) -> Dict[str, List[Any]]:
    sweep: Dict[str, List[Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"sweep 参数格式错误，应为 key=v1,v2,v3：{item}")
        key, raw_values = item.split("=", 1)
        values = [_parse_number(x) for x in raw_values.split(",") if x.strip()]
        if not values:
            raise ValueError(f"sweep 参数没有可用取值：{item}")
        sweep[key.strip()] = values
    return sweep


def generate_sweep_configs(sweep: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    value_lists = [sweep[key] for key in keys]
    configs: List[Dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        configs.append({k: v for k, v in zip(keys, combo)})
    return configs


def print_suite_report(report: Dict[str, Any], verbose: bool = True) -> None:
    summary = report.get("summary", {}) or {}
    coverage = report.get("coverage", {}) or {}
    overrides = report.get("threshold_overrides", {}) or {}

    print("\n" + "=" * 96)
    print(f"threshold_overrides: {json.dumps(overrides, ensure_ascii=False)}")
    print(f"summary: {json.dumps(summary, ensure_ascii=False)}")
    print(f"coverage: {json.dumps(coverage, ensure_ascii=False)}")


    if not verbose:
        return

    for case_report in report.get("case_reports", []) or []:
        print("\n" + "-" * 96)
        print(f"{case_report.get('case_id')} | {case_report.get('title')}")
        print(f"target_behavior: {case_report.get('target_behavior')}")
        print(f"passed: {case_report.get('passed')}")
        print(f"path_signature: {json.dumps(case_report.get('path_signature', {}), ensure_ascii=False)}")
        print(f"speaker_order: {case_report.get('speaker_order')}")
        print(f"role_assignment: {case_report.get('role_assignment')}")
        print(f"skip_mute: {case_report.get('skip_mute')}")
        print(f"diagnostics_brief: {json.dumps(case_report.get('diagnostics_brief', {}), ensure_ascii=False)}")
        print(f"ranking_decision: {json.dumps(case_report.get('ranking', {}).get('decision', {}), ensure_ascii=False)}")
        print(f"main_ranking_top3: {json.dumps((case_report.get('ranking', {}).get('main', [])[:3]), ensure_ascii=False)}")
        print(f"score_details_top5: {json.dumps((case_report.get('score_details', [])[:5]), ensure_ascii=False)}")
        for item in case_report.get("checks", []) or []:
            flag = "PASS" if item.get("ok") else "FAIL"
            print(f"  [{flag}] {item.get('name')} | actual={item.get('actual')} | expected={item.get('expected')}")


def rank_sweep_reports(reports: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        reports,
        key=lambda x: (
            x.get("summary", {}).get("case_pass_rate", 0.0),
            x.get("summary", {}).get("check_pass_rate", 0.0),
            x.get("coverage", {}).get("required_paths_all_hit", False),
        ),
        reverse=True,
    )

from pathlib import Path
import json


def build_suite_report_text(report: Dict[str, Any], verbose: bool = True) -> str:
    summary = report.get("summary", {}) or {}
    coverage = report.get("coverage", {}) or {}
    overrides = report.get("threshold_overrides", {}) or {}

    lines: List[str] = []
    lines.append("\n" + "=" * 96)
    lines.append(f"threshold_overrides: {json.dumps(overrides, ensure_ascii=False)}")
    lines.append(f"summary: {json.dumps(summary, ensure_ascii=False)}")
    lines.append(f"coverage: {json.dumps(coverage, ensure_ascii=False)}")

    if not verbose:
        return "\n".join(lines)

    for case_report in report.get("case_reports", []) or []:
        lines.append("\n" + "-" * 96)
        lines.append(f"{case_report.get('case_id')} | {case_report.get('title')}")
        lines.append(f"target_behavior: {case_report.get('target_behavior')}")
        lines.append(f"passed: {case_report.get('passed')}")
        lines.append(f"path_signature: {json.dumps(case_report.get('path_signature', {}), ensure_ascii=False)}")
        lines.append(f"speaker_order: {case_report.get('speaker_order')}")
        lines.append(f"role_assignment: {case_report.get('role_assignment')}")
        lines.append(f"skip_mute: {case_report.get('skip_mute')}")
        lines.append(f"diagnostics_brief: {json.dumps(case_report.get('diagnostics_brief', {}), ensure_ascii=False)}")
        for item in case_report.get("checks", []) or []:
            flag = "PASS" if item.get("ok") else "FAIL"
            lines.append(
                f"  [{flag}] {item.get('name')} | actual={item.get('actual')} | expected={item.get('expected')}"
            )

    return "\n".join(lines)


def print_suite_report(report: Dict[str, Any], verbose: bool = True) -> None:
    print(build_suite_report_text(report, verbose=verbose))


def save_text_report(output_path: str, text: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_sweep_report_text(
    ranked_reports: Sequence[Dict[str, Any]],
    topk: int = 10,
    verbose: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("===== SWEEP TOP CONFIGS =====")
    for idx, report in enumerate(ranked_reports[:topk], start=1):
        lines.append(
            f"#{idx} | overrides={json.dumps(report.get('threshold_overrides', {}), ensure_ascii=False)} | "
            f"case_pass_rate={report.get('summary', {}).get('case_pass_rate')} | "
            f"check_pass_rate={report.get('summary', {}).get('check_pass_rate')} | "
            f"coverage_ok={report.get('coverage', {}).get('required_paths_all_hit')}"
        )

    best = ranked_reports[0] if ranked_reports else None
    if best is not None:
        lines.append("")
        lines.append(build_suite_report_text(best, verbose=verbose))

    return "\n".join(lines)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多智能体调度器：阈值调参与路径回归测试")
    parser.add_argument("--set", action="append", default=[], help="单次运行的阈值覆盖，格式 key=value，可重复")
    parser.add_argument("--sweep", action="append", default=[], help="阈值扫参，格式 key=v1,v2,v3，可重复；会做笛卡尔积")
    parser.add_argument("--case-id", action="append", default=[], help="只跑指定 case_id，可重复")
    parser.add_argument("--topk", type=int, default=10, help="扫参时输出前 k 组配置")
    parser.add_argument("--quiet", action="store_true", help="只打印总览，不打印每个 case 的细节")
    parser.add_argument("--json-output", default="", help="把结果写入 json 文件")
    parser.add_argument("--txt-output", default="", help="把摘要报告写入 txt 文件")
    parser.add_argument("--sample-cases-module", default="sample_cases_tasktype.py", help="sample cases 模块名、.py 文件名或相对路径")
    parser.add_argument("--orchestrator-module", default="demo_orchestrator", help="调度器模块名、.py 文件名或相对路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_overrides = parse_key_value_list(args.set)
    sweep_space = parse_sweep_items(args.sweep)
    sweep_configs = generate_sweep_configs(sweep_space)

    reports: List[Dict[str, Any]] = []
    for config in sweep_configs:
        merged = dict(base_overrides)
        merged.update(config)
        report = run_regression_suite(
            overrides=merged,
            case_ids=args.case_id,
            orchestrator_module=args.orchestrator_module,
            sample_cases_module=args.sample_cases_module,
        )
        reports.append(report)

    ranked = rank_sweep_reports(reports)

    if sweep_space:
        txt_report = build_sweep_report_text(
            ranked_reports=ranked,
            topk=args.topk,
            verbose=not args.quiet,
        )
        print(txt_report)
    else:
        txt_report = build_suite_report_text(ranked[0], verbose=not args.quiet)
        print(txt_report)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_overrides": base_overrides,
            "sweep_space": sweep_space,
            "reports": ranked,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 结果已保存到：{output_path}")

    if args.txt_output:
        save_text_report(args.txt_output, txt_report)
        print(f"TXT 摘要报告已保存到：{args.txt_output}")


if __name__ == "__main__":
    main()
