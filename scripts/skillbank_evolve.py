#!/usr/bin/env python3

import argparse
import ast
import collections
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


CANDIDATE_ID_COUNTER = 0
SELF_REFINE_MAX_ATTEMPTS = 3

PROBLEM_TYPES = [
    "allocation",
    "assignment",
    "selection",
    "flow",
    "time_planning",
    "routing",
    "scheduling",
    "special",
]

TYPE_PREFIX = {
    "allocation": "alloca",
    "assignment": "assign",
    "selection": "select",
    "flow": "flow",
    "time_planning": "timeplan",
    "routing": "route",
    "scheduling": "sched",
    "special": "special",
}

PROMPT_KEYS = [
    "base_generation_system_prompt",
    "base_generation_user_prompt",
    "problem_type_classification_system_prompt",
    "problem_type_classification_user_prompt",
    "skill_retrieval_system_prompt",
    "skill_retrieval_user_prompt",
    "skill_augmented_generation_system_prompt",
    "skill_augmented_generation_user_prompt",
    "self_refine_system_prompt",
    "self_refine_user_prompt",
    "new_strategy_system_prompt",
    "new_strategy_user_prompt",
    "new_experience_system_prompt",
    "new_experience_user_prompt",
    "repair_skill_system_prompt",
    "repair_skill_user_prompt",
    "candidate_dedup_system_prompt",
    "candidate_dedup_user_prompt",
]

RESULT_PREFIX = "Just print the best solution:"
NO_SOLUTION_TEXT = "No Best Solution"
ADD_SCRIPT = """
if model.status == GRB.OPTIMAL:
    print(f"Just print the best solution: {model.objVal}")
else:
    print("No Best Solution")
""".strip()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, target)


def write_text(path: str, data: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, target)


def safe_path_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unknown"


def dataset_stem(dataset: Any) -> str:
    return safe_path_part(str(dataset).lower())


def step_dir_name(step: Any) -> str:
    return f"step_{safe_int(step, 0):03d}"


def checkpoint_step_from_name(name: Any) -> int:
    match = re.search(r"step_(\d+)$", str(name))
    return int(match.group(1)) if match else 0


def source_code_stem(row_or_dataset: Any, source_id: Any = None) -> str:
    if isinstance(row_or_dataset, dict):
        source_id = row_or_dataset.get("source_id")
    elif source_id is None:
        source_id = row_or_dataset
    return safe_path_part(source_id)


def dataset_source_code_stem(row: Dict[str, Any]) -> str:
    return f"{dataset_stem(row.get('dataset'))}_{source_code_stem(row)}"


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def normalize_problem_type(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in PROBLEM_TYPES else "special"


def compact_text(text: Any, max_chars: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 3].rstrip() + "..."


def zero_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def usage_add(total: Dict[str, int], usage: Optional[Dict[str, Any]]) -> None:
    usage = usage or {}
    for key in total:
        total[key] += safe_int(usage.get(key), 0)


def extract_prompt_var(text: str, var_name: str) -> str:
    match = re.search(rf"{re.escape(var_name)}\s*=\s*\"\"\"(.*?)\"\"\"", text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot find prompt variable: {var_name}")
    return match.group(1)


def load_prompt_bundle(prompt_file: str) -> Dict[str, str]:
    content = Path(prompt_file).read_text(encoding="utf-8")
    return {key: extract_prompt_var(content, key) for key in PROMPT_KEYS}


def fill_template(template: str, mapping: Dict[str, str]) -> str:
    for key, value in mapping.items():
        template = template.replace("{" + key + "}", value)
    return template


def strip_reasoning_tags(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"<think_never_used_[^>]+>.*?</think_never_used_[^>]+>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    return raw.strip()


def extract_balanced_json_block(text: str, start_at: int = 0) -> str:
    start = -1
    opening = ""
    for idx in range(max(0, start_at), len(text)):
        ch = text[idx]
        if ch in "{[":
            start = idx
            opening = ch
            break
    if start < 0:
        return ""

    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return ""


def parse_json_from_text(text: str) -> Any:
    raw = strip_reasoning_tags(text).strip()
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    balanced = extract_balanced_json_block(raw)
    if balanced:
        candidates.append(balanced)

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            python_candidate = re.sub(r"\bnull\b", "None", candidate)
            python_candidate = re.sub(r"\btrue\b", "True", python_candidate, flags=re.IGNORECASE)
            python_candidate = re.sub(r"\bfalse\b", "False", python_candidate, flags=re.IGNORECASE)
            parsed = ast.literal_eval(python_candidate)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass
    raise ValueError(f"Cannot parse JSON from output: {raw[:300]}")


def sanitize_python_snippet(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = strip_reasoning_tags(raw)
    raw = re.sub(r"^\s*<python>\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*</python>\s*$", "", raw, flags=re.IGNORECASE)

    previous = None
    while raw != previous:
        previous = raw
        fenced = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1).strip()
            continue
        generic = re.fullmatch(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
        if generic:
            raw = generic.group(1).strip()
            continue

    raw = re.sub(r"^\s*(?:```|'''|\"\"\")\s*python\s*\n", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^\s*(?:```|'''|\"\"\")\s*\n", "", raw)
    raw = re.sub(r"\n\s*(?:```|'''|\"\"\")\s*$", "", raw)
    raw = re.sub(r"^\s*python\s*\n", "", raw, flags=re.IGNORECASE)
    return raw.strip()


def extract_python_block(text: str) -> str:
    raw = strip_reasoning_tags(str(text or ""))
    tagged = re.search(r"<python>\s*(.*?)\s*</python>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        return sanitize_python_snippet(tagged.group(1))
    fenced = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return sanitize_python_snippet(fenced.group(1))
    generic = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if generic:
        return sanitize_python_snippet(generic.group(1))
    return ""


def call_model(
    system_prompt: str,
    user_prompt: str,
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    base_url = url.rstrip("/")
    request_url = base_url if base_url.endswith("/v1/chat/completions") else base_url + "/v1/chat/completions"
    payload = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    last_error = None
    for _ in range(3):
        try:
            resp = requests.post(request_url, headers=headers, data=json.dumps(payload), timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            return strip_reasoning_tags(data["choices"][0]["message"]["content"]), data.get("usage", {})
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"call_model failed: {last_error}")


def parse_problem_type_label(text: str) -> str:
    raw = strip_reasoning_tags(text).lower()
    lines = [line.strip("`* \t") for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        if line in PROBLEM_TYPES:
            return line
    pattern = r"\b(" + "|".join(re.escape(key) for key in PROBLEM_TYPES) + r")\b"
    matches = re.findall(pattern, raw)
    return matches[-1] if matches else "special"


def parse_labeled_index_list(text: str, label: str) -> Optional[List[str]]:
    raw = strip_reasoning_tags(text)
    pattern = rf"(?im)^\s*{re.escape(label)}\s*=\s*(.+?)\s*$"
    matches = re.findall(pattern, raw)
    if not matches:
        return None
    rhs = matches[-1].strip()
    if rhs.lower() == "none":
        return None
    if re.fullmatch(r"\[\s*[A-Za-z0-9_\-,\s]*\]", rhs):
        inner = rhs[1:-1].strip()
        if not inner:
            return []
        return [item.strip() for item in inner.split(",") if item.strip()]
    if re.fullmatch(r"[A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*", rhs):
        return [item.strip() for item in rhs.split(",") if item.strip()]
    return None


def parse_skill_retrieval_output(text: str) -> Tuple[List[str], List[str]]:
    strategies = parse_labeled_index_list(text, "Strategy") or []
    experiences = parse_labeled_index_list(text, "Experience") or []
    return strategies[:1], experiences[:2]


def parse_label(text: str, label: str, default: str = "") -> str:
    match = re.search(rf"(?im)^\s*{re.escape(label)}\s*=\s*(.*?)\s*$", str(text or ""))
    return match.group(1).strip() if match else default


def default_skill_metadata(created_step: int = 0) -> Dict[str, Any]:
    return {
        "use_count": 0,
        "positive_credit": 0,
        "negative_credit": 0,
        "score": 0.0,
        "reliability": "中",
        "created_step": created_step,
        "replaced_step": 0,
        "version": 0,
    }


def update_reliability(item: Dict[str, Any], alpha: float) -> None:
    pos = safe_int(item.get("positive_credit"), 0)
    neg = safe_int(item.get("negative_credit"), 0)
    use_count = pos + neg
    score = pos - alpha * neg
    item["use_count"] = use_count
    item["score"] = round(score, 4)
    if use_count < 3:
        item["reliability"] = "中"
        return
    neg_rate = neg / use_count if use_count else 0.0
    if score >= 5 and neg_rate <= 0.2:
        item["reliability"] = "高"
    elif score >= 0 and neg_rate <= 0.4:
        item["reliability"] = "中"
    elif score < 0 and neg_rate <= 0.6:
        item["reliability"] = "低"
    else:
        item["reliability"] = "风险"


def ensure_skill_metadata(item: Dict[str, Any], created_step: int = 0, alpha: float = 1.5) -> Dict[str, Any]:
    payload = dict(item)
    for key, value in default_skill_metadata(created_step).items():
        payload.setdefault(key, value)
    update_reliability(payload, alpha)
    return payload


def normalize_skillbank(raw: Dict[str, Any], alpha: float) -> Dict[str, Any]:
    if "strategies" in raw or "experiences" in raw:
        grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            ptype: {"strategies": [], "experiences": []} for ptype in PROBLEM_TYPES
        }
        for item in raw.get("strategies", []):
            ptype = normalize_problem_type(item.get("problem_type"))
            grouped[ptype]["strategies"].append(ensure_skill_metadata(item, alpha=alpha))
        for item in raw.get("experiences", []):
            ptype = normalize_problem_type(item.get("problem_type"))
            grouped[ptype]["experiences"].append(ensure_skill_metadata(item, alpha=alpha))
        return grouped

    grouped = {ptype: {"strategies": [], "experiences": []} for ptype in PROBLEM_TYPES}
    for ptype in PROBLEM_TYPES:
        bucket = raw.get(ptype, {}) or {}
        for item in bucket.get("strategies", []):
            payload = dict(item)
            payload.setdefault("problem_type", ptype)
            grouped[ptype]["strategies"].append(ensure_skill_metadata(payload, alpha=alpha))
        for item in bucket.get("experiences", []):
            payload = dict(item)
            payload.setdefault("problem_type", ptype)
            grouped[ptype]["experiences"].append(ensure_skill_metadata(payload, alpha=alpha))
    return grouped


def build_skillbank_index(skillbank: Dict[str, Any]) -> Dict[str, Any]:
    strategies_by_type = {ptype: [] for ptype in PROBLEM_TYPES}
    experiences_by_type = {ptype: [] for ptype in PROBLEM_TYPES}
    skills_by_index: Dict[str, Dict[str, Any]] = {}
    for ptype in PROBLEM_TYPES:
        bucket = skillbank.get(ptype, {}) or {}
        for item in bucket.get("strategies", []):
            payload = dict(item, problem_type=ptype, skill_kind="strategy")
            strategies_by_type[ptype].append(payload)
            if payload.get("index") is not None:
                skills_by_index[str(payload["index"])] = payload
        for item in bucket.get("experiences", []):
            payload = dict(item, problem_type=ptype, skill_kind="experience")
            experiences_by_type[ptype].append(payload)
            if payload.get("index") is not None:
                skills_by_index[str(payload["index"])] = payload
    return {
        "strategies_by_type": strategies_by_type,
        "experiences_by_type": experiences_by_type,
        "skills_by_index": skills_by_index,
    }


def format_strategy_candidates(items: List[Dict[str, Any]]) -> str:
    payload = []
    for item in items:
        payload.append(
            {
                "index": item.get("index"),
                "summary": compact_text(item.get("summary", ""), 700),
                "reliability": item.get("reliability", "中"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_experience_candidates(items: List[Dict[str, Any]]) -> str:
    payload = []
    for item in items:
        payload.append(
            {
                "index": item.get("index"),
                "trigger": compact_text(item.get("trigger", ""), 300),
                "reliability": item.get("reliability", "中"),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_selected_strategy(item: Optional[Dict[str, Any]]) -> str:
    if not item:
        return "None"
    return json.dumps(
        {
            "index": item.get("index"),
            "summary": item.get("summary"),
            "procedure": item.get("procedure", []),
            "reliability": item.get("reliability", "中"),
        },
        ensure_ascii=False,
        indent=2,
    )


def format_selected_experiences(items: List[Dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "index": item.get("index"),
                "trigger": item.get("trigger"),
                "guidance": item.get("guidance"),
                "reliability": item.get("reliability", "中"),
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=2,
    )


def classify_problem_type(question: str, prompts: Dict[str, str], url: str, model_name: str, max_tokens: int) -> Tuple[str, str, Dict[str, Any]]:
    content, usage = call_model(
        prompts["problem_type_classification_system_prompt"],
        fill_template(prompts["problem_type_classification_user_prompt"], {"question": question}),
        url,
        model_name,
        0.0,
        max_tokens,
    )
    return normalize_problem_type(parse_problem_type_label(content)), content, usage


def retrieve_skills(
    question: str,
    problem_type: str,
    prompts: Dict[str, str],
    skillbank_index: Dict[str, Any],
    url: str,
    model_name: str,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    strategy_candidates = skillbank_index["strategies_by_type"].get(problem_type, [])
    experience_candidates = skillbank_index["experiences_by_type"].get(problem_type, [])
    raw, usage = call_model(
        prompts["skill_retrieval_system_prompt"],
        fill_template(
            prompts["skill_retrieval_user_prompt"],
            {
                "question": question,
                "candidate_strategies": format_strategy_candidates(strategy_candidates),
                "candidate_experiences": format_experience_candidates(experience_candidates),
            },
        ),
        url,
        model_name,
        0.0,
        max_tokens,
    )
    error = None
    strategy_indexes, exp_indexes = parse_skill_retrieval_output(raw)
    strategy_by_index = {str(item.get("index")): item for item in strategy_candidates if item.get("index") is not None}
    exp_by_index = {str(item.get("index")): item for item in experience_candidates if item.get("index") is not None}
    selected_strategy = None
    selected_exps: List[Dict[str, Any]] = []
    if strategy_indexes:
        selected_strategy = strategy_by_index.get(strategy_indexes[0])
        if selected_strategy is None:
            error = f"Invalid strategy index: {strategy_indexes[0]}"
    for exp_index in exp_indexes:
        item = exp_by_index.get(exp_index)
        if item is None:
            error = f"Invalid experience index: {exp_index}"
            continue
        if all(str(existing.get("index")) != exp_index for existing in selected_exps):
            selected_exps.append(item)
    return selected_strategy, selected_exps, raw, usage, error


def generate_base_solution(question: str, prompts: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    return call_model(
        prompts["base_generation_system_prompt"],
        fill_template(prompts["base_generation_user_prompt"], {"question": question}),
        url,
        model_name,
        temperature,
        max_tokens,
    )


def generate_skillbank_solution(
    question: str,
    selected_strategy: Optional[Dict[str, Any]],
    selected_experiences: List[Dict[str, Any]],
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    return call_model(
        prompts["skill_augmented_generation_system_prompt"],
        fill_template(
            prompts["skill_augmented_generation_user_prompt"],
            {
                "question": question,
                "selected_strategy": format_selected_strategy(selected_strategy),
                "selected_experiences": format_selected_experiences(selected_experiences),
            },
        ),
        url,
        model_name,
        temperature,
        max_tokens,
    )


def run_self_refine(
    question: str,
    source_code: str,
    execution_feedback: str,
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    return call_model(
        prompts["self_refine_system_prompt"],
        fill_template(
            prompts["self_refine_user_prompt"],
            {
                "question": question,
                "source_code": source_code or "(no code extracted)",
                "execution_feedback": execution_feedback,
            },
        ),
        url,
        model_name,
        temperature,
        max_tokens,
    )


def ensure_strategy_schema(obj: Any, problem_type: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Strategy candidate must be an object.")
    procedure = obj.get("procedure", [])
    if not isinstance(procedure, list):
        procedure = []
    return {
        "problem_type": normalize_problem_type(obj.get("problem_type") or problem_type),
        "summary": str(obj.get("summary", "")).strip(),
        "procedure": [str(step).strip() for step in procedure if str(step).strip()],
    }


def ensure_experience_schema(obj: Any, problem_type: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("Experience candidate must be an object.")
    return {
        "problem_type": normalize_problem_type(obj.get("problem_type") or problem_type),
        "trigger": str(obj.get("trigger", "")).strip(),
        "guidance": str(obj.get("guidance", "")).strip(),
    }


def distill_new_strategy(
    question: str,
    correct_trajectory: str,
    problem_type: str,
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    raw, usage = call_model(
        prompts["new_strategy_system_prompt"],
        fill_template(prompts["new_strategy_user_prompt"], {"question": question, "correct_trajectory": correct_trajectory}),
        url,
        model_name,
        temperature,
        max_tokens,
    )
    try:
        return ensure_strategy_schema(parse_json_from_text(raw), problem_type), raw, usage, None
    except Exception as exc:
        return None, raw, usage, str(exc)


def distill_new_experience(
    question: str,
    incorrect_code: str,
    correct_trajectory: str,
    problem_type: str,
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    raw, usage = call_model(
        prompts["new_experience_system_prompt"],
        fill_template(
            prompts["new_experience_user_prompt"],
            {
                "question": question,
                "incorrect_code": incorrect_code or "(no code extracted)",
                "correct_trajectory": correct_trajectory,
            },
        ),
        url,
        model_name,
        temperature,
        max_tokens,
    )
    try:
        return ensure_experience_schema(parse_json_from_text(raw), problem_type), raw, usage, None
    except Exception as exc:
        return None, raw, usage, str(exc)


def parse_repair_output(raw: str, problem_type: str) -> Dict[str, Any]:
    owner = parse_label(raw, "FAILURE_OWNER", "unresolved").strip()
    allowed_owners = {
        "experience_misleading",
        "strategy_misleading",
        "experience_miss",
        "strategy_miss",
        "code_error_only",
        "unresolved",
    }
    if owner not in allowed_owners:
        owner = "unresolved"
    target_index = parse_label(raw, "TARGET_INDEX", "None").strip()
    if target_index.lower() == "none":
        target_index = ""
    repaired = None
    new_skill = None
    marker = re.search(r"REPAIRED_SKILL_JSON\s*=", raw, flags=re.IGNORECASE)
    if marker and owner in {"experience_misleading", "strategy_misleading"}:
        block = extract_balanced_json_block(raw, marker.end())
        if block:
            parsed = parse_json_from_text(block)
            repaired = (
                ensure_experience_schema(parsed, problem_type)
                if owner == "experience_misleading"
                else ensure_strategy_schema(parsed, problem_type)
            )
    marker = re.search(r"NEW_SKILL_JSON\s*=", raw, flags=re.IGNORECASE)
    if marker and owner in {"experience_miss", "strategy_miss"}:
        block = extract_balanced_json_block(raw, marker.end())
        if block:
            parsed = parse_json_from_text(block)
            new_skill = (
                ensure_experience_schema(parsed, problem_type)
                if owner == "experience_miss"
                else ensure_strategy_schema(parsed, problem_type)
            )
    return {"failure_owner": owner, "target_index": target_index, "repaired_skill": repaired, "new_skill": new_skill}


def judge_and_repair_skill(
    question: str,
    selected_strategy: Optional[Dict[str, Any]],
    selected_experiences: List[Dict[str, Any]],
    source_code: str,
    debug_info: str,
    problem_type: str,
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Optional[str]]:
    raw, usage = call_model(
        prompts["repair_skill_system_prompt"],
        fill_template(
            prompts["repair_skill_user_prompt"],
            {
                "question": question,
                "selected_strategy": format_selected_strategy(selected_strategy),
                "selected_experiences": format_selected_experiences(selected_experiences),
                "source_code": source_code or "(no code extracted)",
                "debug_info": debug_info,
            },
        ),
        url,
        model_name,
        temperature,
        max_tokens,
    )
    try:
        return parse_repair_output(raw, problem_type), raw, usage, None
    except Exception as exc:
        return {"failure_owner": "unresolved", "target_index": "", "repaired_skill": None}, raw, usage, str(exc)


def _inject_main_block_result_print(code: str) -> Tuple[str, bool]:
    lines = code.splitlines()
    for idx, line in enumerate(lines):
        match = re.match(r"^([ \t]*)if __name__\s*==\s*['\"]__main__['\"]\s*:\s*$", line)
        if not match:
            continue
        base_indent = match.group(1)
        block_indent = base_indent + "    "
        result_name = None
        insert_at = None
        j = idx + 1
        while j < len(lines):
            current = lines[j]
            if current.strip():
                current_indent = current[: len(current) - len(current.lstrip())]
                if len(current_indent.expandtabs(4)) <= len(base_indent.expandtabs(4)):
                    break
                assign = re.match(rf"^{re.escape(block_indent)}([A-Za-z_]\w*)\s*=\s*[\w\.]+\s*\(.*\)\s*$", current)
                if assign:
                    result_name = assign.group(1)
                    insert_at = j + 1
            j += 1
        if result_name and insert_at is not None:
            probe_lines = [
                f"{block_indent}if {result_name} is not None:",
                f'{block_indent}    print(f"{RESULT_PREFIX} {{{result_name}}}")',
                f"{block_indent}else:",
                f'{block_indent}    print("{NO_SOLUTION_TEXT}")',
            ]
            lines[insert_at:insert_at] = probe_lines
            return "\n".join(lines), True
    return code, False


def prepare_script_for_execution(code: str) -> str:
    cleaned = sanitize_python_snippet(code) or str(code or "")
    if RESULT_PREFIX in cleaned or NO_SOLUTION_TEXT in cleaned:
        return cleaned
    injected, ok = _inject_main_block_result_print(cleaned)
    if ok:
        return injected.rstrip() + "\n"
    return injected.rstrip() + "\n\n" + ADD_SCRIPT + "\n"


def extract_obj(log: str) -> Optional[float]:
    if RESULT_PREFIX not in str(log or ""):
        return None
    for line in str(log).splitlines():
        if RESULT_PREFIX in line:
            result = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if result:
                return float(result[-1])
    return None


def execute_script(code: str, output_dir: str, example_id: str, timeout: int) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, f"{example_id}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(prepare_script_for_execution(code))
    try:
        proc = subprocess.run(
            ["python3", script_path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout or ""
        best = extract_obj(stdout)
        if best is not None:
            state = "Execution Successful and Best Solution Found"
        elif NO_SOLUTION_TEXT in stdout:
            best = NO_SOLUTION_TEXT
            state = "Execution Successful but No Best Solution Found"
        elif proc.returncode != 0:
            best = None
            state = f"Execution Failed: {proc.stderr}"
        else:
            best = None
            state = "Execution Successful but Out of Expectation"
        return {"execution_state": state, "execution_best_solution": best, "stdout": stdout}
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        best = extract_obj(stdout)
        if best is not None:
            state = "Execution Timed Out after Printing Objective"
        elif NO_SOLUTION_TEXT in stdout:
            best = NO_SOLUTION_TEXT
            state = "Execution Timed Out after Printing No Best Solution"
        else:
            best = None
            state = "Execution Failed: Timeout"
        return {"execution_state": state, "execution_best_solution": best, "stdout": stdout}


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        return _to_float(value[0])
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "no best solution":
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def evaluate_python_code(
    code: str,
    output_dir: str,
    example_id: str,
    ground_truth: Any,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
) -> Dict[str, Any]:
    if not code:
        return {"is_correct": False, "execution_state": "Execution Failed: No code", "prediction": None, "stdout": ""}
    exec_out = execute_script(code, output_dir, example_id, timeout)
    pred = exec_out["execution_best_solution"]
    gt = _to_float(ground_truth)
    ok = False
    if isinstance(ground_truth, str) and ground_truth.strip().lower() == NO_SOLUTION_TEXT.lower():
        ok = pred == NO_SOLUTION_TEXT
    elif gt is not None and isinstance(pred, (int, float)):
        if gt == 0:
            ok = abs(pred) <= err_tolerance
        elif use_percentage_err_tolerance:
            ok = abs((pred - gt) / gt) <= err_tolerance
        else:
            ok = abs(pred - gt) <= err_tolerance
    return {
        "is_correct": ok,
        "execution_state": exec_out["execution_state"],
        "prediction": pred,
        "stdout": exec_out["stdout"],
    }


def build_feedback(eval_result: Dict[str, Any], ground_truth: Any) -> str:
    return (
        f"Execution state: {eval_result.get('execution_state')}\n"
        f"Prediction: {eval_result.get('prediction')}\n"
        f"Ground truth: {ground_truth}\n"
        f"Stdout:\n{compact_text(eval_result.get('stdout', ''), 1200)}"
    )


def run_self_refine_until_success(
    question: str,
    initial_code: str,
    initial_eval: Dict[str, Any],
    ground_truth: Any,
    record: Dict[str, Any],
    window_step: int,
    prompts: Dict[str, str],
    output_dir: str,
    url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
    max_attempts: int = SELF_REFINE_MAX_ATTEMPTS,
) -> Dict[str, Any]:
    attempts = []
    total_usage = zero_usage()

    for attempt_id in range(1, max_attempts + 1):
        refine_text, usage = run_self_refine(
            question,
            initial_code,
            build_feedback(initial_eval, ground_truth),
            prompts,
            url,
            model_name,
            temperature,
            max_tokens,
        )
        usage_add(total_usage, usage)
        refine_code = extract_python_block(refine_text)
        refine_eval = evaluate_python_code(
            refine_code,
            os.path.join(output_dir, "evolution", step_dir_name(window_step), "refine", "code"),
            f"{dataset_source_code_stem(record)}_{attempt_id}",
            ground_truth,
            timeout,
            err_tolerance,
            use_percentage_err_tolerance,
        )
        attempts.append(
            {
                "attempt": attempt_id,
                "raw": refine_text,
                "code": refine_code,
                "is_correct": refine_eval["is_correct"],
                "execution_state": refine_eval["execution_state"],
                "prediction": refine_eval["prediction"],
                "stdout": refine_eval.get("stdout", ""),
                "usage": usage,
            }
        )
        if refine_eval["is_correct"]:
            break

    final_attempt = attempts[-1] if attempts else {
        "attempt": 0,
        "raw": "",
        "code": "",
        "is_correct": False,
        "execution_state": "Execution Failed: No refine attempt",
        "prediction": None,
        "stdout": "",
        "usage": zero_usage(),
    }
    return {
        "raw": final_attempt.get("raw", ""),
        "code": final_attempt.get("code", ""),
        "is_correct": bool(final_attempt.get("is_correct")),
        "execution_state": final_attempt.get("execution_state"),
        "prediction": final_attempt.get("prediction"),
        "stdout": final_attempt.get("stdout", ""),
        "usage": total_usage,
        "attempt_count": len(attempts),
        "success_attempt": final_attempt.get("attempt") if final_attempt.get("is_correct") else None,
        "attempts": attempts,
    }


def build_item_id(item: Dict[str, Any], fallback_index: int) -> Any:
    for key in ("index", "id"):
        if key in item:
            return item[key]
    return fallback_index


def discover_datasets(testset_dir: str, dataset_names: Optional[List[str]]) -> List[Path]:
    paths = sorted(Path(testset_dir).glob("*.jsonl"))
    if not dataset_names:
        return paths
    wanted = {name if name.endswith(".jsonl") else f"{name}.jsonl" for name in dataset_names if name.strip()}
    return [path for path in paths if path.name in wanted]


def build_mixed_tasks(testset_dir: str, dataset_names: Optional[List[str]], shuffle_seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(shuffle_seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for path in discover_datasets(testset_dir, dataset_names):
        rows = load_jsonl(str(path))
        bucket = [{"dataset": path.stem, "item_index": idx, "item": item} for idx, item in enumerate(rows)]
        rng.shuffle(bucket)
        buckets[path.stem] = bucket
    if not buckets:
        raise ValueError("No dataset files found.")
    names = sorted(buckets)
    mixed: List[Dict[str, Any]] = []
    while True:
        any_added = False
        rng.shuffle(names)
        for name in names:
            if buckets[name]:
                mixed.append(buckets[name].pop())
                any_added = True
        if not any_added:
            break
    return mixed


def make_case_key(dataset: str, source_id: Any) -> str:
    return f"{dataset}::{source_id}"


def build_processing_error_record(step: int, task: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
    item = task["item"]
    source_id = build_item_id(item, int(task["item_index"]))
    return {
        "step": step,
        "dataset": task["dataset"],
        "source_index": int(task["item_index"]),
        "source_id": source_id,
        "question": str(item.get("en_question") or item.get("question") or "").strip(),
        "en_answer": item.get("en_answer", item.get("answer")),
        "status": "error",
        "error_message": str(exc),
        "used_skillbank": False,
        "is_correct": False,
        "skill_events": [],
        "candidate_events": [],
        "usage": zero_usage(),
    }


def make_candidate_event(
    kind: str,
    source: str,
    skill: Dict[str, Any],
    step: int,
    dataset: str,
    source_id: Any,
    parent_index: str = "",
    source_detail: str = "",
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "source": source,
        "source_detail": source_detail,
        "parent_index": parent_index,
        "problem_type": normalize_problem_type(skill.get("problem_type")),
        "skill": skill,
        "support_case": {"step": step, "dataset": dataset, "source_id": source_id},
    }


def process_task(
    step: int,
    window_step: int,
    task: Dict[str, Any],
    prompts: Dict[str, str],
    skillbank_snapshot: Dict[str, Any],
    output_dir: str,
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    routing_max_tokens: int,
    distill_temperature: float,
    distill_max_tokens: int,
    repair_temperature: float,
    repair_max_tokens: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
) -> Dict[str, Any]:
    item = task["item"]
    dataset = task["dataset"]
    item_index = int(task["item_index"])
    source_id = build_item_id(item, item_index)
    question = str(item.get("en_question") or item.get("question") or "").strip()
    ground_truth = item.get("en_answer", item.get("answer"))
    if not question:
        raise ValueError("Missing question text.")

    record: Dict[str, Any] = {
        "step": step,
        "dataset": dataset,
        "source_index": item_index,
        "source_id": source_id,
        "question": question,
        "en_answer": ground_truth,
        "difficulty": item.get("difficulty", "unknown"),
        "status": "success",
        "usage": zero_usage(),
        "skill_events": [],
        "candidate_events": [],
        "postprocess_errors": [],
    }
    started = time.time()
    skillbank_index = build_skillbank_index(skillbank_snapshot)

    problem_type, classify_raw, usage = classify_problem_type(question, prompts, url, model_name, routing_max_tokens)
    usage_add(record["usage"], usage)

    selected_strategy, selected_exps, retrieval_raw, usage, retrieval_error = retrieve_skills(
        question,
        problem_type,
        prompts,
        skillbank_index,
        url,
        model_name,
        routing_max_tokens,
    )
    usage_add(record["usage"], usage)
    used_skillbank = selected_strategy is not None or bool(selected_exps)

    if used_skillbank:
        response_text, usage = generate_skillbank_solution(
            question,
            selected_strategy,
            selected_exps,
            prompts,
            url,
            model_name,
            generation_temperature,
            generation_max_tokens,
        )
    else:
        response_text, usage = generate_base_solution(
            question,
            prompts,
            url,
            model_name,
            generation_temperature,
            generation_max_tokens,
        )
    usage_add(record["usage"], usage)

    python_code = extract_python_block(response_text)
    eval_result = evaluate_python_code(
        python_code,
        os.path.join(output_dir, "execution", dataset, "code"),
        source_code_stem(source_id),
        ground_truth,
        timeout,
        err_tolerance,
        use_percentage_err_tolerance,
    )

    record.update(
        {
            "question_type": problem_type,
            "problem_type_classification_raw": classify_raw,
            "raw_retrieval": retrieval_raw,
            "retrieval_error": retrieval_error,
            "used_skillbank": used_skillbank,
            "selected_strategy_index": selected_strategy.get("index") if selected_strategy else None,
            "selected_experience_indexes": [exp.get("index") for exp in selected_exps],
            "selected_strategy_full": selected_strategy,
            "selected_experiences_full": selected_exps,
            "raw_generations": response_text,
            "en_gurobi_code": python_code,
            "is_correct": eval_result["is_correct"],
            "execution_state": eval_result["execution_state"],
            "prediction": eval_result["prediction"],
            "execution_stdout": eval_result["stdout"],
        }
    )

    if used_skillbank and eval_result["is_correct"]:
        if selected_strategy:
            record["skill_events"].append({"kind": "strategy", "index": selected_strategy.get("index"), "credit": "positive"})
        for exp in selected_exps:
            record["skill_events"].append({"kind": "experience", "index": exp.get("index"), "credit": "positive"})

    elif used_skillbank and not eval_result["is_correct"]:
        refine_summary = run_self_refine_until_success(
            question,
            python_code,
            eval_result,
            ground_truth,
            record,
            window_step,
            prompts,
            output_dir,
            url,
            model_name,
            generation_temperature,
            generation_max_tokens,
            timeout,
            err_tolerance,
            use_percentage_err_tolerance,
        )
        usage_add(record["usage"], refine_summary["usage"])
        record["self_refine"] = refine_summary
        if refine_summary["is_correct"]:
            repair_result, repair_raw, usage, repair_error = judge_and_repair_skill(
                question,
                selected_strategy,
                selected_exps,
                python_code,
                refine_summary["raw"],
                problem_type,
                prompts,
                url,
                model_name,
                repair_temperature,
                repair_max_tokens,
            )
            usage_add(record["usage"], usage)
            record["repair_judgment"] = {"raw": repair_raw, "parsed": repair_result, "error": repair_error, "usage": usage}
            owner = repair_result.get("failure_owner")
            target_index = str(repair_result.get("target_index") or "").strip()
            repaired_skill = repair_result.get("repaired_skill")
            new_skill = repair_result.get("new_skill")
            if owner == "experience_misleading" and target_index and repaired_skill:
                record["skill_events"].append({"kind": "experience", "index": target_index, "credit": "negative"})
                record["candidate_events"].append(
                    make_candidate_event("experience", "repair", repaired_skill, step, dataset, source_id, target_index, "misleading")
                )
            elif owner == "strategy_misleading" and target_index and repaired_skill:
                record["skill_events"].append({"kind": "strategy", "index": target_index, "credit": "negative"})
                record["candidate_events"].append(
                    make_candidate_event("strategy", "repair", repaired_skill, step, dataset, source_id, target_index, "misleading")
                )
            elif owner == "experience_miss" and new_skill:
                record["candidate_events"].append(
                    make_candidate_event("experience", "new", new_skill, step, dataset, source_id, source_detail="skillbank_miss")
                )
            elif owner == "strategy_miss" and new_skill:
                record["candidate_events"].append(
                    make_candidate_event("strategy", "new", new_skill, step, dataset, source_id, source_detail="skillbank_miss")
                )

    elif not used_skillbank and eval_result["is_correct"]:
        candidate, raw, usage, error = distill_new_strategy(
            question,
            response_text,
            problem_type,
            prompts,
            url,
            model_name,
            distill_temperature,
            distill_max_tokens,
        )
        usage_add(record["usage"], usage)
        record["new_strategy_distill"] = {"raw": raw, "error": error, "usage": usage}
        if candidate:
            record["candidate_events"].append(
                make_candidate_event("strategy", "new", candidate, step, dataset, source_id, source_detail="base_correct")
            )

    elif not used_skillbank and not eval_result["is_correct"]:
        refine_summary = run_self_refine_until_success(
            question,
            python_code,
            eval_result,
            ground_truth,
            record,
            window_step,
            prompts,
            output_dir,
            url,
            model_name,
            generation_temperature,
            generation_max_tokens,
            timeout,
            err_tolerance,
            use_percentage_err_tolerance,
        )
        usage_add(record["usage"], refine_summary["usage"])
        record["self_refine"] = refine_summary
        if refine_summary["is_correct"]:
            candidate, raw, usage, error = distill_new_experience(
                question,
                python_code,
                refine_summary["raw"],
                problem_type,
                prompts,
                url,
                model_name,
                distill_temperature,
                distill_max_tokens,
            )
            usage_add(record["usage"], usage)
            record["new_experience_distill"] = {"raw": raw, "error": error, "usage": usage}
            if candidate:
                record["candidate_events"].append(
                    make_candidate_event("experience", "new", candidate, step, dataset, source_id, source_detail="base_refine_correct")
                )

    record["latency_sec"] = round(time.time() - started, 3)
    return record


def candidate_id_number(candidate_id: str) -> int:
    match = re.search(r"(\d+)$", str(candidate_id))
    return int(match.group(1)) if match else 0


def iter_candidate_id_numbers(obj: Any) -> List[int]:
    found: List[int] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "candidate_id":
                num = candidate_id_number(str(value))
                if num:
                    found.append(num)
            else:
                found.extend(iter_candidate_id_numbers(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(iter_candidate_id_numbers(item))
    return found


def sync_candidate_id_counter(*objs: Any) -> None:
    global CANDIDATE_ID_COUNTER
    max_num = CANDIDATE_ID_COUNTER
    for obj in objs:
        nums = iter_candidate_id_numbers(obj)
        if nums:
            max_num = max(max_num, max(nums))
    CANDIDATE_ID_COUNTER = max_num


def sync_candidate_id_counter_from_output(output_dir: str, max_step: Optional[int] = None) -> None:
    for path in sorted(Path(output_dir, "evolution").glob("step_*/update/generation_records.jsonl")):
        step = checkpoint_step_from_name(path.parts[-3]) if len(path.parts) >= 3 else 0
        if max_step is not None and step > max_step:
            continue
        for row in load_jsonl(str(path)):
            sync_candidate_id_counter(row)


def next_candidate_id(candidates: List[Dict[str, Any]]) -> str:
    global CANDIDATE_ID_COUNTER
    sync_candidate_id_counter(candidates)
    CANDIDATE_ID_COUNTER += 1
    return f"cand_{CANDIDATE_ID_COUNTER:03d}"


def next_skill_index(skillbank: Dict[str, Any], problem_type: str, kind: str) -> str:
    prefix = TYPE_PREFIX.get(problem_type, problem_type)
    mid = "strategy" if kind == "strategy" else "exp"
    existing = []
    bucket = skillbank.get(problem_type, {}) or {}
    for item in bucket.get("strategies" if kind == "strategy" else "experiences", []):
        existing.append(str(item.get("index", "")))
    max_num = 0
    pattern = rf"^{re.escape(prefix)}_{mid}_(\d+)$"
    for value in existing:
        match = re.search(pattern, value)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"{prefix}_{mid}_{max_num + 1:03d}"


def format_active_skills_for_dedup(skillbank: Dict[str, Any], problem_type: str, kind: str) -> str:
    bucket = skillbank.get(problem_type, {}) or {}
    items = bucket.get("strategies" if kind == "strategy" else "experiences", [])
    payload = []
    for item in items:
        if kind == "strategy":
            payload.append({"index": item.get("index"), "summary": compact_text(item.get("summary", ""), 500)})
        else:
            payload.append(
                {
                    "index": item.get("index"),
                    "trigger": compact_text(item.get("trigger", ""), 250),
                    "guidance": compact_text(item.get("guidance", ""), 350),
                }
            )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_candidates_for_dedup(candidates: List[Dict[str, Any]], problem_type: str, kind: str, source: str, parent_index: str) -> str:
    payload = []
    for item in candidates:
        if item.get("problem_type") != problem_type or item.get("kind") != kind or item.get("source") != source:
            continue
        if source == "repair" and str(item.get("parent_index", "")) != parent_index:
            continue
        skill = item.get("skill", {})
        payload.append(
            {
                "candidate_id": item.get("candidate_id"),
                "support_count": item.get("support_count", 0),
                "parent_index": item.get("parent_index", ""),
                "skill": skill,
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_dedup_output(
    raw: str,
    problem_type: str,
    kind: str,
    allowed_decisions: Optional[set] = None,
) -> Dict[str, Any]:
    allowed = allowed_decisions or {"duplicate_active", "merge_candidate", "new"}
    decision = parse_label(raw, "DECISION", "new").strip()
    if decision not in allowed:
        decision = "new"
    match_id = parse_label(raw, "MATCH_ID", "None").strip()
    if match_id.lower() == "none":
        match_id = ""
    merged = None
    marker = re.search(r"MERGED_SKILL_JSON\s*=", raw, flags=re.IGNORECASE)
    if marker:
        block = extract_balanced_json_block(raw, marker.end())
        if block:
            parsed = parse_json_from_text(block)
            merged = ensure_strategy_schema(parsed, problem_type) if kind == "strategy" else ensure_experience_schema(parsed, problem_type)
    return {"decision": decision, "match_id": match_id, "merged_skill": merged}


def fallback_candidate_signature(event: Dict[str, Any]) -> str:
    skill = event.get("skill", {})
    if event.get("kind") == "strategy":
        base = skill.get("summary", "")
    else:
        base = f"{skill.get('trigger', '')} {skill.get('guidance', '')}"
    return re.sub(r"\s+", " ", str(base).lower()).strip()[:220]


def merge_candidate_into_pool(
    event: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    max_tokens: int,
    current_step: int,
) -> Dict[str, Any]:
    problem_type = normalize_problem_type(event.get("problem_type"))
    kind = str(event.get("kind"))
    source = str(event.get("source"))
    parent_index = str(event.get("parent_index") or "")
    candidate_skill = event.get("skill", {})
    candidate_skills = format_candidates_for_dedup(candidates, problem_type, kind, source, parent_index)
    raw = ""
    parsed: Dict[str, Any] = {"decision": "new", "match_id": "", "merged_skill": None}
    error = None
    try:
        raw, _ = call_model(
            prompts["candidate_dedup_system_prompt"],
            fill_template(
                prompts["candidate_dedup_user_prompt"],
                {
                    "candidate_skill": json.dumps(candidate_skill, ensure_ascii=False, indent=2),
                    "active_skills": "[]",
                    "candidate_skills": candidate_skills,
                },
            ),
            url,
            model_name,
            0.0,
            max_tokens,
        )
        parsed = parse_dedup_output(raw, problem_type, kind, {"merge_candidate", "new"})
    except Exception as exc:
        error = str(exc)
        signature = fallback_candidate_signature(event)
        for item in candidates:
            if (
                item.get("problem_type") == problem_type
                and item.get("kind") == kind
                and item.get("source") == source
                and str(item.get("parent_index", "")) == parent_index
                and item.get("fallback_signature") == signature
            ):
                parsed = {"decision": "merge_candidate", "match_id": item.get("candidate_id"), "merged_skill": None}
                break

    decision = parsed.get("decision")
    match_id = str(parsed.get("match_id") or "")
    if decision == "merge_candidate" and match_id:
        for item in candidates:
            if str(item.get("candidate_id")) == match_id:
                item["support_count"] = safe_int(item.get("support_count"), 0) + 1
                item.setdefault("support_cases", []).append(event.get("support_case"))
                item["last_updated_step"] = current_step
                if event.get("source_detail"):
                    details = set(item.get("source_details") or [])
                    details.add(event.get("source_detail"))
                    item["source_details"] = sorted(details)
                if parsed.get("merged_skill"):
                    item["skill"] = parsed["merged_skill"]
                return {"action": "merge_candidate", "candidate_id": match_id, "raw": raw, "error": error}

    candidate_id = next_candidate_id(candidates)
    candidates.append(
        {
            "candidate_id": candidate_id,
            "kind": kind,
            "source": source,
            "parent_index": parent_index,
            "problem_type": problem_type,
            "skill": candidate_skill,
            "support_count": 1,
            "support_cases": [event.get("support_case")],
            "source_details": [event.get("source_detail")] if event.get("source_detail") else [],
            "created_step": current_step,
            "last_updated_step": current_step,
            "fallback_signature": fallback_candidate_signature(event),
        }
    )
    return {"action": "new_candidate", "candidate_id": candidate_id, "raw": raw, "error": error}


def check_new_candidate_against_active(
    skillbank: Dict[str, Any],
    candidate: Dict[str, Any],
    prompts: Dict[str, str],
    url: str,
    model_name: str,
    max_tokens: int,
) -> Dict[str, Any]:
    problem_type = normalize_problem_type(candidate.get("problem_type"))
    kind = str(candidate.get("kind"))
    active_skills = format_active_skills_for_dedup(skillbank, problem_type, kind)
    raw = ""
    parsed: Dict[str, Any] = {"decision": "new", "match_id": "", "merged_skill": None}
    error = None
    try:
        raw, _ = call_model(
            prompts["candidate_dedup_system_prompt"],
            fill_template(
                prompts["candidate_dedup_user_prompt"],
                {
                    "candidate_skill": json.dumps(candidate.get("skill", {}), ensure_ascii=False, indent=2),
                    "active_skills": active_skills,
                    "candidate_skills": "[]",
                },
            ),
            url,
            model_name,
            0.0,
            max_tokens,
        )
        parsed = parse_dedup_output(raw, problem_type, kind, {"duplicate_active", "new"})
    except Exception as exc:
        error = str(exc)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "decision": parsed.get("decision", "new"),
        "match_id": parsed.get("match_id", ""),
        "raw": raw,
        "error": error,
    }


def apply_skill_events(skillbank: Dict[str, Any], events: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    updates = []
    index_map = build_skillbank_index(skillbank)["skills_by_index"]
    for event in events:
        skill_index = str(event.get("index") or "")
        credit = event.get("credit")
        item = index_map.get(skill_index)
        if item is None:
            continue
        # Mutate the item in the original grouped skillbank.
        ptype = normalize_problem_type(item.get("problem_type"))
        key = "strategies" if item.get("skill_kind") == "strategy" else "experiences"
        for target in skillbank.get(ptype, {}).get(key, []):
            if str(target.get("index")) == skill_index:
                if credit == "positive":
                    target["positive_credit"] = safe_int(target.get("positive_credit"), 0) + 1
                elif credit == "negative":
                    target["negative_credit"] = safe_int(target.get("negative_credit"), 0) + 1
                update_reliability(target, alpha)
                updates.append({"index": skill_index, "credit": credit, "score": target.get("score"), "reliability": target.get("reliability")})
                break
    return updates


def find_active_skill(skillbank: Dict[str, Any], index: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    for ptype in PROBLEM_TYPES:
        for key, kind in (("strategies", "strategy"), ("experiences", "experience")):
            for item in skillbank.get(ptype, {}).get(key, []):
                if str(item.get("index")) == str(index):
                    return ptype, kind, item
    return None


def replacement_skill_for_validation(candidate: Dict[str, Any], parent: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(candidate.get("skill", {}))
    payload["index"] = parent.get("index")
    payload["problem_type"] = parent.get("problem_type", candidate.get("problem_type"))
    payload["reliability"] = "中"
    return payload


def validation_rows_for_candidate(
    candidate: Dict[str, Any],
    records: List[Dict[str, Any]],
    max_failures: int,
    max_regressions: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    support_keys = {
        make_case_key(str(case.get("dataset")), case.get("source_id"))
        for case in candidate.get("support_cases", [])
        if case
    }
    failure_rows = [
        row
        for row in records
        if make_case_key(str(row.get("dataset")), row.get("source_id")) in support_keys
    ][:max_failures]
    parent_index = str(candidate.get("parent_index"))
    success_rows = []
    for row in reversed(records):
        if len(success_rows) >= max_regressions:
            break
        if not row.get("is_correct"):
            continue
        if row.get("selected_strategy_index") == parent_index or parent_index in (row.get("selected_experience_indexes") or []):
            if make_case_key(str(row.get("dataset")), row.get("source_id")) not in support_keys:
                success_rows.append(row)
    return failure_rows, success_rows


def run_repair_validation(
    candidate: Dict[str, Any],
    skillbank: Dict[str, Any],
    records: List[Dict[str, Any]],
    prompts: Dict[str, str],
    output_dir: str,
    current_step: int,
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
    max_failures: int,
    max_regressions: int,
) -> Dict[str, Any]:
    parent_found = find_active_skill(skillbank, str(candidate.get("parent_index")))
    if parent_found is None:
        return {"candidate_id": candidate.get("candidate_id"), "validated": False, "reason": "parent_not_found"}
    _, parent_kind, parent = parent_found
    replacement = replacement_skill_for_validation(candidate, parent)
    failure_rows, regression_rows = validation_rows_for_candidate(candidate, records, max_failures, max_regressions)
    if not failure_rows:
        return {"candidate_id": candidate.get("candidate_id"), "validated": False, "reason": "no_failure_rows"}

    def run_row(row: Dict[str, Any], tag: str) -> Dict[str, Any]:
        selected_strategy = row.get("selected_strategy_full")
        selected_exps = list(row.get("selected_experiences_full") or [])
        if parent_kind == "strategy":
            selected_strategy = replacement
        else:
            selected_exps = [replacement if str(exp.get("index")) == str(parent.get("index")) else exp for exp in selected_exps]
        response, _ = generate_skillbank_solution(
            row["question"],
            selected_strategy,
            selected_exps,
            prompts,
            url,
            model_name,
            generation_temperature,
            generation_max_tokens,
        )
        code = extract_python_block(response)
        eval_result = evaluate_python_code(
            code,
            os.path.join(
                output_dir,
                "evolution",
                step_dir_name(current_step),
                "update",
                "code",
                safe_path_part(candidate.get("candidate_id")),
            ),
            f"{tag}_{dataset_source_code_stem(row)}",
            row.get("en_answer"),
            timeout,
            err_tolerance,
            use_percentage_err_tolerance,
        )
        return {
            "step": row.get("step"),
            "source_id": row.get("source_id"),
            "is_correct": eval_result["is_correct"],
            "execution_state": eval_result["execution_state"],
            "prediction": eval_result["prediction"],
        }

    failure_results = [run_row(row, "fail") for row in failure_rows]
    regression_results = [run_row(row, "reg") for row in regression_rows]
    repair_rate = sum(1 for row in failure_results if row["is_correct"]) / len(failure_results)
    regression_rate = (
        sum(1 for row in regression_results if row["is_correct"]) / len(regression_results)
        if regression_results
        else 1.0
    )
    return {
        "candidate_id": candidate.get("candidate_id"),
        "validated": True,
        "repair_success_rate": repair_rate,
        "regression_success_rate": regression_rate,
        "failure_results": failure_results,
        "regression_results": regression_results,
    }


def promote_new_candidate(skillbank: Dict[str, Any], candidate: Dict[str, Any], current_step: int, alpha: float) -> Dict[str, Any]:
    ptype = normalize_problem_type(candidate.get("problem_type"))
    kind = candidate.get("kind")
    key = "strategies" if kind == "strategy" else "experiences"
    skill = dict(candidate.get("skill", {}))
    skill["problem_type"] = ptype
    skill["index"] = next_skill_index(skillbank, ptype, kind)
    skill.update(default_skill_metadata(current_step))
    update_reliability(skill, alpha)
    skillbank.setdefault(ptype, {"strategies": [], "experiences": []})
    skillbank[ptype][key].append(skill)
    return {"candidate_id": candidate.get("candidate_id"), "new_index": skill["index"], "kind": kind, "problem_type": ptype}


def replace_with_repair_candidate(skillbank: Dict[str, Any], candidate: Dict[str, Any], current_step: int, alpha: float) -> Dict[str, Any]:
    parent_index = str(candidate.get("parent_index"))
    found = find_active_skill(skillbank, parent_index)
    if found is None:
        return {"candidate_id": candidate.get("candidate_id"), "replaced": False, "reason": "parent_not_found"}
    ptype, kind, parent = found
    repaired = dict(candidate.get("skill", {}))
    repaired["index"] = parent.get("index")
    repaired["problem_type"] = ptype
    repaired.update(default_skill_metadata(current_step))
    repaired["version"] = safe_int(parent.get("version"), 0) + 1
    update_reliability(repaired, alpha)
    key = "strategies" if kind == "strategy" else "experiences"
    bucket = skillbank[ptype][key]
    for idx, item in enumerate(bucket):
        if str(item.get("index")) == parent_index:
            bucket[idx] = repaired
            return {
                "candidate_id": candidate.get("candidate_id"),
                "replaced": True,
                "parent_index": parent_index,
                "kind": kind,
                "version": repaired["version"],
            }
    return {"candidate_id": candidate.get("candidate_id"), "replaced": False, "reason": "parent_not_found"}


def update_candidates_and_skillbank(
    window_rows: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    skillbank: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    prompts: Dict[str, str],
    output_dir: str,
    current_step: int,
    alpha: float,
    support_threshold: int,
    repair_success_threshold: float,
    regression_success_threshold: float,
    repair_validation_failures: int,
    repair_validation_regressions: int,
    update_max_workers: int,
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    dedup_max_tokens: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
) -> Dict[str, Any]:
    skill_events = [event for row in window_rows for event in row.get("skill_events", [])]
    candidate_events = [event for row in window_rows for event in row.get("candidate_events", [])]
    skill_updates = apply_skill_events(skillbank, skill_events, alpha)

    candidate_updates = []
    for event in candidate_events:
        try:
            update = merge_candidate_into_pool(
                event,
                candidates,
                prompts,
                url,
                model_name,
                dedup_max_tokens,
                current_step,
            )
            update["event"] = event
            candidate_updates.append(update)
        except Exception as exc:
            candidate_updates.append({"action": "candidate_merge_error", "error": str(exc), "event": event})

    active_dedup_checks = []
    duplicate_new_skills = []
    promoted = []
    remaining_candidates = []
    for candidate in candidates:
        if candidate.get("source") == "new" and safe_int(candidate.get("support_count"), 0) >= support_threshold:
            active_check = check_new_candidate_against_active(
                skillbank,
                candidate,
                prompts,
                url,
                model_name,
                dedup_max_tokens,
            )
            active_dedup_checks.append(active_check)
            if active_check.get("decision") == "duplicate_active":
                duplicate_new_skills.append(active_check)
            else:
                promoted.append(promote_new_candidate(skillbank, candidate, current_step, alpha))
        else:
            remaining_candidates.append(candidate)
    candidates[:] = remaining_candidates

    repair_candidates = [candidate for candidate in candidates if candidate.get("source") == "repair"]
    validation_results = []
    if repair_candidates:
        with ThreadPoolExecutor(max_workers=max(1, update_max_workers)) as executor:
            future_to_candidate = {
                executor.submit(
                    run_repair_validation,
                    candidate,
                    skillbank,
                    all_records,
                    prompts,
                    output_dir,
                    current_step,
                    url,
                    model_name,
                    generation_temperature,
                    generation_max_tokens,
                    timeout,
                    err_tolerance,
                    use_percentage_err_tolerance,
                    repair_validation_failures,
                    repair_validation_regressions,
                ): candidate
                for candidate in repair_candidates
            }
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"candidate_id": candidate.get("candidate_id"), "validated": False, "reason": str(exc)}
                validation_results.append(result)

    passed_ids = {
        str(result.get("candidate_id"))
        for result in validation_results
        if result.get("validated")
        and safe_float(result.get("repair_success_rate")) >= repair_success_threshold
        and safe_float(result.get("regression_success_rate")) >= regression_success_threshold
    }
    replacements = []
    remaining_candidates = []
    for candidate in candidates:
        if str(candidate.get("candidate_id")) in passed_ids:
            replacements.append(replace_with_repair_candidate(skillbank, candidate, current_step, alpha))
        else:
            remaining_candidates.append(candidate)
    candidates[:] = remaining_candidates

    return {
        "step": current_step,
        "window_start_step": window_rows[0]["step"] if window_rows else current_step,
        "window_end_step": window_rows[-1]["step"] if window_rows else current_step,
        "skill_updates": skill_updates,
        "candidate_updates": candidate_updates,
        "active_dedup_checks": active_dedup_checks,
        "duplicate_new_skills": duplicate_new_skills,
        "promoted_new_skills": promoted,
        "repair_validations": sorted(validation_results, key=lambda item: str(item.get("candidate_id"))),
        "replacements": replacements,
        "candidate_count": len(candidates),
    }


def save_checkpoint(skillbank: Dict[str, Any], candidates: List[Dict[str, Any]], output_dir: str, step: int) -> None:
    skillbank_dir = os.path.join(output_dir, "skillbank")
    write_json(os.path.join(skillbank_dir, "skillbank.json"), skillbank)
    write_json(os.path.join(skillbank_dir, "candidates.json"), {"candidates": candidates})
    if safe_int(step, 0) > 0:
        checkpoint_dir = os.path.join(skillbank_dir, "checkpoints", step_dir_name(step))
        write_json(os.path.join(checkpoint_dir, "skillbank.json"), skillbank)
        write_json(os.path.join(checkpoint_dir, "candidates.json"), {"candidates": candidates})


def ordered_generation_record(row: Dict[str, Any]) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for key in ("dataset", "source_index", "source_id", "step"):
        if key in row:
            ordered[key] = row.get(key)
    for key, value in row.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def persist_generation_records(output_dir: str, records: List[Dict[str, Any]]) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        groups[str(row.get("dataset", "unknown"))].append(row)
    generation_dir = os.path.join(output_dir, "generation")
    for dataset, rows in groups.items():
        rows = sorted(rows, key=lambda item: (safe_int(item.get("source_index"), 0), safe_int(item.get("step"), 0)))
        write_jsonl(os.path.join(generation_dir, f"{safe_path_part(dataset)}.jsonl"), [ordered_generation_record(row) for row in rows])


def execution_report_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda item: (safe_int(item.get("source_index"), 0), safe_int(item.get("step"), 0)))
    total = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct"))
    difficulty_counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {"correct": 0, "total": 0})
    execution_state_counts: Dict[str, int] = collections.Counter(str(row.get("execution_state", "unknown")) for row in rows)
    results = []
    for row in rows:
        difficulty = str(row.get("difficulty", "unknown"))
        difficulty_counts[difficulty]["total"] += 1
        if row.get("is_correct"):
            difficulty_counts[difficulty]["correct"] += 1
        results.append(
            {
                "id": row.get("source_id"),
                "source_index": row.get("source_index"),
                "step": row.get("step"),
                "difficulty": difficulty,
                "pass_1": bool(row.get("is_correct")),
                "results_correct": [bool(row.get("is_correct"))],
                "execution_details": [
                    {
                        "state": row.get("execution_state"),
                        "best_solution": row.get("prediction"),
                    }
                ],
            }
        )
    report: Dict[str, Any] = {
        "accuracy": correct / total if total else 0.0,
        "correct_count": correct,
        "total_count": total,
        "execution_state_counts": dict(execution_state_counts),
        "results": results,
    }
    for difficulty in ("easy", "medium", "hard"):
        count = difficulty_counts[difficulty]
        report[f"{difficulty}_accuracy"] = count["correct"] / count["total"] if count["total"] else 0.0
        report[f"{difficulty}_correct_count"] = count["correct"]
        report[f"{difficulty}_total_count"] = count["total"]
    return report


def persist_execution_reports(output_dir: str, records: List[Dict[str, Any]]) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        groups[str(row.get("dataset", "unknown"))].append(row)
    for dataset, rows in groups.items():
        report_path = os.path.join(output_dir, "execution", safe_path_part(dataset), "evaluation_report.json")
        write_json(report_path, execution_report_for_rows(rows))


def build_refine_generation_records(window_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for row in sorted(window_rows, key=lambda item: safe_int(item.get("step"), 0)):
        refine = row.get("self_refine")
        if not isinstance(refine, dict):
            continue
        attempts = refine.get("attempts")
        if not isinstance(attempts, list):
            attempts = [refine]
        for idx, attempt in enumerate(attempts, start=1):
            attempt_id = safe_int(attempt.get("attempt"), idx)
            records.append(
                {
                    "record_type": "self_refine",
                    "dataset": row.get("dataset"),
                    "source_index": row.get("source_index"),
                    "source_id": row.get("source_id"),
                    "step": row.get("step"),
                    "attempt": attempt_id,
                    "used_skillbank": row.get("used_skillbank"),
                    "question": row.get("question"),
                    "en_answer": row.get("en_answer"),
                    "source_code": row.get("en_gurobi_code"),
                    "raw_generation": attempt.get("raw"),
                    "code": attempt.get("code"),
                    "is_correct": attempt.get("is_correct"),
                    "execution_state": attempt.get("execution_state"),
                    "prediction": attempt.get("prediction"),
                    "usage": attempt.get("usage", zero_usage()),
                }
            )
    return records


def build_update_generation_records(window_rows: List[Dict[str, Any]], window_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in sorted(window_rows, key=lambda item: safe_int(item.get("step"), 0)):
        common = {
            "dataset": row.get("dataset"),
            "source_index": row.get("source_index"),
            "source_id": row.get("source_id"),
            "step": row.get("step"),
            "question": row.get("question"),
            "en_answer": row.get("en_answer"),
            "used_skillbank": row.get("used_skillbank"),
        }
        if isinstance(row.get("new_strategy_distill"), dict):
            payload = row["new_strategy_distill"]
            records.append(
                {
                    "record_type": "new_strategy_candidate_generation",
                    **common,
                    "raw_generation": payload.get("raw"),
                    "error": payload.get("error"),
                    "usage": payload.get("usage", zero_usage()),
                    "candidate_events": [event for event in row.get("candidate_events", []) if event.get("source") == "new" and event.get("kind") == "strategy"],
                }
            )
        if isinstance(row.get("new_experience_distill"), dict):
            payload = row["new_experience_distill"]
            records.append(
                {
                    "record_type": "new_experience_candidate_generation",
                    **common,
                    "raw_generation": payload.get("raw"),
                    "error": payload.get("error"),
                    "usage": payload.get("usage", zero_usage()),
                    "candidate_events": [event for event in row.get("candidate_events", []) if event.get("source") == "new" and event.get("kind") == "experience"],
                }
            )
        if isinstance(row.get("repair_judgment"), dict):
            payload = row["repair_judgment"]
            records.append(
                {
                    "record_type": "failure_owner_and_skill_generation",
                    **common,
                    "raw_generation": payload.get("raw"),
                    "parsed": payload.get("parsed"),
                    "error": payload.get("error"),
                    "usage": payload.get("usage", zero_usage()),
                    "candidate_events": row.get("candidate_events", []),
                }
            )
    for item in window_log.get("candidate_updates", []):
        records.append({"record_type": "candidate_pool_merge", **item})
    for item in window_log.get("active_dedup_checks", []):
        records.append({"record_type": "active_entry_check", **item})
    for item in window_log.get("repair_validations", []):
        records.append({"record_type": "repair_validation", **item})
    records.append(
        {
            "record_type": "window_update_summary",
            "step": window_log.get("step"),
            "window_start_step": window_log.get("window_start_step"),
            "window_end_step": window_log.get("window_end_step"),
            "candidate_count": window_log.get("candidate_count", 0),
            "promoted_count": len(window_log.get("promoted_new_skills", [])),
            "duplicate_new_skill_count": len(window_log.get("duplicate_new_skills", [])),
            "replacement_count": len(window_log.get("replacements", [])),
            "promoted_new_skills": window_log.get("promoted_new_skills", []),
            "duplicate_new_skills": window_log.get("duplicate_new_skills", []),
            "replacements": window_log.get("replacements", []),
        }
    )
    return records


def persist_evolution_window_outputs(
    output_dir: str,
    window_step: int,
    window_rows: List[Dict[str, Any]],
    window_log: Dict[str, Any],
) -> None:
    window_dir = os.path.join(output_dir, "evolution", step_dir_name(window_step))
    write_jsonl(os.path.join(window_dir, "refine", "generation_records.jsonl"), build_refine_generation_records(window_rows))
    write_jsonl(os.path.join(window_dir, "update", "generation_records.jsonl"), build_update_generation_records(window_rows, window_log))


def load_generation_records_from_output(output_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(output_dir, "generation").glob("*.jsonl")):
        rows.extend(load_jsonl(str(path)))
    return sorted(rows, key=lambda item: safe_int(item.get("step"), 0))


def load_window_logs_from_output(output_dir: str) -> List[Dict[str, Any]]:
    logs = []
    for path in sorted(Path(output_dir, "evolution").glob("step_*/update/generation_records.jsonl")):
        for row in load_jsonl(str(path)):
            if row.get("record_type") == "window_update_summary":
                logs.append(
                    {
                        "step": row.get("step"),
                        "window_start_step": row.get("window_start_step"),
                        "window_end_step": row.get("window_end_step"),
                        "candidate_count": row.get("candidate_count", 0),
                        "promoted_new_skills": row.get("promoted_new_skills", []),
                        "duplicate_new_skills": row.get("duplicate_new_skills", []),
                        "replacements": row.get("replacements", []),
                    }
                )
    return sorted(logs, key=lambda item: safe_int(item.get("step"), 0))


def available_checkpoint_steps(output_dir: str) -> List[int]:
    steps = []
    for path in sorted(Path(output_dir, "skillbank", "checkpoints").glob("step_*")):
        if not path.is_dir():
            continue
        if not (path / "skillbank.json").exists() or not (path / "candidates.json").exists():
            continue
        step = checkpoint_step_from_name(path.name)
        if step > 0:
            steps.append(step)
    return sorted(set(steps))


def latest_stable_resume_step(output_dir: str, records: List[Dict[str, Any]], window_logs: List[Dict[str, Any]]) -> int:
    max_record_step = max((safe_int(row.get("step"), 0) for row in records), default=0)
    log_steps = {safe_int(log.get("step"), 0) for log in window_logs}
    checkpoint_steps = set(available_checkpoint_steps(output_dir))
    stable_steps = [step for step in checkpoint_steps if step <= max_record_step and step in log_steps]
    return max(stable_steps, default=0)


def load_checkpoint_state(
    output_dir: str,
    step: int,
    alpha: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    checkpoint_dir = Path(output_dir, "skillbank", "checkpoints", step_dir_name(step))
    skillbank = normalize_skillbank(read_json(str(checkpoint_dir / "skillbank.json")), alpha)
    candidates_payload = read_json(str(checkpoint_dir / "candidates.json"))
    return skillbank, list(candidates_payload.get("candidates", []))


def build_summary_markdown(records: List[Dict[str, Any]], window_logs: List[Dict[str, Any]]) -> str:
    total = len(records)
    correct = sum(1 for row in records if row.get("is_correct"))
    used = sum(1 for row in records if row.get("used_skillbank"))
    candidate_count = sum(len(row.get("candidate_events", [])) for row in records)
    lines = ["# Skillbank Evolve DSV3 Summary", ""]
    lines.append(f"- Accuracy: {(correct / total if total else 0.0):.2%} ({correct}/{total})")
    lines.append(f"- Used skillbank: {used}/{total}")
    lines.append(f"- Candidate events: {candidate_count}")
    lines.append(f"- Update windows: {len(window_logs)}")
    lines.append("")
    lines.append("## Dataset Accuracy")
    lines.append("")
    lines.append("| Dataset | Correct | Total | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    dataset_accs = []
    grouped: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in records:
        grouped[str(row.get("dataset", "unknown"))].append(row)
    for dataset in sorted(grouped):
        rows = grouped[dataset]
        ds_total = len(rows)
        ds_correct = sum(1 for row in rows if row.get("is_correct"))
        ds_acc = ds_correct / ds_total if ds_total else 0.0
        dataset_accs.append(ds_acc)
        lines.append(f"| {dataset} | {ds_correct} | {ds_total} | {ds_acc:.2%} |")
    macro_avg = sum(dataset_accs) / len(dataset_accs) if dataset_accs else 0.0
    lines.append(f"| Macro AVG | - | - | {macro_avg:.2%} |")
    lines.append("")
    lines.append("## Window Updates")
    lines.append("")
    lines.append("| Step | Candidates | Promoted | Duplicate Drops | Replacements |")
    lines.append("|---:|---:|---:|---:|---:|")
    for log in window_logs:
        lines.append(
            f"| {log.get('step')} | {log.get('candidate_count', 0)} | "
            f"{len(log.get('promoted_new_skills', []))} | "
            f"{len(log.get('duplicate_new_skills', []))} | "
            f"{len(log.get('replacements', []))} |"
        )
    lines.append("")
    return "\n".join(lines)


def persist_outputs(output_dir: str, records: List[Dict[str, Any]], window_logs: List[Dict[str, Any]]) -> None:
    persist_generation_records(output_dir, records)
    persist_execution_reports(output_dir, records)
    write_text(os.path.join(output_dir, "summary.md"), build_summary_markdown(records, window_logs))


def initialize_or_resume_state(
    base_skillbank_json: str,
    output_dir: str,
    resume: bool,
    alpha: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    global CANDIDATE_ID_COUNTER
    CANDIDATE_ID_COUNTER = 0
    skillbank_path = os.path.join(output_dir, "skillbank", "skillbank.json")
    if resume and os.path.exists(skillbank_path):
        records = load_generation_records_from_output(output_dir)
        window_logs = load_window_logs_from_output(output_dir)
        stable_step = latest_stable_resume_step(output_dir, records, window_logs)
        if stable_step > 0:
            skillbank, candidates = load_checkpoint_state(output_dir, stable_step, alpha)
            records = [row for row in records if safe_int(row.get("step"), 0) <= stable_step]
            window_logs = [log for log in window_logs if safe_int(log.get("step"), 0) <= stable_step]
        else:
            skillbank = normalize_skillbank(read_json(base_skillbank_json), alpha)
            candidates = []
            records = []
            window_logs = []
        sync_candidate_id_counter(candidates)
        sync_candidate_id_counter_from_output(output_dir, stable_step if stable_step > 0 else 0)
        resume_step = stable_step
        return skillbank, candidates, records, window_logs, resume_step
    skillbank = normalize_skillbank(read_json(base_skillbank_json), alpha)
    return skillbank, [], [], [], 0


def check_environment() -> None:
    subprocess.run(["python3", "--version"], text=True, check=True)
    subprocess.run(["python3", "-c", "import gurobipy"], text=True, check=True)


def run_evolve(
    testset_dir: str,
    base_skillbank_json: str,
    prompt_file: str,
    output_dir: str,
    dataset_names: Optional[List[str]],
    shuffle_seed: int,
    window_size: int,
    max_workers: int,
    update_max_workers: int,
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    routing_max_tokens: int,
    distill_temperature: float,
    distill_max_tokens: int,
    repair_temperature: float,
    repair_max_tokens: int,
    dedup_max_tokens: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
    alpha: float,
    support_threshold: int,
    repair_success_threshold: float,
    regression_success_threshold: float,
    repair_validation_failures: int,
    repair_validation_regressions: int,
    resume: bool,
) -> None:
    prompts = load_prompt_bundle(prompt_file)
    tasks = build_mixed_tasks(testset_dir, dataset_names, shuffle_seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for dirname in ("generation", "execution", "evolution", "skillbank"):
        Path(output_dir, dirname).mkdir(parents=True, exist_ok=True)
    skillbank, candidates, records, window_logs, resume_step = initialize_or_resume_state(
        base_skillbank_json, output_dir, resume, alpha
    )
    save_checkpoint(skillbank, candidates, output_dir, resume_step)

    total_tasks = len(tasks)
    print(f"total_tasks={total_tasks}, resume_step={resume_step}, window_size={window_size}, workers={max_workers}")
    if resume_step >= total_tasks:
        print("Run already complete.")
        return

    with tqdm(total=total_tasks, initial=resume_step, desc="Evolving", dynamic_ncols=True) as pbar:
        for window_start in range(resume_step, total_tasks, window_size):
            window_tasks = tasks[window_start: window_start + window_size]
            indexed_tasks = [(window_start + offset + 1, task) for offset, task in enumerate(window_tasks)]
            window_step = indexed_tasks[-1][0] if indexed_tasks else window_start
            skillbank_snapshot = json.loads(json.dumps(skillbank, ensure_ascii=False))
            batch_rows: List[Dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
                future_to_task = {
                    executor.submit(
                        process_task,
                        step,
                        window_step,
                        task,
                        prompts,
                        skillbank_snapshot,
                        output_dir,
                        url,
                        model_name,
                        generation_temperature,
                        generation_max_tokens,
                        routing_max_tokens,
                        distill_temperature,
                        distill_max_tokens,
                        repair_temperature,
                        repair_max_tokens,
                        timeout,
                        err_tolerance,
                        use_percentage_err_tolerance,
                    ): (step, task)
                    for step, task in indexed_tasks
                }
                for future in as_completed(future_to_task):
                    step, task = future_to_task[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = build_processing_error_record(step, task, exc)
                    batch_rows.append(row)
                    pbar.update(1)

            batch_rows.sort(key=lambda item: item["step"])
            records.extend(batch_rows)
            last_step = batch_rows[-1]["step"] if batch_rows else window_start
            log = update_candidates_and_skillbank(
                window_rows=batch_rows,
                all_records=records,
                skillbank=skillbank,
                candidates=candidates,
                prompts=prompts,
                output_dir=output_dir,
                current_step=last_step,
                alpha=alpha,
                support_threshold=support_threshold,
                repair_success_threshold=repair_success_threshold,
                regression_success_threshold=regression_success_threshold,
                repair_validation_failures=repair_validation_failures,
                repair_validation_regressions=repair_validation_regressions,
                update_max_workers=update_max_workers,
                url=url,
                model_name=model_name,
                generation_temperature=generation_temperature,
                generation_max_tokens=generation_max_tokens,
                dedup_max_tokens=dedup_max_tokens,
                timeout=timeout,
                err_tolerance=err_tolerance,
                use_percentage_err_tolerance=use_percentage_err_tolerance,
            )
            window_logs.append(log)
            save_checkpoint(skillbank, candidates, output_dir, last_step)
            persist_evolution_window_outputs(output_dir, last_step, batch_rows, log)
            persist_outputs(output_dir, records, window_logs)
            correct = sum(1 for row in records if row.get("is_correct"))
            pbar.set_postfix_str(f"acc={correct / len(records):.2%}, candidates={len(candidates)}")

    print("Finished.")
    print("Summary:", os.path.join(output_dir, "summary.md"))
    print("Current skillbank:", os.path.join(output_dir, "skillbank", "skillbank.json"))
    print("Candidate skills:", os.path.join(output_dir, "skillbank", "candidates.json"))


def main() -> None:
    parser = argparse.ArgumentParser("skillbank_evolve")
    parser.add_argument("--testset_dir", type=str, default=os.path.join("data", "testset"))
    parser.add_argument("--base_skillbank_json", type=str, default=os.path.join("skillbank", "skillbank.json"))
    parser.add_argument("--prompt_file", type=str, default=os.path.join("template", "evolve_prompt.txt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "evolve"))
    parser.add_argument("--base_url", "--url", dest="base_url", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--shuffle_seed", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--update_max_workers", type=int, default=4)
    parser.add_argument("--generation_temperature", type=float, default=0.2)
    parser.add_argument("--generation_max_tokens", type=int, default=8192)
    parser.add_argument("--routing_max_tokens", type=int, default=2048)
    parser.add_argument("--distill_temperature", type=float, default=0.2)
    parser.add_argument("--distill_max_tokens", type=int, default=2048)
    parser.add_argument("--repair_temperature", type=float, default=0.2)
    parser.add_argument("--repair_max_tokens", type=int, default=2048)
    parser.add_argument("--dedup_max_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--err_tolerance", type=float, default=0.05)
    parser.add_argument("--use_percentage_err_tolerance", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--support_threshold", type=int, default=5)
    parser.add_argument("--repair_success_threshold", type=float, default=0.6)
    parser.add_argument("--regression_success_threshold", type=float, default=0.8)
    parser.add_argument("--repair_validation_failures", type=int, default=3)
    parser.add_argument("--repair_validation_regressions", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_env_check", action="store_true")
    args = parser.parse_args()

    if not args.skip_env_check:
        check_environment()
    run_evolve(
        testset_dir=args.testset_dir,
        base_skillbank_json=args.base_skillbank_json,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        shuffle_seed=args.shuffle_seed,
        window_size=args.window_size,
        max_workers=args.max_workers,
        update_max_workers=args.update_max_workers,
        url=args.base_url,
        model_name=args.model_name,
        generation_temperature=args.generation_temperature,
        generation_max_tokens=args.generation_max_tokens,
        routing_max_tokens=args.routing_max_tokens,
        distill_temperature=args.distill_temperature,
        distill_max_tokens=args.distill_max_tokens,
        repair_temperature=args.repair_temperature,
        repair_max_tokens=args.repair_max_tokens,
        dedup_max_tokens=args.dedup_max_tokens,
        timeout=args.timeout,
        err_tolerance=args.err_tolerance,
        use_percentage_err_tolerance=args.use_percentage_err_tolerance,
        alpha=args.alpha,
        support_threshold=args.support_threshold,
        repair_success_threshold=args.repair_success_threshold,
        regression_success_threshold=args.regression_success_threshold,
        repair_validation_failures=args.repair_validation_failures,
        repair_validation_regressions=args.repair_validation_regressions,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
