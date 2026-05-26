#!/usr/bin/env python3

import argparse
import collections
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm


PROBLEM_TYPE_DESCRIPTIONS = {
    "allocation": "Problems whose main decision is how much resource, budget, investment, production, or mixture quantity to allocate. Includes resource allocation, budget allocation, investment planning, production quantity decisions, blending/diet models, and simple LP/ILP quantity optimization.",
    "assignment": "Problems whose main decision is assigning one set of entities to another. Includes person-to-task, customer-to-facility, project-to-worker, and other matching-style formulations.",
    "selection": "Problems whose main decision is selecting a subset of candidate options. Includes knapsack-style selection, set covering, facility location, fire station placement, media selection, and binary buy-or-not-buy decisions.",
    "flow": "Problems centered on movement through a network with flow balance or supply-demand structure. Includes transportation, transshipment, maximum flow, minimum-cost flow, and supply-demand network optimization.",
    "time_planning": "Problems with decisions linked across multiple time periods. Includes multi-period planning, inventory balance, backlog, carry-over state transitions, and quarter-by-quarter production planning.",
    "routing": "Problems whose core decision is a path, tour, or visit order. Includes TSP, VRP, route design, tour construction, and closed-loop visitation problems.",
    "scheduling": "Problems whose core decision is the timing or sequencing of tasks, jobs, machines, crews, or operations. Includes job shop scheduling, machine scheduling, crew scheduling, aircraft landing scheduling, and precedence-constrained sequencing.",
    "special": "Use only when the main modeling difficulty is a special mathematical structure rather than the domain skeleton. Includes fractional objectives, multi-objective optimization, explicit disjunctive logic, and absolute-value-dominant formulations.",
}

PROMPT_KEYS = [
    "base_generation_system_prompt", "base_generation_user_prompt", "problem_type_classification_system_prompt",
    "problem_type_classification_user_prompt", "skill_retrieval_system_prompt", "skill_retrieval_user_prompt",
    "skill_augmented_generation_system_prompt", "skill_augmented_generation_user_prompt",
    "strategy_augmented_generation_user_prompt", "experience_augmented_generation_user_prompt",
]

OUTPUT_RECORD_FIELDS = [
    "dataset",
    "mode",
    "source_id",
    "difficulty",
    "question",
    "en_answer",
    "question_type",
    "selected_strategy_index",
    "selected_experience_indexes",
    "raw_retrieval",
    "raw_generations",
    "en_gurobi_code",
    "status",
    "error_message",
    "usage",
    "latency_sec",
]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_json(path: str, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, target)


def serialize_output_record(row: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key in OUTPUT_RECORD_FIELDS:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        if key == "raw_retrieval" and value == "":
            continue
        serialized[key] = value
    return serialized


def write_text(path: str, data: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, target)


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


def parse_labeled_index_list(text: str, label: str) -> Optional[List[str]]:
    raw = strip_reasoning_tags(text)
    pattern = rf"(?im)^\s*{re.escape(label)}\s*=\s*(.+?)\s*$"
    matches = re.findall(pattern, raw)
    if len(matches) != 1:
        raise ValueError(f"Cannot find exactly one `{label}=...` line in retrieval output.")

    rhs = matches[0].strip()
    if rhs.lower() == "none":
        return None

    if re.fullmatch(r"\[\s*[A-Za-z0-9_\-,\s]*\]", rhs):
        inner = rhs[1:-1].strip()
        if not inner:
            return []
        values = [item.strip() for item in inner.split(",")]
    elif re.fullmatch(r"[A-Za-z0-9_\-]+(?:\s*,\s*[A-Za-z0-9_\-]+)*", rhs):
        # Be tolerant to minor format drift like `Strategy=foo_001` or
        # `Experience=exp_001, exp_002`; these are semantically unambiguous.
        values = [item.strip() for item in rhs.split(",")]
    else:
        raise ValueError(f"Invalid `{label}` list format: {rhs}")

    if not all(values):
        raise ValueError(f"Invalid empty `{label}` index in list: {rhs}")
    return values


def parse_skill_retrieval_output(text: str) -> Dict[str, Optional[List[str]]]:
    return {
        "strategy_indexes": parse_labeled_index_list(text, "Strategy"),
        "experience_indexes": parse_labeled_index_list(text, "Experience"),
    }


def extract_python_block(text: str) -> str:
    raw = str(text or "")
    raw = strip_reasoning_tags(raw)
    tagged = re.search(r"<python>\s*(.*?)\s*</python>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tagged:
        inner = tagged.group(1)
        fenced = re.search(r"```python\s*(.*?)```", inner, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return sanitize_python_snippet(fenced.group(1))
        return sanitize_python_snippet(inner)
    fenced = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return sanitize_python_snippet(fenced.group(1))
    generic = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if generic:
        return sanitize_python_snippet(generic.group(1))
    return ""


def compact_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_problem_type(problem_type: str) -> str:
    problem_type = (problem_type or "").strip()
    return problem_type if problem_type in PROBLEM_TYPE_DESCRIPTIONS else "special"


def parse_problem_type_label(text: str) -> str:
    raw = text.strip()
    raw = re.sub(r"<think_never_used_[^>]+>.*?</think_never_used_[^>]+>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty problem_type output.")

    lines = [line.strip("`* \t").strip().lower() for line in raw.splitlines() if line.strip()]
    for line in reversed(lines):
        if line in PROBLEM_TYPE_DESCRIPTIONS:
            return line

    pattern = r"\b(" + "|".join(re.escape(key) for key in PROBLEM_TYPE_DESCRIPTIONS) + r")\b"
    matches = re.findall(pattern, raw.lower())
    if matches:
        return matches[-1]
    raise ValueError(f"Cannot parse problem_type from model output: {text[:500]}")


def call_model(system_prompt: str, user_prompt: str, url: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    base_url = url.rstrip("/")
    request_url = base_url
    if not request_url.endswith("/v1/chat/completions"):
        request_url = request_url + "/v1/chat/completions"

    payload = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "model": model_name,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    last_error = None
    for _ in range(3):
        try:
            resp = requests.post(request_url, headers=headers, data=json.dumps(payload), timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return strip_reasoning_tags(content), data.get("usage", {})
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"call_model failed: {last_error}")


def build_skillbank_index(skillbank_path: str) -> Dict[str, Any]:
    skillbank = read_json(skillbank_path)
    strategies_by_type: Dict[str, List[Dict[str, Any]]] = {key: [] for key in PROBLEM_TYPE_DESCRIPTIONS}
    experiences_by_type: Dict[str, List[Dict[str, Any]]] = {key: [] for key in PROBLEM_TYPE_DESCRIPTIONS}
    strategies_by_index: Dict[str, Dict[str, Any]] = {}
    experiences_by_index: Dict[str, Dict[str, Any]] = {}

    if "strategies" in skillbank or "experiences" in skillbank:
        grouped_skillbank = {
            problem_type: {
                "strategies": [dict(item, problem_type=problem_type) for item in skillbank.get("strategies", []) if normalize_problem_type(item.get("problem_type", "")) == problem_type],
                "experiences": [dict(item, problem_type=problem_type) for item in skillbank.get("experiences", []) if normalize_problem_type(item.get("problem_type", "")) == problem_type],
            }
            for problem_type in PROBLEM_TYPE_DESCRIPTIONS
        }
    else:
        grouped_skillbank = skillbank

    for problem_type in PROBLEM_TYPE_DESCRIPTIONS:
        bucket = grouped_skillbank.get(problem_type, {}) or {}
        for raw_item in bucket.get("strategies", []):
            item = dict(raw_item, problem_type=problem_type)
            strategies_by_type[problem_type].append(item)
            if item.get("index") is not None:
                strategies_by_index[str(item.get("index"))] = item

        for raw_item in bucket.get("experiences", []):
            item = dict(raw_item, problem_type=problem_type)
            experiences_by_type[problem_type].append(item)
            if item.get("index") is not None:
                experiences_by_index[str(item.get("index"))] = item

    return {
        "raw": skillbank,
        "strategies_by_type": strategies_by_type,
        "experiences_by_type": experiences_by_type,
        "strategies_by_index": strategies_by_index,
        "experiences_by_index": experiences_by_index,
    }


def format_strategy_candidates(items: List[Dict[str, Any]]) -> str:
    payload = []
    for item in items:
        payload.append({"index": item.get("index"), "summary": compact_text(item.get("summary", ""), 700)})
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_experience_candidates(items: List[Dict[str, Any]]) -> str:
    payload = []
    for item in items:
        trigger = item.get("trigger", "")
        payload.append({"index": item.get("index"), "trigger": compact_text(trigger, 300)})
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_selected_experiences(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "[]"
    return json.dumps(
        [
            {
                "index": item.get("index"),
                "trigger": item.get("trigger"),
                "guidance": item.get("guidance"),
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=2,
    )


def classify_problem_type(question: str, prompt_bundle: Dict[str, str], url: str, model_name: str, max_tokens: int) -> Tuple[str, str, Dict[str, Any]]:
    user_prompt = fill_template(prompt_bundle["problem_type_classification_user_prompt"], {"question": question})
    content, usage = call_model(
        prompt_bundle["problem_type_classification_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return normalize_problem_type(parse_problem_type_label(content)), content, usage


def retrieve_skills(
    question: str, problem_type: str, prompt_bundle: Dict[str, str], url: str, model_name: str, max_tokens: int, skillbank_index: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    strategy_candidates = skillbank_index["strategies_by_type"].get(problem_type, [])
    experience_candidates = skillbank_index["experiences_by_type"].get(problem_type, [])
    user_prompt = fill_template(
        prompt_bundle["skill_retrieval_user_prompt"],
        {
            "question": question,
            "candidate_strategies": format_strategy_candidates(strategy_candidates),
            "candidate_experiences": format_experience_candidates(experience_candidates),
        },
    )
    content, usage = call_model(
        prompt_bundle["skill_retrieval_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return content, usage


def select_skill_retrieval(
    retrieval_raw: str, problem_type: str, skillbank_index: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[List[str]], Optional[List[str]]]:
    strategy_candidates = skillbank_index["strategies_by_type"].get(problem_type, [])
    experience_candidates = skillbank_index["experiences_by_type"].get(problem_type, [])
    strategy_candidates_by_index = {str(item.get("index")): item for item in strategy_candidates if item.get("index") is not None}
    experience_candidates_by_index = {str(item.get("index")): item for item in experience_candidates if item.get("index") is not None}

    data = parse_skill_retrieval_output(retrieval_raw)
    raw_strategy_indexes = data.get("strategy_indexes")
    if raw_strategy_indexes is not None and len(raw_strategy_indexes) > 1:
        raise ValueError(f"Expected at most 1 strategy index, got {raw_strategy_indexes}")

    selected_strategy = None
    if raw_strategy_indexes:
        selected_strategy = strategy_candidates_by_index.get(str(raw_strategy_indexes[0]))
        if selected_strategy is None:
            raise ValueError(f"Strategy index is not in current `{problem_type}` bucket: {raw_strategy_indexes[0]}")

    raw_experience_indexes = data.get("experience_indexes")
    if raw_experience_indexes is None:
        raw_experience_indexes = []
    if len(raw_experience_indexes) > 2:
        raise ValueError(f"Expected at most 2 experience indexes, got {raw_experience_indexes}")

    selected_experiences: List[Dict[str, Any]] = []
    seen = set()
    for exp_index in raw_experience_indexes:
        exp_item = experience_candidates_by_index.get(str(exp_index))
        if exp_item is None:
            raise ValueError(f"Experience index is not in current `{problem_type}` bucket: {exp_index}")
        if exp_index in seen:
            continue
        selected_experiences.append(exp_item)
        seen.add(exp_index)

    return selected_strategy, selected_experiences, raw_strategy_indexes, raw_experience_indexes


def generate_base_solution(question: str, prompt_bundle: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    user_prompt = fill_template(prompt_bundle["base_generation_user_prompt"], {"question": question})
    return call_model(
        prompt_bundle["base_generation_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_skillbank_solution(
    question: str, selected_strategy: Optional[Dict[str, Any]], selected_experiences: List[Dict[str, Any]],
    prompt_bundle: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    strategy_text = "None"
    if selected_strategy is not None:
        strategy_text = json.dumps(
            {
                "index": selected_strategy.get("index"),
                "summary": selected_strategy.get("summary"),
                "procedure": selected_strategy.get("procedure", []),
            }, ensure_ascii=False, indent=2
        )

    experiences_text = "[]"
    if selected_experiences:
        experiences_text = format_selected_experiences(selected_experiences)

    user_prompt = fill_template(
        prompt_bundle["skill_augmented_generation_user_prompt"],
        {
            "question": question,
            "selected_strategy": strategy_text,
            "selected_experiences": experiences_text,
        },
    )
    return call_model(
        prompt_bundle["skill_augmented_generation_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_strategy_only_solution(
    question: str, selected_strategy: Optional[Dict[str, Any]], prompt_bundle: Dict[str, str],
    url: str, model_name: str, temperature: float, max_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    strategy_text = "None"
    if selected_strategy is not None:
        strategy_text = json.dumps(
            {
                "index": selected_strategy.get("index"),
                "summary": selected_strategy.get("summary"),
                "procedure": selected_strategy.get("procedure", []),
            }, ensure_ascii=False, indent=2
        )

    user_prompt = fill_template(
        prompt_bundle["strategy_augmented_generation_user_prompt"],
        {
            "question": question,
            "selected_strategy": strategy_text,
        },
    )
    return call_model(
        prompt_bundle["skill_augmented_generation_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_experience_only_solution(
    question: str, selected_experiences: List[Dict[str, Any]], prompt_bundle: Dict[str, str],
    url: str, model_name: str, temperature: float, max_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    experiences_text = "[]"
    if selected_experiences:
        experiences_text = format_selected_experiences(selected_experiences)

    user_prompt = fill_template(
        prompt_bundle["experience_augmented_generation_user_prompt"],
        {
            "question": question,
            "selected_experiences": experiences_text,
        },
    )
    return call_model(
        prompt_bundle["skill_augmented_generation_system_prompt"],
        user_prompt,
        url,
        model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def zero_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def merge_usage(total_usage: Dict[str, int], usage: Optional[Dict[str, Any]]) -> Dict[str, int]:
    usage = usage or {}
    for key in total_usage:
        total_usage[key] += safe_int(usage.get(key), 0)
    return total_usage


def build_item_id(item: Dict[str, Any], fallback_index: int) -> Any:
    for key in ("index", "id"):
        if key in item:
            return item[key]
    return fallback_index


def initialize_record(dataset_name: str, mode: str, item_index: int, item: Dict[str, Any]) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "mode": mode,
        "source_index": item_index,
        "source_id": build_item_id(item, item_index),
        "question": str(item.get("en_question") or item.get("question") or "").strip(),
        "en_answer": item.get("en_answer", item.get("answer")),
        "raw_generations": "",
        "en_gurobi_code": "",
        "status": "success",
    }
    if item.get("difficulty") is not None:
        record["difficulty"] = item.get("difficulty")
    return record


def classify_skill_context(
    question: str,
    prompt_bundle: Dict[str, str],
    url: str,
    model_name: str,
    routing_max_tokens: int,
) -> Dict[str, Any]:
    start_time = time.time()
    result: Dict[str, Any] = {
        "question_type": "special",
        "classification_error": None,
        "usage": zero_usage(),
        "latency_sec": 0.0,
    }
    try:
        problem_type, _, usage = classify_problem_type(
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            max_tokens=routing_max_tokens,
        )
        result["question_type"] = problem_type
        merge_usage(result["usage"], usage)
    except Exception as exc:
        result["classification_error"] = str(exc)
    result["latency_sec"] = round(time.time() - start_time, 3)
    return result


def retrieve_skill_context(
    question: str,
    problem_type: str,
    prompt_bundle: Dict[str, str],
    skillbank_index: Dict[str, Any],
    url: str,
    model_name: str,
    routing_max_tokens: int,
) -> Dict[str, Any]:
    start_time = time.time()
    result: Dict[str, Any] = {
        "raw_retrieval": "",
        "selected_strategy": None,
        "selected_experiences": [],
        "retrieval_error": None,
        "usage": zero_usage(),
        "latency_sec": 0.0,
    }
    try:
        retrieval_raw, usage = retrieve_skills(
            question=question,
            problem_type=problem_type,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            max_tokens=routing_max_tokens,
            skillbank_index=skillbank_index,
        )
        selected_strategy, selected_experiences, _, _ = select_skill_retrieval(
            retrieval_raw, problem_type, skillbank_index
        )
        result["raw_retrieval"] = retrieval_raw
        result["selected_strategy"] = selected_strategy
        result["selected_experiences"] = selected_experiences
        merge_usage(result["usage"], usage)
    except Exception as exc:
        result["retrieval_error"] = str(exc)
    result["latency_sec"] = round(time.time() - start_time, 3)
    return result


def attach_context_fields(record: Dict[str, Any], mode: str, skill_context: Optional[Dict[str, Any]]) -> None:
    if not skill_context or mode == "base":
        return

    if skill_context.get("classification_error") is None:
        record["question_type"] = skill_context.get("question_type", "special")

    raw_retrieval = skill_context.get("raw_retrieval", "")
    if raw_retrieval:
        record["raw_retrieval"] = raw_retrieval

    selected_strategy = skill_context.get("selected_strategy")
    if mode != "no_strategy" and selected_strategy is not None:
        record["selected_strategy_index"] = selected_strategy.get("index")

    selected_experiences = skill_context.get("selected_experiences") or []
    if mode != "no_exp" and selected_experiences:
        record["selected_experience_indexes"] = [entry.get("index") for entry in selected_experiences]


def generate_response_for_mode(
    mode: str,
    question: str,
    prompt_bundle: Dict[str, str],
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    skill_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    if mode == "base":
        return generate_base_solution(
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
        )

    selected_strategy = None
    selected_experiences: List[Dict[str, Any]] = []
    if skill_context:
        selected_strategy = skill_context.get("selected_strategy")
        selected_experiences = list(skill_context.get("selected_experiences") or [])

    if mode == "skillbank":
        if selected_strategy is not None and selected_experiences:
            return generate_skillbank_solution(
                question=question,
                selected_strategy=selected_strategy,
                selected_experiences=selected_experiences,
                prompt_bundle=prompt_bundle,
                url=url,
                model_name=model_name,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        if selected_strategy is not None:
            return generate_strategy_only_solution(
                question=question,
                selected_strategy=selected_strategy,
                prompt_bundle=prompt_bundle,
                url=url,
                model_name=model_name,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        if selected_experiences:
            return generate_experience_only_solution(
                question=question,
                selected_experiences=selected_experiences,
                prompt_bundle=prompt_bundle,
                url=url,
                model_name=model_name,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        return generate_base_solution(
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
        )

    if mode == "no_exp":
        if selected_strategy is not None:
            return generate_strategy_only_solution(
                question=question,
                selected_strategy=selected_strategy,
                prompt_bundle=prompt_bundle,
                url=url,
                model_name=model_name,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        return generate_base_solution(
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
        )

    if mode == "no_strategy":
        if selected_experiences:
            return generate_experience_only_solution(
                question=question,
                selected_experiences=selected_experiences,
                prompt_bundle=prompt_bundle,
                url=url,
                model_name=model_name,
                temperature=generation_temperature,
                max_tokens=generation_max_tokens,
            )
        return generate_base_solution(
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            temperature=generation_temperature,
            max_tokens=generation_max_tokens,
        )

    raise ValueError(f"Unsupported mode: {mode}")


def generate_mode_record(
    dataset_name: str,
    mode: str,
    item_index: int,
    item: Dict[str, Any],
    prompt_bundle: Dict[str, str],
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    skill_context: Optional[Dict[str, Any]] = None,
    routing_usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    record = initialize_record(dataset_name, mode, item_index, item)
    start_time = time.time()
    routing_usage_value = dict(routing_usage or zero_usage())
    total_usage = zero_usage()

    merge_usage(total_usage, routing_usage_value)
    attach_context_fields(record, mode, skill_context)

    try:
        question = record["question"]
        if not question:
            raise ValueError("Missing question text.")

        response_text, usage = generate_response_for_mode(
            mode=mode,
            question=question,
            prompt_bundle=prompt_bundle,
            url=url,
            model_name=model_name,
            generation_temperature=generation_temperature,
            generation_max_tokens=generation_max_tokens,
            skill_context=skill_context,
        )
        merge_usage(total_usage, usage)
        record["raw_generations"] = response_text
        record["en_gurobi_code"] = extract_python_block(response_text)
    except Exception as exc:
        record["status"] = "error"
        record["error_message"] = str(exc)

    shared_latency = float((skill_context or {}).get("latency_sec", 0.0)) if mode != "base" else 0.0
    record["usage"] = total_usage
    record["latency_sec"] = round(shared_latency + (time.time() - start_time), 3)
    return record


def result_path(output_dir: str, mode: str, dataset_name: str) -> str:
    return os.path.join(output_dir, mode, f"{dataset_name}.jsonl")


def load_existing_results(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return load_jsonl(path)


def row_order_key(row: Dict[str, Any], source_order: Dict[str, int]) -> Tuple[int, str]:
    source_id = str(row.get("source_id"))
    return (source_order.get(source_id, 10**12), source_id)


def flush_grouped_results(
    output_dir: str,
    grouped_results: Dict[Tuple[str, str], List[Dict[str, Any]]],
    dirty_keys: List[Tuple[str, str]],
    dataset_source_orders: Dict[str, Dict[str, int]],
) -> None:
    if not dirty_keys:
        return

    for mode, dataset_name in sorted(set(dirty_keys)):
        grouped_results[(mode, dataset_name)].sort(
            key=lambda item: row_order_key(item, dataset_source_orders[dataset_name])
        )
        write_jsonl(
            result_path(output_dir, mode, dataset_name),
            [serialize_output_record(item) for item in grouped_results[(mode, dataset_name)]],
        )

    summarize_results(output_dir, grouped_results)


def summarize_results(output_dir: str, grouped_results: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> None:
    summary: Dict[str, Any] = {"jobs": []}
    for (mode, dataset_name), rows in sorted(grouped_results.items()):
        rows_sorted = list(rows)
        success_count = sum(1 for row in rows_sorted if row.get("status", "success") == "success")
        error_count = sum(1 for row in rows_sorted if row.get("status") == "error")
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for row in rows_sorted:
            row_usage = row.get("usage", {})
            for key in usage:
                usage[key] += safe_int(row_usage.get(key), 0)
        summary["jobs"].append(
            {
                "mode": mode,
                "dataset": dataset_name,
                "records": len(rows_sorted),
                "success": success_count,
                "error": error_count,
                "usage": usage,
                "output_file": result_path(output_dir, mode, dataset_name),
            }
        )
    write_json(os.path.join(output_dir, "summary.json"), summary)


RESULT_PREFIX = "Just print the best solution:"
NO_SOLUTION_TEXT = "No Best Solution"
ADD_SCRIPT = """
if model.status == GRB.OPTIMAL:
    print(f"Just print the best solution: {model.objVal}")
else:
    print("No Best Solution")
""".strip()


def check_environment() -> None:
    subprocess.run(["python3", "--version"], text=True, check=True)
    subprocess.run(["python3", "-c", "import gurobipy"], text=True, check=True)


def sanitize_python_snippet(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

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
    cleaned_code = sanitize_python_snippet(code) or str(code or "")
    if RESULT_PREFIX in cleaned_code or NO_SOLUTION_TEXT in cleaned_code:
        return cleaned_code

    injected_code, injected = _inject_main_block_result_print(cleaned_code)
    if injected:
        return injected_code.rstrip() + "\n"

    return injected_code.rstrip() + "\n\n" + ADD_SCRIPT + "\n"


def extract_obj(str_log: str) -> Optional[float]:
    if RESULT_PREFIX not in str(str_log or ""):
        return None
    for item in str(str_log).splitlines():
        if RESULT_PREFIX in item:
            result = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", item)
            if result:
                return float(result[-1])
    return None


def execute_script(code: str, output_dir: str, example_id: str, timeout: int) -> Dict[str, Any]:
    gurobi_dir = os.path.join(output_dir, "gurobi_code")
    os.makedirs(gurobi_dir, exist_ok=True)

    script_content = prepare_script_for_execution(code)
    script_path = os.path.join(gurobi_dir, f"{example_id}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    try:
        proc = subprocess.run(
            ["python3", script_path],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        execution_result = proc.stdout or ""
        execution_best_solution = extract_obj(execution_result)
        if execution_best_solution is not None:
            execution_state = "Execution Successful and Best Solution Found"
        elif NO_SOLUTION_TEXT in execution_result:
            execution_best_solution = NO_SOLUTION_TEXT
            execution_state = "Execution Successful but No Best Solution Found"
        elif proc.returncode != 0:
            execution_best_solution = None
            execution_state = f"Execution Failed: {proc.stderr}"
        else:
            execution_best_solution = None
            execution_state = "Execution Successful but Out of Expectation"
        return {
            "execution_state": execution_state,
            "execution_best_solution": execution_best_solution,
            "execution_result": execution_result,
        }
    except subprocess.TimeoutExpired as exc:
        execution_result = exc.stdout or ""
        execution_best_solution = extract_obj(execution_result)
        if execution_best_solution is not None:
            execution_state = "Execution Timed Out after Printing Objective"
        elif NO_SOLUTION_TEXT in execution_result:
            execution_best_solution = NO_SOLUTION_TEXT
            execution_state = "Execution Timed Out after Printing No Best Solution"
        else:
            execution_best_solution = None
            execution_state = "Execution Failed: Timeout"
        return {
            "execution_state": execution_state,
            "execution_best_solution": execution_best_solution,
            "execution_result": execution_result,
        }


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


def evaluate_example(
    example: Dict[str, Any],
    idx: int,
    output_dir: str,
    passk: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
    timeout: int,
) -> Dict[str, Any]:
    example_id = example.get("source_id", example.get("id", idx))
    difficulty = example.get("difficulty", "unknown")
    code_candidates = example.get("en_gurobi_code", [])
    if not isinstance(code_candidates, list):
        code_candidates = [code_candidates]

    parsed_scripts = []
    seen = set()
    for item in code_candidates:
        parsed = sanitize_python_snippet(item)
        if not parsed or parsed in seen:
            continue
        parsed_scripts.append(parsed)
        seen.add(parsed)
        if len(parsed_scripts) >= passk:
            break

    if not parsed_scripts:
        return {
            "id": example_id,
            "difficulty": difficulty,
            f"pass_{passk}": False,
            "results_correct": [False],
            "execution_details": [{"state": "Execution Failed: No code", "best_solution": None}],
        }

    raw_gt = example.get("en_answer")
    gt = _to_float(raw_gt)
    results_correct: List[bool] = []
    execution_details: List[Dict[str, Any]] = []

    for script_idx, script in enumerate(parsed_scripts):
        exec_out = execute_script(script, output_dir=output_dir, example_id=f"{example_id}_{script_idx}", timeout=timeout)
        pred = exec_out["execution_best_solution"]
        ok = False

        if isinstance(raw_gt, str) and raw_gt.strip().lower() == NO_SOLUTION_TEXT.lower():
            ok = pred == NO_SOLUTION_TEXT
        elif gt is not None and isinstance(pred, (int, float)):
            if gt == 0:
                ok = abs(pred) <= err_tolerance
            elif use_percentage_err_tolerance:
                ok = abs((pred - gt) / gt) <= err_tolerance
            else:
                ok = abs(pred - gt) <= err_tolerance

        results_correct.append(ok)
        execution_details.append({"state": exec_out["execution_state"], "best_solution": pred})

    return {
        "id": example_id,
        "difficulty": difficulty,
        f"pass_{passk}": any(results_correct),
        "results_correct": results_correct,
        "execution_details": execution_details,
    }


def build_evaluation_report(results: List[Dict[str, Any]], output_dir: str, passk: int) -> Dict[str, Any]:
    pass_key = f"pass_{passk}"
    correct_count = sum(1 for row in results if row[pass_key])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count else 0.0

    execution_state_counts = collections.defaultdict(int)
    for row in results:
        for detail in row["execution_details"]:
            execution_state_counts[detail["state"]] += 1

    easy_correct_count = sum(1 for row in results if row[pass_key] and row["difficulty"] in ["Easy", "easy"])
    easy_total_count = sum(1 for row in results if row["difficulty"] in ["Easy", "easy"])
    medium_correct_count = sum(1 for row in results if row[pass_key] and row["difficulty"] in ["Medium", "medium"])
    medium_total_count = sum(1 for row in results if row["difficulty"] in ["Medium", "medium"])
    hard_correct_count = sum(1 for row in results if row[pass_key] and row["difficulty"] in ["Hard", "hard"])
    hard_total_count = sum(1 for row in results if row["difficulty"] in ["Hard", "hard"])

    report = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "easy_accuracy": (easy_correct_count / easy_total_count) if easy_total_count else 0.0,
        "easy_correct_count": easy_correct_count,
        "easy_total_count": easy_total_count,
        "medium_accuracy": (medium_correct_count / medium_total_count) if medium_total_count else 0.0,
        "medium_correct_count": medium_correct_count,
        "medium_total_count": medium_total_count,
        "hard_accuracy": (hard_correct_count / hard_total_count) if hard_total_count else 0.0,
        "hard_correct_count": hard_correct_count,
        "hard_total_count": hard_total_count,
        "execution_state_counts": dict(execution_state_counts),
        "results": results,
    }
    write_json(os.path.join(output_dir, "evaluation_report.json"), report)
    return report


def _format_acc(correct: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{correct / total:.2%} ({correct}/{total})"


def build_eval_summary_markdown(job_rows: List[Dict[str, Any]]) -> str:
    by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in job_rows:
        by_dataset.setdefault(row["dataset"], {})[row["mode"]] = row

    mode_totals: Dict[str, Dict[str, int]] = {}
    for row in job_rows:
        bucket = mode_totals.setdefault(row["mode"], {"correct_count": 0, "total_count": 0})
        bucket["correct_count"] += int(row.get("correct_count", 0))
        bucket["total_count"] += int(row.get("total_count", 0))

    datasets = sorted(by_dataset)
    header = "| Method | " + " | ".join(datasets + ["AVG"]) + " |"
    divider = "|---|" + "---:|" * (len(datasets) + 1)
    lines = ["# Evaluation Summary", "", header, divider]

    preferred_modes = ["skillbank", "no_exp", "no_strategy", "base"]
    ordered_modes = [mode for mode in preferred_modes if mode in mode_totals]
    ordered_modes.extend(sorted(mode for mode in mode_totals if mode not in preferred_modes))

    for mode in ordered_modes:
        cells = [mode]
        total_correct = 0
        total_count = 0
        for dataset in datasets:
            row = by_dataset.get(dataset, {}).get(mode)
            correct = int(row.get("correct_count", 0)) if row else 0
            total = int(row.get("total_count", 0)) if row else 0
            total_correct += correct
            total_count += total
            cells.append(_format_acc(correct, total))
        cells.append(_format_acc(total_correct, total_count))
        lines.append("| " + " | ".join(cells) + " |")

    if "base" in mode_totals:
        base_total_count = mode_totals["base"]["total_count"]
        base_avg = (mode_totals["base"]["correct_count"] / base_total_count) if base_total_count else None
        for mode in ordered_modes:
            if mode == "base":
                continue
            gain_cells = [f"gain_vs_base({mode})"]
            for dataset in datasets:
                base_row = by_dataset.get(dataset, {}).get("base")
                target_row = by_dataset.get(dataset, {}).get(mode)
                base_correct = int(base_row.get("correct_count", 0)) if base_row else 0
                base_total = int(base_row.get("total_count", 0)) if base_row else 0
                target_correct = int(target_row.get("correct_count", 0)) if target_row else 0
                target_total = int(target_row.get("total_count", 0)) if target_row else 0
                dataset_base_acc = (base_correct / base_total) if base_total else None
                dataset_target_acc = (target_correct / target_total) if target_total else None
                gain_cells.append(
                    "-" if dataset_base_acc is None or dataset_target_acc is None else f"{dataset_target_acc - dataset_base_acc:+.2%}"
                )
            target_total_count = mode_totals[mode]["total_count"]
            target_avg = (mode_totals[mode]["correct_count"] / target_total_count) if target_total_count else None
            gain_cells.append("-" if base_avg is None or target_avg is None else f"{target_avg - base_avg:+.2%}")
            lines.append("| " + " | ".join(gain_cells) + " |")

    lines.append("")
    return "\n".join(lines)


def run_execution_evaluation(
    generation_output_dir: str,
    eval_output_dir: str,
    modes: List[str],
    datasets: Optional[List[str]],
    max_workers: int,
    timeout: int,
    err_tolerance: float,
    use_percentage_err_tolerance: bool,
) -> None:
    dataset_filter = None
    if datasets:
        dataset_filter = {name.strip().removesuffix(".jsonl") for name in datasets if name.strip()}

    jobs = []
    for mode in modes:
        mode_dir = Path(generation_output_dir) / mode
        if not mode_dir.exists():
            continue
        for file_path in sorted(mode_dir.glob("*.jsonl")):
            dataset_name = file_path.stem
            if dataset_filter and dataset_name not in dataset_filter:
                continue
            jobs.append((mode, dataset_name, str(file_path)))

    if not jobs:
        raise ValueError(f"No generation jsonl files found under: {generation_output_dir}")

    job_inputs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for mode, dataset_name, input_file in jobs:
        data = load_jsonl(input_file)
        report_dir = os.path.join(eval_output_dir, mode, dataset_name)
        os.makedirs(report_dir, exist_ok=True)
        job_inputs[(mode, dataset_name)] = {
            "input_file": input_file,
            "report_dir": report_dir,
            "data": data,
            "results": [None] * len(data),
        }

    job_rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_example = {}
        for (mode, dataset_name), job in job_inputs.items():
            for idx, example in enumerate(job["data"]):
                future = executor.submit(
                    evaluate_example,
                    example,
                    idx,
                    job["report_dir"],
                    1,
                    err_tolerance,
                    use_percentage_err_tolerance,
                    timeout,
                )
                future_to_example[future] = (mode, dataset_name, idx)

        for future in tqdm(as_completed(future_to_example), total=len(future_to_example), desc="Running code"):
            mode, dataset_name, idx = future_to_example[future]
            try:
                job_inputs[(mode, dataset_name)]["results"][idx] = future.result()
            except Exception as exc:
                example = job_inputs[(mode, dataset_name)]["data"][idx]
                example_id = example.get("source_id", example.get("id", idx))
                job_inputs[(mode, dataset_name)]["results"][idx] = {
                    "id": example_id,
                    "difficulty": example.get("difficulty", "unknown"),
                    "pass_1": False,
                    "results_correct": [False],
                    "execution_details": [{"state": f"Execution Failed: {exc}", "best_solution": None}],
                }

    for (mode, dataset_name), job in sorted(job_inputs.items()):
        input_file = job["input_file"]
        report_dir = job["report_dir"]
        try:
            results = [row for row in job["results"] if row is not None]
            report = build_evaluation_report(results, report_dir, 1)
            job_rows.append(
                {
                    "mode": mode,
                    "dataset": dataset_name,
                    "input_file": input_file,
                    "report_dir": report_dir,
                    "accuracy": report.get("accuracy", 0.0),
                    "correct_count": report.get("correct_count", 0),
                    "total_count": report.get("total_count", 0),
                    "easy_accuracy": report.get("easy_accuracy", 0.0),
                    "medium_accuracy": report.get("medium_accuracy", 0.0),
                    "hard_accuracy": report.get("hard_accuracy", 0.0),
                    "execution_state_counts": report.get("execution_state_counts", {}),
                }
            )
        except Exception as exc:
            job_rows.append(
                {
                    "mode": mode,
                    "dataset": dataset_name,
                    "input_file": input_file,
                    "report_dir": report_dir,
                    "accuracy": 0.0,
                    "correct_count": 0,
                    "total_count": 0,
                    "easy_accuracy": 0.0,
                    "medium_accuracy": 0.0,
                    "hard_accuracy": 0.0,
                    "execution_state_counts": {},
                    "error": str(exc),
                }
            )

    summary_md = build_eval_summary_markdown(job_rows)
    write_text(os.path.join(eval_output_dir, "summary.md"), summary_md)
    print("\n===== Execution Evaluation Summary =====")
    print(summary_md)
    print(f"\nSummary Markdown: {os.path.join(eval_output_dir, 'summary.md')}")


def discover_datasets(testset_dir: str, dataset_names: Optional[List[str]]) -> List[Path]:
    all_paths = sorted(Path(testset_dir).glob("*.jsonl"))
    if not dataset_names:
        return all_paths

    normalized = set()
    for name in dataset_names:
        name = name.strip()
        if not name:
            continue
        normalized.add(name if name.endswith(".jsonl") else f"{name}.jsonl")
    return [path for path in all_paths if path.name in normalized]


ROUTED_MODES = {"skillbank", "no_exp", "no_strategy"}


def build_sample_skill_context(
    sample: Dict[str, Any],
    prompt_bundle: Dict[str, str],
    skillbank_index: Dict[str, Any],
    url: str,
    model_name: str,
    routing_max_tokens: int,
) -> Dict[str, Any]:
    question = str(sample["item"].get("en_question") or sample["item"].get("question") or "").strip()
    result = classify_skill_context(
        question,
        prompt_bundle,
        url,
        model_name,
        routing_max_tokens,
    )
    context = {
        "question_type": result.get("question_type", "special"),
        "classification_error": result.get("classification_error"),
        "retrieval_error": None,
        "raw_retrieval": "",
        "selected_strategy": None,
        "selected_experiences": [],
        "usage": dict(result.get("usage", zero_usage())),
        "latency_sec": float(result.get("latency_sec", 0.0)),
    }

    if context["classification_error"] is not None:
        return context

    retrieval = retrieve_skill_context(
        question,
        context["question_type"],
        prompt_bundle,
        skillbank_index,
        url,
        model_name,
        routing_max_tokens,
    )
    context["retrieval_error"] = retrieval.get("retrieval_error")
    context["raw_retrieval"] = retrieval.get("raw_retrieval", "")
    context["selected_strategy"] = retrieval.get("selected_strategy")
    context["selected_experiences"] = retrieval.get("selected_experiences", [])
    merge_usage(context["usage"], retrieval.get("usage"))
    context["latency_sec"] = round(
        float(context.get("latency_sec", 0.0)) + float(retrieval.get("latency_sec", 0.0)),
        3,
    )
    return context


def process_sample_records(
    sample: Dict[str, Any],
    prompt_bundle: Dict[str, str],
    skillbank_index: Dict[str, Any],
    url: str,
    model_name: str,
    generation_temperature: float,
    generation_max_tokens: int,
    routing_max_tokens: int,
) -> List[Dict[str, Any]]:
    missing_modes = list(sample["missing_modes"])
    skill_context = None
    if any(mode in ROUTED_MODES for mode in missing_modes):
        skill_context = build_sample_skill_context(
            sample,
            prompt_bundle,
            skillbank_index,
            url,
            model_name,
            routing_max_tokens,
        )

    rows = []
    for mode in missing_modes:
        routing_usage = (skill_context or {}).get("usage", zero_usage()) if mode in ROUTED_MODES else zero_usage()
        rows.append(
            generate_mode_record(
                sample["dataset_name"],
                mode,
                sample["item_index"],
                sample["item"],
                prompt_bundle,
                url,
                model_name,
                generation_temperature,
                generation_max_tokens,
                skill_context,
                routing_usage,
            )
        )
    return rows


def process_all(
    testset_dir: str, skillbank_json: str, prompt_file: str, output_dir: str, url: str, model_name: str, modes: List[str],
    dataset_names: Optional[List[str]], max_workers: int, generation_temperature: float, generation_max_tokens: int, routing_max_tokens: int
) -> None:
    prompt_bundle = load_prompt_bundle(prompt_file)
    skillbank_index = build_skillbank_index(skillbank_json)
    datasets = discover_datasets(testset_dir, dataset_names)
    if not datasets:
        raise ValueError("No dataset files found.")

    grouped_results: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    dataset_source_orders: Dict[str, Dict[str, int]] = {}
    pending_samples: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for dataset_path in datasets:
        dataset_name = dataset_path.stem
        items = load_jsonl(str(dataset_path))
        dataset_source_orders[dataset_name] = {
            str(build_item_id(item, item_index)): item_index for item_index, item in enumerate(items)
        }
        for mode in modes:
            path = result_path(output_dir, mode, dataset_name)
            existing_rows = load_existing_results(path)
            grouped_results[(mode, dataset_name)] = existing_rows
            completed_source_ids = {str(row.get("source_id")) for row in existing_rows if "source_id" in row}
            for item_index, item in enumerate(items):
                item_source_id = str(build_item_id(item, item_index))
                if item_source_id in completed_source_ids:
                    continue
                sample_key = (dataset_name, item_source_id)
                sample_state = pending_samples.setdefault(
                    sample_key,
                    {
                        "dataset_name": dataset_name,
                        "item_index": item_index,
                        "item": item,
                        "missing_modes": [],
                    },
                )
                sample_state["missing_modes"].append(mode)

    pending_mode_tasks = sum(len(sample["missing_modes"]) for sample in pending_samples.values())
    skill_context_tasks = sum(
        1 for sample in pending_samples.values() if any(mode in ROUTED_MODES for mode in sample["missing_modes"])
    )

    print(
        f"datasets={len(datasets)}, modes={modes}, pending_samples={len(pending_samples)}, "
        f"pending_mode_tasks={pending_mode_tasks}, skill_context_tasks={skill_context_tasks}, workers={max_workers}"
    )
    summarize_results(output_dir, grouped_results)
    if not pending_samples:
        print("No pending tasks. Finished.")
        return

    dirty_keys: List[Tuple[str, str]] = []
    flush_every = max(20, max_workers * 2)
    generation_completed = 0
    last_flush_completed = 0

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_sample = {
            executor.submit(
                process_sample_records,
                sample,
                prompt_bundle,
                skillbank_index,
                url,
                model_name,
                generation_temperature,
                generation_max_tokens,
                routing_max_tokens,
            ): sample
            for sample in pending_samples.values()
        }

        with tqdm(total=pending_mode_tasks, desc="Generating") as generation_bar:
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    rows = future.result()
                except Exception as exc:
                    rows = []
                    for mode in sample["missing_modes"]:
                        row = initialize_record(sample["dataset_name"], mode, sample["item_index"], sample["item"])
                        row["status"] = "error"
                        row["error_message"] = str(exc)
                        row["usage"] = zero_usage()
                        row["latency_sec"] = 0.0
                        rows.append(row)

                for row in rows:
                    key = (row["mode"], sample["dataset_name"])
                    grouped_results[key].append(row)
                    dirty_keys.append(key)

                generation_completed += len(rows)
                generation_bar.update(len(rows))

                if generation_completed - last_flush_completed >= flush_every:
                    flush_grouped_results(output_dir, grouped_results, dirty_keys, dataset_source_orders)
                    dirty_keys = []
                    last_flush_completed = generation_completed

    flush_grouped_results(output_dir, grouped_results, dirty_keys, dataset_source_orders)

    print("Finished.")
    print("Summary:", os.path.join(output_dir, "summary.json"))


def main() -> None:
    parser = argparse.ArgumentParser("skillbank_test")
    parser.add_argument("--testset_dir", type=str, default=os.path.join("data", "testset"))
    parser.add_argument("--skillbank_json", type=str, default=os.path.join("skillbank", "skillbank.json"))
    parser.add_argument("--prompt_file", type=str, default=os.path.join("template", "eval_prompt.txt"))
    parser.add_argument("--output_dir", type=str, default=os.path.join("outputs", "evaluation", "generation"))
    parser.add_argument("--eval_output_dir", type=str, default=os.path.join("outputs", "evaluation", "results"))
    parser.add_argument(
        "--base_url",
        "--url",
        dest="base_url",
        type=str,
        required=True,
        help="OpenAI-compatible model server base URL.",
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--modes", nargs="+", default=["skillbank"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--eval_max_workers", type=int, default=1)
    parser.add_argument("--generation_temperature", type=float, default=0.2)
    parser.add_argument("--generation_max_tokens", type=int, default=8192)
    parser.add_argument("--routing_max_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--err_tolerance", type=float, default=0.05)
    parser.add_argument("--use_percentage_err_tolerance", action="store_true")
    args = parser.parse_args()

    
    process_all(
        testset_dir=args.testset_dir, skillbank_json=args.skillbank_json, prompt_file=args.prompt_file, output_dir=args.output_dir,
        url=args.base_url, model_name=args.model_name, modes=args.modes, dataset_names=args.datasets, max_workers=args.max_workers,
        generation_temperature=args.generation_temperature, generation_max_tokens=args.generation_max_tokens,
        routing_max_tokens=args.routing_max_tokens,
    )
    check_environment()
    run_execution_evaluation(
        generation_output_dir=args.output_dir,
        eval_output_dir=args.eval_output_dir,
        modes=args.modes,
        datasets=args.datasets,
        max_workers=args.eval_max_workers,
        timeout=args.timeout,
        err_tolerance=args.err_tolerance,
        use_percentage_err_tolerance=args.use_percentage_err_tolerance,
    )


if __name__ == "__main__":
    main()
