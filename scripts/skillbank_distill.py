#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from tqdm import tqdm


ALLOWED_PROBLEM_TYPES = {
    "allocation", "assignment", "selection", "flow",
    "time_planning", "routing", "scheduling", "special",
}


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, target)


def extract_prompt_var(text: str, var_name: str) -> str:
    match = re.search(rf"{re.escape(var_name)}\s*=\s*\"\"\"(.*?)\"\"\"", text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot find prompt variable: {var_name}")
    return match.group(1)


def load_prompt_bundle(prompt_file: str) -> Dict[str, str]:
    content = Path(prompt_file).read_text(encoding="utf-8")
    keys = [
        "strategy_generation_system_prompt", "strategy_generation_user_prompt",
        "experience_generation_system_prompt", "experience_generation_user_prompt",
        "strategy_dedup_system_prompt", "strategy_dedup_user_prompt",
        "experience_dedup_system_prompt", "experience_dedup_user_prompt",
    ]
    return {key: extract_prompt_var(content, key) for key in keys}


def fill_template(template: str, mapping: Dict[str, str]) -> str:
    for key, value in mapping.items():
        template = template.replace("{" + key + "}", value)
    return template


def parse_json_from_text(text: str) -> Any:
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except Exception:
            pass
    obj = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if obj:
        return json.loads(obj.group(0))
    raise ValueError(f"Cannot parse JSON from model output: {raw[:500]}")


def normalize_problem_type(problem_type: str) -> str:
    problem_type = (problem_type or "").strip()
    return problem_type if problem_type in ALLOWED_PROBLEM_TYPES else "special"


def next_id(existing_ids: List[str], prefix: str) -> str:
    max_num = 0
    for value in existing_ids:
        match = re.search(r"(\d+)$", str(value))
        if match:
            max_num = max(max_num, int(match.group(1)))
    return f"{prefix}{max_num + 1:04d}"


def build_trajectory(block: Dict[str, Any]) -> str:
    parts = [str(block.get("strategy", "")).strip(), str(block.get("sets_params_vars", "")).strip(), str(block.get("objective_constraints", "")).strip()]
    return "\n\n".join(part for part in parts if part)


def ensure_strategy_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(obj.get("summary") or obj.get("core_feature") or "").strip()
    procedure = obj.get("procedure", [])
    if not isinstance(procedure, list):
        procedure = []
    return {
        "problem_type": normalize_problem_type(str(obj.get("problem_type", ""))),
        "summary": summary,
        "procedure": procedure,
    }


def ensure_experience_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    guidance = str(obj.get("guidance") or obj.get("action") or obj.get("reason") or "").strip()
    return {
        "problem_type": normalize_problem_type(str(obj.get("problem_type", ""))),
        "trigger": str(obj.get("trigger", "")).strip(),
        "guidance": guidance,
    }


def call_model(system_prompt: str, prompt: str, url: str, model_name: str, temperature: float = 0.5, max_tokens: int = 4096) -> Tuple[str, int]:
    request_url = f"{url}/v1/chat/completions"
    payload = json.dumps({
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "model": model_name,
        "max_tokens": max_tokens,
        "stop": None,
        "stream": False,
        "temperature": temperature,
    })
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    last_error = None
    for _ in range(3):
        try:
            resp = requests.post(request_url, headers=headers, data=payload, timeout=600)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return data["choices"][0]["message"]["content"], data.get("usage", {}).get("completion_tokens", 0)
        except Exception as e:
            last_error = e
            time.sleep(1)
    raise RuntimeError(f"call_model failed: {last_error}")


def load_state(output_skillbank: str, trace_file: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], set]:
    skillbank_exists, trace_exists = os.path.exists(output_skillbank), os.path.exists(trace_file)
    if skillbank_exists != trace_exists:
        raise ValueError("output_skillbank and trace_file must either both exist or both not exist.")
    if not skillbank_exists:
        return {"strategies": [], "experiences": []}, [], set()
    skillbank = read_json(output_skillbank)
    skillbank.setdefault("strategies", [])
    skillbank.setdefault("experiences", [])
    trace_records = read_json(trace_file)
    processed = set()
    for record in trace_records:
        if record.get("status") == "success" or ("status" not in record and "data_index" in record):
            processed.add(int(record["data_index"]))
    return skillbank, trace_records, processed


def save_state(output_skillbank: str, trace_file: str, strategies: List[Dict[str, Any]], experiences: List[Dict[str, Any]], trace_records: List[Dict[str, Any]], input_json: str, url: str, model_name: str, max_workers: int) -> None:
    success_records = [record for record in trace_records if record.get("status") == "success" or ("status" not in record and "data_index" in record)]
    success_indices = {int(record["data_index"]) for record in success_records}
    error_indices = {int(record["data_index"]) for record in trace_records if record.get("status") == "error" and int(record.get("data_index", -1)) not in success_indices}
    write_json(output_skillbank, {
        "strategies": strategies,
        "experiences": experiences,
        "meta": {
            "model": model_name,
            "url": url,
            "input_json": input_json,
            "successful_samples": len(success_indices),
            "error_samples": len(error_indices),
            "added_strategies": sum(1 for record in success_records if record.get("strategy_added_id")),
            "added_experiences": sum(1 for record in success_records if record.get("experience_added_id")),
            "workers": max_workers,
        },
    })
    write_json(trace_file, trace_records)


def generate_strategy(question: str, correct_trajectory: str, prompt_bundle: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    user_prompt = fill_template(prompt_bundle["strategy_generation_user_prompt"], {"question": question, "correct_trajectory": correct_trajectory})
    content, _ = call_model(prompt_bundle["strategy_generation_system_prompt"], user_prompt, url, model_name, temperature, max_tokens)
    return ensure_strategy_schema(parse_json_from_text(content))


def generate_experience(question: str, correct_trajectory: str, incorrect_trajectory: str, prompt_bundle: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    user_prompt = fill_template(prompt_bundle["experience_generation_user_prompt"], {"question": question, "correct_trajectory": correct_trajectory, "incorrect_trajectory": incorrect_trajectory})
    content, _ = call_model(prompt_bundle["experience_generation_system_prompt"], user_prompt, url, model_name, temperature, max_tokens)
    return ensure_experience_schema(parse_json_from_text(content))


def dedup_strategy(candidate: Dict[str, Any], strategies: List[Dict[str, Any]], prompt_bundle: Dict[str, str], url: str, model_name: str, max_tokens: int) -> Dict[str, Any]:
    same_type_items = [{"strategy_id": item.get("strategy_id"), "summary": item.get("summary", ""), "procedure": item.get("procedure", [])} for item in strategies if normalize_problem_type(item.get("problem_type", "")) == candidate["problem_type"]]
    user_prompt = fill_template(prompt_bundle["strategy_dedup_user_prompt"], {"candidate_strategy": json.dumps(candidate, ensure_ascii=False, indent=2), "existing_strategy_list": json.dumps(same_type_items, ensure_ascii=False, indent=2)})
    content, _ = call_model(prompt_bundle["strategy_dedup_system_prompt"], user_prompt, url, model_name, 0.0, max_tokens)
    return parse_json_from_text(content)


def dedup_experience(candidate: Dict[str, Any], experiences: List[Dict[str, Any]], prompt_bundle: Dict[str, str], url: str, model_name: str, max_tokens: int) -> Dict[str, Any]:
    same_type_items = [{"exp_id": item.get("exp_id"), "trigger": item.get("trigger", ""), "guidance": item.get("guidance", "")} for item in experiences if normalize_problem_type(item.get("problem_type", "")) == candidate["problem_type"]]
    user_prompt = fill_template(prompt_bundle["experience_dedup_user_prompt"], {"candidate_experience": json.dumps(candidate, ensure_ascii=False, indent=2), "existing_experience_list": json.dumps(same_type_items, ensure_ascii=False, indent=2)})
    content, _ = call_model(prompt_bundle["experience_dedup_system_prompt"], user_prompt, url, model_name, 0.0, max_tokens)
    return parse_json_from_text(content)


def apply_strategy_decision(strategies: List[Dict[str, Any]], candidate: Dict[str, Any], dedup_result: Dict[str, Any]) -> str:
    decision = str(dedup_result.get("decision", "duplicate")).strip().lower()
    if decision == "new":
        strategy_id = next_id([str(item.get("strategy_id", "")) for item in strategies], "strategy_")
        strategies.append({"strategy_id": strategy_id, "problem_type": candidate["problem_type"], "summary": candidate["summary"], "procedure": candidate["procedure"]})
        return strategy_id
    if decision == "mergeable":
        matched_id, merged_skill = str(dedup_result.get("matched_id", "")).strip(), dedup_result.get("merged_skill")
        if matched_id and isinstance(merged_skill, dict):
            merged = ensure_strategy_schema(merged_skill)
            for item in strategies:
                if str(item.get("strategy_id", "")) == matched_id:
                    item.update(merged)
                    break
    return ""


def apply_experience_decision(experiences: List[Dict[str, Any]], candidate: Dict[str, Any], dedup_result: Dict[str, Any]) -> str:
    if str(dedup_result.get("decision", "duplicate")).strip().lower() != "new":
        return ""
    exp_id = next_id([str(item.get("exp_id", "")) for item in experiences], "exp_")
    experiences.append({"exp_id": exp_id, "problem_type": candidate["problem_type"], "trigger": candidate["trigger"], "guidance": candidate["guidance"]})
    return exp_id


def generate_candidates_for_item(global_idx: int, item: Dict[str, Any], prompt_bundle: Dict[str, str], url: str, model_name: str, temperature: float, max_tokens: int) -> Dict[str, Any]:
    question = str(item.get("question", "")).strip()
    correct, incorrect = item.get("correct", {}) or {}, item.get("incorrect", {}) or {}
    correct_trajectory, incorrect_trajectory = build_trajectory(correct), build_trajectory(incorrect)
    strategy_candidate = generate_strategy(question, correct_trajectory, prompt_bundle, url, model_name, temperature, max_tokens)
    exp_candidate = generate_experience(question, correct_trajectory, incorrect_trajectory, prompt_bundle, url, model_name, temperature, max_tokens)
    return {
        "data_index": global_idx,
        "question_preview": question[:120],
        "strategy_candidate": strategy_candidate,
        "exp_candidate": exp_candidate,
    }


def commit_item(candidates: Dict[str, Any], executor: ThreadPoolExecutor, strategies: List[Dict[str, Any]], experiences: List[Dict[str, Any]], prompt_bundle: Dict[str, str], url: str, model_name: str, max_tokens: int) -> Dict[str, Any]:
    strategy_candidate = candidates["strategy_candidate"]
    exp_candidate = candidates["exp_candidate"]
    if executor:
        strategy_future = executor.submit(dedup_strategy, strategy_candidate, strategies, prompt_bundle, url, model_name, max_tokens)
        exp_future = executor.submit(dedup_experience, exp_candidate, experiences, prompt_bundle, url, model_name, max_tokens)
        strategy_dedup, exp_dedup = strategy_future.result(), exp_future.result()
    else:
        strategy_dedup = dedup_strategy(strategy_candidate, strategies, prompt_bundle, url, model_name, max_tokens)
        exp_dedup = dedup_experience(exp_candidate, experiences, prompt_bundle, url, model_name, max_tokens)
    return {
        "data_index": candidates["data_index"],
        "status": "success",
        "question_preview": candidates["question_preview"],
        "strategy_candidate": strategy_candidate,
        "strategy_dedup": strategy_dedup,
        "strategy_added_id": apply_strategy_decision(strategies, strategy_candidate, strategy_dedup),
        "experience_candidate": exp_candidate,
        "experience_dedup": exp_dedup,
        "experience_added_id": apply_experience_decision(experiences, exp_candidate, exp_dedup),
    }


def process(input_json: str, prompt_file: str, output_skillbank: str, trace_file: str, url: str, model_name: str, max_workers: int = 2, temperature: float = 0.6, max_tokens: int = 8192) -> None:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    prompt_bundle = load_prompt_bundle(prompt_file)
    dataset = read_json(input_json)
    skillbank, trace_records, processed = load_state(output_skillbank, trace_file)
    strategies, experiences = skillbank["strategies"], skillbank["experiences"]
    pending_indices = [idx for idx in range(len(dataset)) if idx not in processed]

    print(f"读取轨迹数据: {len(dataset)} 条")
    print(f"已完成样本: {len(processed)} 条")
    print(f"待处理样本: {len(pending_indices)} 条")
    if not pending_indices:
        print("无待处理样本, 结束")
        return

    executor = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    futures: Dict[int, Any] = {}
    submit_pos = 0
    try:
        while executor and len(futures) < max_workers and submit_pos < len(pending_indices):
            idx = pending_indices[submit_pos]
            futures[idx] = executor.submit(generate_candidates_for_item, idx, dataset[idx], prompt_bundle, url, model_name, temperature, max_tokens)
            submit_pos += 1

        for global_idx in tqdm(pending_indices, desc="Distilling"):
            item = dataset[global_idx]
            try:
                if executor:
                    candidates = futures.pop(global_idx).result()
                    while len(futures) < max_workers and submit_pos < len(pending_indices):
                        idx = pending_indices[submit_pos]
                        futures[idx] = executor.submit(generate_candidates_for_item, idx, dataset[idx], prompt_bundle, url, model_name, temperature, max_tokens)
                        submit_pos += 1
                else:
                    candidates = generate_candidates_for_item(global_idx, item, prompt_bundle, url, model_name, temperature, max_tokens)
                record = commit_item(candidates, executor, strategies, experiences, prompt_bundle, url, model_name, max_tokens)
            except Exception as e:
                trace_records.append({"data_index": global_idx, "status": "error", "question_preview": str(item.get("question", ""))[:120], "error_message": str(e)})
                save_state(output_skillbank, trace_file, strategies, experiences, trace_records, input_json, url, model_name, max_workers)
                raise
            trace_records.append(record)
            save_state(output_skillbank, trace_file, strategies, experiences, trace_records, input_json, url, model_name, max_workers)
    finally:
        if executor:
            executor.shutdown(wait=True)

    meta = read_json(output_skillbank)["meta"]
    print(f"完成. successful={meta['successful_samples']}, errors={meta['error_samples']}, new_strategies={meta['added_strategies']}, new_experiences={meta['added_experiences']}")
    print("输出 skillbank:", output_skillbank)
    print("输出 trace:", trace_file)


def main() -> None:
    parser = argparse.ArgumentParser("skillbank_distill")
    parser.add_argument("--input_json", type=str, default=os.path.join("data", "trajectory_grouped_1k.json"))
    parser.add_argument("--prompt_file", type=str, default=os.path.join("template", "distill_prompt.txt"))
    parser.add_argument("--output_skillbank", type=str, default=os.path.join("outputs", "skillbank_distilled.json"))
    parser.add_argument("--trace_file", type=str, default=os.path.join("outputs", "skillbank_trace.json"))
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    process(args.input_json, args.prompt_file, args.output_skillbank, args.trace_file, args.url, args.model_name, args.max_workers, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()
