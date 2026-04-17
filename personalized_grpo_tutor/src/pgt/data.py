from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

from datasets import Dataset

from .prompts import build_system_prompt, build_user_prompt


def read_jsonl(path: str | Path) -> List[Dict]:
    file_path = Path(path)

    # Allow both jsonl and json(list[dict]) sources.
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        raise ValueError(f"Expected list JSON in {file_path}")

    items: List[Dict] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _build_tests_from_input_output(input_output: Mapping) -> List[Dict[str, str]]:
    if not isinstance(input_output, Mapping):
        return []

    if isinstance(input_output.get("inputs"), list) and isinstance(input_output.get("outputs"), list):
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        tests: List[Dict[str, str]] = []
        for inp, out in zip(inputs, outputs):
            tests.append({"input": str(inp), "output": str(out)})
        return tests

    if "input" in input_output and "output" in input_output:
        return [{"input": str(input_output.get("input", "")), "output": str(input_output.get("output", ""))}]

    return []


def build_sft_dataset(path: str | Path) -> Dataset:
    rows = read_jsonl(path)
    records = []
    for item in rows:
        student_level = item.get("student_level", "beginner")

        if "instruction" not in item or "output" not in item:
            raise KeyError("Unsupported SFT schema: expected instruction/input/output")

        instruction = str(item.get("instruction", "")).strip()
        input_text = str(item.get("input", "")).strip()
        if instruction and input_text:
            user_text = f"{instruction}\n\n{input_text}"
        else:
            user_text = instruction or input_text
        assistant_text = str(item.get("output", ""))

        prompt = {
            "system": build_system_prompt(student_level),
            "user": user_text,
        }
        records.append(
            {
                "system": prompt["system"],
                "user": prompt["user"],
                "assistant": assistant_text,
            }
        )
    return Dataset.from_list(records)


def build_rl_dataset(path: str | Path) -> Dataset:
    rows = read_jsonl(path)
    records = []
    for item in rows:
        student_level = item.get("student_level", "beginner")
        instruction = str(item.get("instruction", "")).strip()
        problem_text = str(item.get("input", item.get("question", item.get("problem", "")))).strip()

        if instruction and problem_text:
            user_prompt = f"{instruction}\n\n{problem_text}"
        elif problem_text:
            user_prompt = build_user_prompt(problem_text)
        else:
            raise KeyError("Unsupported RL schema: expected input/question/problem text")

        tests = item.get("tests") if isinstance(item.get("tests"), list) else _build_tests_from_input_output(item.get("input_output", {}))

        records.append(
            {
                "problem": problem_text,
                "student_level": student_level,
                "tests": tests,
                "target_step_count": int(item.get("target_step_count", 4)),
                "prompt": user_prompt,
                "system": build_system_prompt(student_level),
            }
        )
    return Dataset.from_list(records)
