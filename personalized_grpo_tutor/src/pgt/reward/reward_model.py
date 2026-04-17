from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping

from .compiler_runner import compile_and_run_c_code
from .extract import (
    extract_code_with_lang,
    extract_step_count,
)


def _run_python_tests(code: str, tests: List[Mapping[str, str]], timeout_sec: float = 1.5) -> float:
    if not code.strip() or not tests:
        return 0.0

    passed = 0
    total = len(tests)
    with tempfile.TemporaryDirectory() as tmp:
        py_file = Path(tmp) / "main.py"
        py_file.write_text(code, encoding="utf-8")

        for t in tests:
            try:
                cp = subprocess.run(
                    ["python", str(py_file)],
                    input=t.get("input", ""),
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                    check=False,
                )
                expected = (t.get("output", "") or "").strip()
                actual = (cp.stdout or "").strip()
                if cp.returncode == 0 and actual == expected:
                    passed += 1
            except subprocess.TimeoutExpired:
                continue

    return passed / float(total) if total > 0 else 0.0


@dataclass
class RewardBreakdown:
    accuracy: float
    steps: float
    total: float


def compute_reward(
    response_text: str,
    sample: Mapping,
    reward_cfg: Mapping,
) -> RewardBreakdown:
    tests = list(sample.get("tests", [])) if isinstance(sample.get("tests", []), list) else []
    preferred_lang = str(reward_cfg.get("code_language", "python")).strip().lower()
    code_lang, code = extract_code_with_lang(response_text, prefer_lang=preferred_lang)
    use_python = preferred_lang.startswith("py") or code_lang.startswith("py")

    if use_python:
        accuracy = _run_python_tests(code=code, tests=tests)
    else:
        run_res = compile_and_run_c_code(code=code, tests=tests)
        accuracy = run_res.pass_ratio

    target_steps = max(1, int(sample.get("target_step_count", 4)))
    observed_steps = extract_step_count(response_text)
    steps = min(observed_steps / target_steps, 1.0)

    total = (
        float(reward_cfg.get("w_accuracy", 1.2)) * accuracy
        + float(reward_cfg.get("w_steps", 0.4)) * steps
    )

    if accuracy >= 0.99:
        total += float(reward_cfg.get("pass_bonus", 0.1))

    return RewardBreakdown(
        accuracy=accuracy,
        steps=steps,
        total=total,
    )


def build_grpo_reward_func(
    train_dataset,
    reward_cfg: Mapping,
    metrics_logger: Callable[[Dict[str, float]], None] | None = None,
):

    call_idx = 0

    def reward_func(completions: List[str], prompts=None, **kwargs) -> List[float]:
        nonlocal call_idx
        call_idx += 1

        rewards: List[float] = []
        acc_values: List[float] = []
        step_values: List[float] = []

        if prompts is None:
            prompts = [None] * len(completions)

        prompt_to_sample: Dict[str, Mapping] = {
            row["prompt"]: row for row in train_dataset
        }

        for completion, prompt in zip(completions, prompts):
            sample = prompt_to_sample.get(prompt)
            if sample is None:
                rewards.append(-0.1)
                continue

            bd = compute_reward(completion, sample, reward_cfg)
            rewards.append(float(bd.total))
            acc_values.append(float(bd.accuracy))
            step_values.append(float(bd.steps))

        if metrics_logger is not None and acc_values:
            count = float(len(acc_values))
            metrics_logger(
                {
                    "grpo_reward/accuracy_mean": sum(acc_values) / count,
                    "grpo_reward/accuracy_pass_rate": sum(1.0 for x in acc_values if x >= 0.99) / count,
                    "grpo_reward/steps_mean": sum(step_values) / count,
                    "grpo_reward/total_mean": sum(rewards) / float(len(rewards)) if rewards else 0.0,
                    "grpo_reward/reward_func_calls": float(call_idx),
                }
            )

        return rewards

    return reward_func
