from __future__ import annotations

import inspect
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments
from trl import SFTTrainer

from pgt.config import load_yaml_with_extends
from pgt.data import build_sft_dataset, read_jsonl
from pgt.prompts import build_system_prompt
from pgt.reward.compiler_runner import compile_and_run_c_code
from pgt.training.utils import build_bnb_config, build_lora_config


_NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*(.*?)```", re.DOTALL)
_SECTION_RE = re.compile(r"【[^】]+】")


def _extract_last_code_block(text: str) -> tuple[str, str]:
    matches = list(_CODE_BLOCK_RE.finditer(text or ""))
    if not matches:
        return "", ""
    last = matches[-1]
    lang = (last.group(1) or "").strip().lower()
    code = (last.group(2) or "").strip()
    return lang, code


def _extract_eval_tests(sample: Mapping[str, Any]) -> List[Mapping[str, str]]:
    if isinstance(sample.get("tests"), list):
        return list(sample.get("tests", []))
    if isinstance(sample.get("test_cases"), list):
        return list(sample.get("test_cases", []))
    meta = sample.get("meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("tests"), list):
        return list(meta.get("tests", []))
    return []


def _format_adherence_score(pred_text: str, gold_text: str) -> float:
    pred = pred_text or ""
    gold = gold_text or ""
    gold_sections = _SECTION_RE.findall(gold)
    if gold_sections:
        section_score = sum(1.0 for sec in gold_sections if sec in pred) / float(len(gold_sections))
    else:
        section_score = 0.0

    checks = [
        section_score,
        1.0 if "【最终代码（可运行）】" in pred else 0.0,
        1.0 if bool(_NUMBERED_STEP_RE.search(pred)) else 0.0,
        1.0 if "```" in pred else 0.0,
    ]
    return sum(checks) / float(len(checks))


def _run_python_code_with_tests(code: str, tests: List[Mapping[str, str]], timeout_sec: float = 1.5) -> float:
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


def _code_correctness_score(completion: str, sample: Mapping[str, Any]) -> float:
    lang, code = _extract_last_code_block(completion)
    if not code:
        return 0.0

    tests = _extract_eval_tests(sample)
    if tests:
        if lang.startswith("py"):
            return _run_python_code_with_tests(code=code, tests=tests)
        run_res = compile_and_run_c_code(code=code, tests=tests)
        return run_res.pass_ratio

    # Fallback for SFT samples without executable tests.
    return 1.0 if code else 0.0


def _build_online_eval_samples(sft_data_path: str, num_samples: int) -> List[Mapping[str, Any]]:
    rows = read_jsonl(sft_data_path)
    if not rows:
        return []
    capped = max(1, int(num_samples))
    return rows[:capped]


class OnlineSFTMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        samples: List[Mapping[str, Any]],
        every_n_steps: int,
        max_new_tokens: int,
        use_wandb: bool,
    ):
        self.tokenizer = tokenizer
        self.samples = samples
        self.every_n_steps = max(1, int(every_n_steps))
        self.max_new_tokens = max(64, int(max_new_tokens))
        self.use_wandb = use_wandb

    def on_step_end(self, args, state, control, **kwargs):
        if not self.samples:
            return control
        if state.global_step <= 0 or state.global_step % self.every_n_steps != 0:
            return control
        if not state.is_world_process_zero:
            return control

        model = kwargs.get("model")
        if model is None:
            return control

        metrics = self._evaluate_model(model)
        metrics["sft_eval/global_step"] = float(state.global_step)
        print(
            f"[SFT-OnlineEval step={state.global_step}] "
            f"format={metrics['sft_eval/format_score']:.4f} "
            f"acc={metrics['sft_eval/code_accuracy']:.4f}"
        )

        if self.use_wandb:
            try:
                import wandb  # type: ignore

                if wandb.run is not None:
                    wandb.log(metrics, step=state.global_step)
            except Exception:
                pass

        return control

    def _evaluate_model(self, model) -> Dict[str, float]:
        was_training = model.training
        model.eval()

        total_format = 0.0
        total_acc = 0.0
        evaluated = 0

        try:
            device = next(model.parameters()).device
            with torch.no_grad():
                for sample in self.samples:
                    instruction = str(sample.get("instruction", "")).strip()
                    input_text = str(sample.get("input", "")).strip()
                    if instruction and input_text:
                        user_text = f"{instruction}\n\n{input_text}"
                    else:
                        user_text = instruction or input_text

                    messages = [
                        {"role": "system", "content": build_system_prompt(sample.get("student_level", "beginner"))},
                        {"role": "user", "content": user_text},
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    model_inputs = self.tokenizer(prompt, return_tensors="pt")
                    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

                    output_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                    completion_ids = output_ids[0, model_inputs["input_ids"].shape[1]:]
                    completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                    total_format += _format_adherence_score(completion, str(sample.get("output", "")))
                    total_acc += _code_correctness_score(completion, sample)
                    evaluated += 1
        finally:
            if was_training:
                model.train()

        if evaluated == 0:
            return {
                "sft_eval/format_score": 0.0,
                "sft_eval/code_accuracy": 0.0,
            }

        return {
            "sft_eval/format_score": total_format / float(evaluated),
            "sft_eval/code_accuracy": total_acc / float(evaluated),
        }


def run_sft(config_path: str):
    cfg = load_yaml_with_extends(config_path)
    model_cfg = cfg["model"]
    sft_cfg = cfg["sft"]
    wandb_cfg = cfg.get("wandb", {})

    dataset = build_sft_dataset(cfg["paths"]["sft_data"])

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = bool(model_cfg.get("use_4bit", True))
    bnb_config = build_bnb_config(use_4bit=use_4bit)

    model_load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_4bit:
        model_load_kwargs["quantization_config"] = bnb_config
    else:
        model_load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        **model_load_kwargs,
    )

    lora_config = build_lora_config(cfg["lora"])

    def formatting_func(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text

    output_dir = str(Path(cfg["paths"]["output_dir"]) / sft_cfg["output_subdir"])
    use_wandb = bool(wandb_cfg.get("enabled", True))
    if use_wandb:
        os.environ["WANDB_PROJECT"] = str(wandb_cfg.get("project", cfg.get("project_name", "personalized_grpo_tutor")))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        if wandb_cfg.get("mode"):
            os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])

    run_name = str(wandb_cfg.get("sft_run_name", f"sft_{sft_cfg['output_subdir']}"))
    online_eval_cfg = cfg.get("sft_online_eval", {})

    args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=float(sft_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(sft_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(sft_cfg["gradient_accumulation_steps"]),
        learning_rate=float(sft_cfg["learning_rate"]),
        logging_steps=int(sft_cfg["logging_steps"]),
        save_steps=int(sft_cfg["save_steps"]),
        warmup_ratio=float(sft_cfg["warmup_ratio"]),
        bf16=True,
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else "none",
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": dataset,
        "peft_config": lora_config,
        "formatting_func": formatting_func,
    }
    seq_length = int(sft_cfg.get("max_seq_length", model_cfg.get("max_length", 2048)))
    sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "max_seq_length" in sft_init_params:
        trainer_kwargs["max_seq_length"] = seq_length
    elif "max_length" in sft_init_params:
        trainer_kwargs["max_length"] = seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    if bool(online_eval_cfg.get("enabled", True)):
        eval_samples = _build_online_eval_samples(
            sft_data_path=cfg["paths"]["sft_data"],
            num_samples=int(online_eval_cfg.get("num_samples", 6)),
        )
        trainer.add_callback(
            OnlineSFTMetricsCallback(
                tokenizer=tokenizer,
                samples=eval_samples,
                every_n_steps=int(online_eval_cfg.get("every_n_steps", 20)),
                max_new_tokens=int(online_eval_cfg.get("max_new_tokens", 256)),
                use_wandb=use_wandb,
            )
        )

    trainer.train()
    trainer.save_model(output_dir)
