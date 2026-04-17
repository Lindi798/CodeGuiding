from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from pgt.config import load_yaml_with_extends
from pgt.data import build_rl_dataset
from pgt.reward.reward_model import build_grpo_reward_func
from pgt.training.utils import build_bnb_config, build_lora_config


def run_grpo(config_path: str):
    cfg = load_yaml_with_extends(config_path)
    model_cfg = cfg["model"]
    grpo_cfg = cfg["grpo"]
    wandb_cfg = cfg.get("wandb", {})

    dataset = build_rl_dataset(cfg["paths"]["rl_data"])

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
    init_adapter_path = str(grpo_cfg.get("init_adapter_path", "")).strip()
    use_existing_adapter = bool(init_adapter_path)
    if use_existing_adapter:
        model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)


    def to_chat_prompt(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["prompt"]},
        ]
        return {"prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)}

    dataset = dataset.map(to_chat_prompt)

    reward_log_every = int(grpo_cfg.get("reward_log_every", 1))
    reward_log_state = {"count": 0}

    def log_reward_metrics(metrics):
        reward_log_state["count"] += 1
        if reward_log_state["count"] % max(1, reward_log_every) != 0:
            return

        if not use_wandb:
            return

        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(metrics)
        except Exception:
            pass

    reward_func = build_grpo_reward_func(
        train_dataset=dataset,
        reward_cfg=cfg["reward"],
        metrics_logger=log_reward_metrics,
    )

    output_dir = str(Path(cfg["paths"]["output_dir"]) / grpo_cfg["output_subdir"])
    use_wandb = bool(wandb_cfg.get("enabled", True))
    if use_wandb:
        os.environ["WANDB_PROJECT"] = str(wandb_cfg.get("project", cfg.get("project_name", "personalized_grpo_tutor")))
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        if wandb_cfg.get("mode"):
            os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])

    run_name = str(wandb_cfg.get("grpo_run_name", f"grpo_{grpo_cfg['output_subdir']}"))

    per_device_bs = int(grpo_cfg["per_device_train_batch_size"])
    grad_acc = int(grpo_cfg["gradient_accumulation_steps"])
    requested_num_generations = int(grpo_cfg["num_generations"])
    # TRL requires generation_batch_size to be divisible by num_generations.
    # In common single-process setups generation_batch_size is effectively
    # per_device_train_batch_size * gradient_accumulation_steps.
    effective_generation_batch_size = max(1, per_device_bs * grad_acc)
    num_generations = requested_num_generations
    if effective_generation_batch_size % num_generations != 0:
        adjusted = math.gcd(effective_generation_batch_size, num_generations)
        num_generations = max(1, adjusted)
        print(
            "[GRPO] Adjusted num_generations from "
            f"{requested_num_generations} to {num_generations} to satisfy "
            f"generation_batch_size({effective_generation_batch_size}) % num_generations == 0"
        )

    config_init_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())

    train_kwargs = {
        "output_dir": output_dir,
        "run_name": run_name,
        "num_train_epochs": float(grpo_cfg["num_train_epochs"]),
        "per_device_train_batch_size": per_device_bs,
        "gradient_accumulation_steps": grad_acc,
        "learning_rate": float(grpo_cfg["learning_rate"]),
        "logging_steps": int(grpo_cfg["logging_steps"]),
        "save_steps": int(grpo_cfg["save_steps"]),
        "num_generations": num_generations,
        "beta": float(grpo_cfg.get("beta", 0.04)),
        "bf16": True,
        "report_to": ["wandb"] if use_wandb else "none",
    }

    if "max_prompt_length" in config_init_params:
        train_kwargs["max_prompt_length"] = int(grpo_cfg["max_prompt_length"])
    elif "max_length" in config_init_params:
        train_kwargs["max_length"] = int(grpo_cfg["max_prompt_length"])

    if "max_completion_length" in config_init_params:
        train_kwargs["max_completion_length"] = int(grpo_cfg["max_completion_length"])
    elif "max_new_tokens" in config_init_params:
        train_kwargs["max_new_tokens"] = int(grpo_cfg["max_completion_length"])

    filtered_kwargs = {k: v for k, v in train_kwargs.items() if k in config_init_params}
    train_args = GRPOConfig(**filtered_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=train_args,
        train_dataset=dataset,
        peft_config=None if use_existing_adapter else lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
