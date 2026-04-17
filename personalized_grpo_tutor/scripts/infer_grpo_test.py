from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# =========================
# Hardcoded runtime config
# =========================
BASE_MODEL = "/home/liyongqi/models/Qwen2.5-7B-Instruct"
# 改成你训练完成后的 GRPO adapter 目录
ADAPTER_PATH = ""
# 测试集路径（支持 jsonl / json(list)）
TEST_DATA_PATH = "/home/liyongqi/project/personalized_grpo_tutor/data/process/grpo_batch/grpo.jsonl"
OUTPUT_PATH = "/home/liyongqi/project/personalized_grpo_tutor/outputs/base_test_predictions.jsonl"
MAX_SAMPLES = 10

USE_BF16 = True
MAX_NEW_TOKENS = 768
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9

# 若为 True，会从模型输出提取代码并在 input_output 上跑测试，记录 accuracy
EVAL_CODE_ACCURACY = True
PY_TIMEOUT_SEC = 2.0

SYSTEM_PROMPT = (
    "你是一位顶级算法竞赛教练，擅长启发式教学。"
    "请输出步进式讲解并在末尾给出【最终代码（可运行）】的 python 代码块。"
)

# Keep the same instruction as training-time GRPO data construction.
FIXED_INSTRUCTION = "请教我如何解决这道算法题，我希望你能一步步引导我，不要直接给答案。"


def read_json_or_jsonl(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list json: {path}")
        return [x for x in data if isinstance(x, dict)]

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def build_user_text(item: Mapping) -> str:
    instruction = FIXED_INSTRUCTION
    input_text = str(item.get("input", item.get("question", item.get("problem", "")))).strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def extract_python_code_from_output(text: str) -> str:
    marker = "【最终代码（可运行）】"
    region = text or ""
    pos = region.find(marker)
    if pos != -1:
        region = region[pos + len(marker):]

    # 优先找 python fenced code block 的最后一个
    blocks: List[Tuple[str, str]] = []
    start = 0
    while True:
        i = region.find("```", start)
        if i == -1:
            break
        j = region.find("\n", i + 3)
        if j == -1:
            break
        lang = region[i + 3:j].strip().lower()
        k = region.find("```", j + 1)
        if k == -1:
            break
        code = region[j + 1:k].strip()
        blocks.append((lang, code))
        start = k + 3

    py_blocks = [code for lang, code in blocks if lang.startswith("py")]
    if py_blocks:
        return py_blocks[-1]
    if blocks:
        return blocks[-1][1]
    return ""


def build_tests(item: Mapping) -> List[Dict[str, str]]:
    tests = item.get("tests")
    if isinstance(tests, list):
        return [
            {"input": str(t.get("input", "")), "output": str(t.get("output", ""))}
            for t in tests
            if isinstance(t, Mapping)
        ]

    input_output = item.get("input_output", {})
    if not isinstance(input_output, Mapping):
        return []

    if isinstance(input_output.get("inputs"), list) and isinstance(input_output.get("outputs"), list):
        inps = input_output.get("inputs", [])
        outs = input_output.get("outputs", [])
        return [{"input": str(i), "output": str(o)} for i, o in zip(inps, outs)]

    if "input" in input_output and "output" in input_output:
        return [{"input": str(input_output.get("input", "")), "output": str(input_output.get("output", ""))}]

    return []


def run_python_tests(code: str, tests: List[Mapping[str, str]], timeout_sec: float) -> float:
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
                if cp.returncode == 0 and expected == actual:
                    passed += 1
            except subprocess.TimeoutExpired:
                continue

    return passed / float(total) if total > 0 else 0.0


def main() -> None:
    test_path = Path(TEST_DATA_PATH)
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_json_or_jsonl(test_path)
    rows = rows[:MAX_SAMPLES]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if USE_BF16:
        model_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    adapter_path = str(ADAPTER_PATH or "").strip()
    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but ADAPTER_PATH is set.")
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        print(f"Using adapter: {adapter_path}")
    else:
        model = base_model
        print("ADAPTER_PATH is empty, using base model only.")
    model.eval()

    total_acc = 0.0
    valid_acc_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(rows):
            user_text = build_user_text(item)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer(prompt, return_tensors="pt")
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

            gen_kwargs = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": DO_SAMPLE,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if DO_SAMPLE:
                gen_kwargs["temperature"] = TEMPERATURE
                gen_kwargs["top_p"] = TOP_P

            with torch.no_grad():
                output_ids = model.generate(**model_inputs, **gen_kwargs)
            completion_ids = output_ids[0, model_inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            record: Dict = {
                "index": idx,
                "instruction": FIXED_INSTRUCTION,
                "input": item.get("input", item.get("question", item.get("problem", ""))),
                "prediction": completion,
            }

            if EVAL_CODE_ACCURACY:
                tests = build_tests(item)
                code = extract_python_code_from_output(completion)
                acc = run_python_tests(code=code, tests=tests, timeout_sec=PY_TIMEOUT_SEC)
                record["accuracy"] = acc
                if tests:
                    total_acc += acc
                    valid_acc_count += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0:
                print(f"processed: {idx + 1}/{len(rows)}")

    print(f"Saved predictions: {out_path}")
    if EVAL_CODE_ACCURACY and valid_acc_count > 0:
        print(f"Mean accuracy on samples with tests: {total_acc / valid_acc_count:.4f}")


if __name__ == "__main__":
    main()
