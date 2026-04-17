from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = "/home/liyongqi/project/personalized_grpo_tutor/data/process/apps_train_process.jsonl"
OUTPUT_SAVE_PATH = "/home/liyongqi/project/personalized_grpo_tutor/data/process/apps_train_sft_stepwise.jsonl"
MODEL_NAME = "qwen3.5-plus-2026-02-15"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV = "DASHSCOPE_API_KEY"
INSTRUCTION = "请教我如何解决这道算法题，我希望你能一步步引导我，不要直接给答案。"
# 跳过前 N 条输入数据（例如设为 10，即从第 11 条开始处理）。
SKIP_ROWS = 370
# 设置为 10 时只处理前 10 条；设为 0 表示处理全部。
PROCESS_LIMIT = 100
MAX_SAMPLES = 0
TEMPERATURE = 0.3
MAX_RETRIES = 3
REQUEST_TIMEOUT = 150
# True 时将新结果追加到已有输出文件末尾；False 时覆盖写入。
APPEND_OUTPUT = True
# True 时每成功转换 1 条就立刻写入输出文件，避免中途中断丢失结果。
REALTIME_WRITE = True

SYSTEM_PROMPT = """# Role
你是一位顶级算法竞赛教练，擅长启发式教学。

# Task
请根据提供的[题目]和[完整参考代码]，将其转化为一份“步进式算法教学”样本。

# Output Constraints (严格遵守)
1. 严禁开篇给答案：禁止在回复的前 50% 字节出现完整可运行的代码。
2. 模块化教学：必须分为以下阶段：
   - 【第一步：题意深度解读】：用通俗易懂的话解释题目在让你干什么，找出潜在约束。
   - 【第二步：核心逻辑推导】：分析问题的分类讨论点。
   - 【第三步：分步代码实现】：讲一个逻辑，写一段代码。严禁一次性给超过 15 行的代码。
   - 【第四步：复杂度与总结】：分析时空复杂度。
3. 代码一致性：分段给出的代码片段，必须能逻辑拼接成最后的完整代码。
4. 回复语言：中文。
5. 代码块规范（必须严格遵守）：
    - 每个代码片段必须放在独立的 fenced code block 中，语言标记必须为 python。
    - 每个代码块前必须有固定标签，按顺序命名为：【局部实现-1】、【局部实现-2】、【局部实现-3】...
    - 严禁在同一个 code block 中混入解释性自然语言。
    - 每个局部实现代码块尽量控制在 15 行以内。
6. 可选汇总块（建议输出）：
    - 在末尾增加【最终代码（可运行）】标签。
    - 该标签下给出一个完整、可运行的 python 代码块，用于自动评测校验。

# Output Format
你必须只输出一个 JSON 对象，且包含以下字段：
- chinese_question: string，题目中文翻译（忠实原文，不要遗漏约束）
- teaching_output: string，符合上述四阶段约束的步进式讲解

# teaching_output 模板（示意）
【第一步：题意深度解读】
...解释...
【第二步：核心逻辑推导】
...解释...
【第三步：分步代码实现】
【局部实现-1】
```python
# 仅该步骤对应代码
```
【局部实现-2】
```python
# 仅该步骤对应代码
```
【第四步：复杂度与总结】
...解释...
【最终代码（可运行）】
```python
# 完整可运行代码
```
"""

def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_solution(item: Dict) -> str:
    solutions = item.get("solutions")
    if isinstance(solutions, list) and solutions:
        first = solutions[0]
        return first if isinstance(first, str) else str(first)
    if isinstance(solutions, str):
        return solutions
    return ""


def build_user_prompt(question_en: str, solution_code: str) -> str:
    return (
        "[Question-English]\n"
        f"{question_en}\n\n"
        "[Reference-Solution-Python]\n"
        f"{solution_code}\n"
    )


def call_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> Dict:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Model response content is not a string")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON object: {e}\nRaw: {content[:500]}") from e

    if not isinstance(parsed, dict):
        raise ValueError("Model JSON output is not an object")

    return parsed


def call_with_retry(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
) -> Dict:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return call_chat_completions(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
            last_err = e
            if attempt == max_retries:
                break
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_err}")


def convert_one(item: Dict, api_key: str) -> Dict:
    question_en = str(item.get("question", "")).strip()
    code = first_solution(item).strip()
    if not question_en or not code:
        raise ValueError("Missing required fields: question/solutions")

    user_prompt = build_user_prompt(question_en, code)
    result = call_with_retry(
        base_url=BASE_URL,
        api_key=api_key,
        model=MODEL_NAME,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=TEMPERATURE,
        max_retries=MAX_RETRIES,
    )

    chinese_question = str(result.get("chinese_question", "")).strip()
    teaching_output = str(result.get("teaching_output", "")).strip()
    if not chinese_question or not teaching_output:
        raise ValueError("Model output missing chinese_question or teaching_output")

    sft_item = {
        "instruction": INSTRUCTION,
        "input": f"题目描述（中文）：{chinese_question}",
        "output": teaching_output,
        "meta": {
            "source": "apps_extracted",
            "model": MODEL_NAME,
        },
    }
    return sft_item


def main() -> None:
    os.chdir(ROOT)

    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Please set env var {API_KEY_ENV} before running."
        )

    in_path = Path(DEFAULT_INPUT)
    out_path = Path(OUTPUT_SAVE_PATH)
    rows = read_jsonl(in_path)

    if SKIP_ROWS > 0:
        rows = rows[SKIP_ROWS:]

    limit = PROCESS_LIMIT if PROCESS_LIMIT > 0 else MAX_SAMPLES
    if limit > 0:
        rows = rows[:limit]

    converted: List[Dict] = []
    failed: List[Dict] = []

    if REALTIME_WRITE and not APPEND_OUTPUT:
        # 覆盖模式下先清空文件，再逐条实时写入。
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8"):
            pass

    for idx, item in enumerate(tqdm(rows, desc="Converting to stepwise SFT"), start=1):
        try:
            sft_item = convert_one(item, api_key)
            converted.append(sft_item)
            if REALTIME_WRITE:
                append_jsonl([sft_item], out_path)
        except Exception as e:  # noqa: BLE001
            source_index = SKIP_ROWS + idx
            failed.append({"index": source_index, "error": str(e)})

    if not REALTIME_WRITE:
        if APPEND_OUTPUT:
            append_jsonl(converted, out_path)
        else:
            write_jsonl(converted, out_path)

    print(f"Converted: {len(converted)}")
    print(f"Output: {out_path}")
    print(f"Failed: {len(failed)}")
    print(f"Skip rows: {SKIP_ROWS}")
    print(f"Append output: {APPEND_OUTPUT}")
    print(f"Realtime write: {REALTIME_WRITE}")

    if failed:
        err_path = out_path.with_suffix(out_path.suffix + ".errors.jsonl")
        write_jsonl(failed, err_path)
        print(f"Failure log: {err_path}")


if __name__ == "__main__":
    main()
