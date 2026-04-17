from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_PATH = "/home/liyongqi/project/personalized_grpo_tutor/data/process/apps_train_process.jsonl"
DEFAULT_BASE_DIR = "/home/liyongqi/project/personalized_grpo_tutor/data/process/grpo_batch"
DEFAULT_MODEL = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_SEED = 42
DEFAULT_INSTRUCTION = "请教我如何解决这道算法题，我希望你能一步步引导我，不要直接给答案。"

TRANSLATION_SYSTEM_PROMPT = (
    "你是算法题翻译助手。"
    "请把用户给出的英文算法题完整翻译成中文，保留输入输出格式、约束、示例，不要删减信息。"
    "只返回 JSON 对象，格式为: {\"question\": \"...\"}。"
)


@dataclass
class PipelineConfig:
    base_dir: Path
    source_path: Path
    model: str
    base_url: str
    api_key: str
    sample_size: int
    seed: int
    instruction: str

    request_jsonl: Path
    manifest_jsonl: Path
    submit_result_json: Path
    batch_result_jsonl: Path
    merged_grpo_jsonl: Path
    merge_missing_jsonl: Path
    parsed_map_json: Path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_config(args: argparse.Namespace) -> PipelineConfig:
    base_dir = Path(args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source).resolve() if Path(args.source).is_absolute() else (ROOT / args.source).resolve()
    api_key = args.api_key.strip() if args.api_key else os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key. Use --api-key or set env {args.api_key_env}")

    return PipelineConfig(
        base_dir=base_dir,
        source_path=source_path,
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        instruction=args.instruction,
        request_jsonl=base_dir / "request.jsonl",
        manifest_jsonl=base_dir / "manifest.jsonl",
        submit_result_json=base_dir / "batch_submit_result.json",
        batch_result_jsonl=base_dir / "batch_results.jsonl",
        merged_grpo_jsonl=base_dir / "grpo.jsonl",
        merge_missing_jsonl=base_dir / "grpo_missing.jsonl",
        parsed_map_json=base_dir / "results_map.json",
    )


def make_client(config: PipelineConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def sample_rows(rows: List[Dict[str, Any]], sample_size: int, seed: int) -> List[Tuple[int, Dict[str, Any]]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be > 0")
    if not rows:
        raise ValueError("source dataset is empty")

    sample_size = min(sample_size, len(rows))
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:sample_size])
    return [(idx, rows[idx]) for idx in selected]


def build_requests(config: PipelineConfig) -> None:
    rows = read_jsonl(config.source_path)
    sampled = sample_rows(rows, config.sample_size, config.seed)

    requests_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for local_idx, (src_index, item) in enumerate(sampled):
        question_en = str(item.get("question", "")).strip()
        if not question_en:
            continue

        custom_id = f"grpo_{local_idx:06d}_src_{src_index}"
        requests_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": config.model,
                    "temperature": 0,
                    "enable_thinking": False,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                        {"role": "user", "content": question_en},
                    ],
                },
            }
        )

        manifest_rows.append(
            {
                "custom_id": custom_id,
                "source_index": src_index,
                "instruction": config.instruction,
                "question_en": question_en,
                "input_output": item.get("input_output", {}),
            }
        )

    if not requests_rows:
        raise ValueError("No valid rows with question field found after sampling")

    write_jsonl(config.request_jsonl, requests_rows)
    write_jsonl(config.manifest_jsonl, manifest_rows)

    print(f"sampled: {len(sampled)}")
    print(f"requests: {len(requests_rows)}")
    print(f"request file: {config.request_jsonl}")
    print(f"manifest file: {config.manifest_jsonl}")


def submit_batch(config: PipelineConfig) -> str:
    if not config.request_jsonl.exists():
        raise FileNotFoundError(f"request file not found: {config.request_jsonl}")

    client = make_client(config)

    with config.request_jsonl.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"task": "translate_question_for_grpo_dataset"},
    )

    payload = {
        "base_url": config.base_url,
        "input_file_id": input_file.id,
        "batch": batch.model_dump() if hasattr(batch, "model_dump") else dict(batch),
        "request_file": str(config.request_jsonl),
        "manifest_file": str(config.manifest_jsonl),
    }
    write_json(config.submit_result_json, payload)

    print(f"uploaded file_id: {input_file.id}")
    print(f"batch_id: {batch.id}")
    print(f"status: {getattr(batch, 'status', 'unknown')}")
    print(f"submit meta: {config.submit_result_json}")
    return str(batch.id)


def poll_batch(config: PipelineConfig, batch_id: str, interval_sec: int = 60) -> None:
    client = make_client(config)
    while True:
        job = client.batches.retrieve(batch_id)
        status = str(getattr(job, "status", "unknown"))
        counts = getattr(job, "request_counts", None)
        print(f"batch_id={batch_id} status={status} request_counts={counts}")
        if status in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(max(1, interval_sec))


def download_results(config: PipelineConfig, batch_id: str) -> None:
    client = make_client(config)
    job = client.batches.retrieve(batch_id)
    status = str(getattr(job, "status", ""))
    if status != "completed":
        raise RuntimeError(f"Batch not completed. status={status}")

    output_file_id = getattr(job, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError("Batch output_file_id is empty")

    content = client.files.content(output_file_id)
    with config.batch_result_jsonl.open("wb") as f:
        f.write(content.read())

    latest_status = job.model_dump() if hasattr(job, "model_dump") else dict(job)
    write_json(config.base_dir / "batch_status_latest.json", latest_status)

    print(f"output_file_id: {output_file_id}")
    print(f"saved batch results: {config.batch_result_jsonl}")


def parse_results(config: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    if not config.batch_result_jsonl.exists():
        raise FileNotFoundError(f"batch result file not found: {config.batch_result_jsonl}")

    result_map: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(config.batch_result_jsonl):
        custom_id = str(row.get("custom_id", "")).strip()
        if not custom_id:
            continue

        response = row.get("response", {})
        if not isinstance(response, Mapping):
            continue

        status_code = int(response.get("status_code", 0) or 0)
        if status_code >= 300:
            continue

        body = response.get("body", {})
        if not isinstance(body, Mapping):
            continue

        choices = body.get("choices", [])
        if not isinstance(choices, list) or not choices:
            continue

        msg = choices[0].get("message", {}) if isinstance(choices[0], Mapping) else {}
        content = msg.get("content", "") if isinstance(msg, Mapping) else ""
        if not isinstance(content, str) or not content.strip():
            continue

        translated = ""
        try:
            parsed = json.loads(content)
            if isinstance(parsed, Mapping):
                translated = str(parsed.get("question", "")).strip()
        except json.JSONDecodeError:
            translated = content.strip()

        if translated:
            result_map[custom_id] = {"question": translated}

    write_json(config.parsed_map_json, result_map)
    print(f"parsed map size: {len(result_map)}")
    print(f"parsed map: {config.parsed_map_json}")
    return result_map


def merge_results(config: PipelineConfig) -> None:
    if not config.manifest_jsonl.exists():
        raise FileNotFoundError(f"manifest not found: {config.manifest_jsonl}")

    results_map = parse_results(config)
    manifest_rows = read_jsonl(config.manifest_jsonl)

    merged_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []

    for item in manifest_rows:
        custom_id = str(item.get("custom_id", "")).strip()
        translated = str(results_map.get(custom_id, {}).get("question", "")).strip()
        if not translated:
            missing_rows.append({"custom_id": custom_id, "reason": "missing_translation"})
            continue

        merged_rows.append(
            {
                # Use unified instruction for all GRPO records.
                "instruction": config.instruction,
                "input": translated,
                "input_output": item.get("input_output", {}),
                "meta": {
                    "custom_id": custom_id,
                    "source_index": item.get("source_index"),
                    "question_en": item.get("question_en", ""),
                },
            }
        )

    write_jsonl(config.merged_grpo_jsonl, merged_rows)
    write_jsonl(config.merge_missing_jsonl, missing_rows)

    print(f"merged rows: {len(merged_rows)}")
    print(f"missing rows: {len(missing_rows)}")
    print(f"final grpo: {config.merged_grpo_jsonl}")
    print(f"missing file: {config.merge_missing_jsonl}")


def validate(config: PipelineConfig) -> None:
    if not config.merged_grpo_jsonl.exists():
        raise FileNotFoundError(f"merged grpo not found: {config.merged_grpo_jsonl}")

    rows = read_jsonl(config.merged_grpo_jsonl)
    total = len(rows)
    bad = 0
    for row in rows:
        ok = (
            isinstance(row.get("instruction"), str)
            and isinstance(row.get("input"), str)
            and isinstance(row.get("input_output"), Mapping)
        )
        if not ok:
            bad += 1

    report = {
        "total": total,
        "invalid_rows": bad,
        "valid_rows": total - bad,
    }
    report_path = config.base_dir / "grpo_validate_report.json"
    write_json(report_path, report)

    print(f"validate total: {total}")
    print(f"validate invalid: {bad}")
    print(f"report: {report_path}")


def infer_batch_id_from_submit_meta(config: PipelineConfig) -> str:
    if not config.submit_result_json.exists():
        raise FileNotFoundError(f"submit meta not found: {config.submit_result_json}")
    data = read_json(config.submit_result_json)
    if not isinstance(data, Mapping):
        raise ValueError("invalid submit meta")
    batch = data.get("batch", {})
    if not isinstance(batch, Mapping):
        raise ValueError("invalid batch object in submit meta")
    batch_id = str(batch.get("id", "")).strip()
    if not batch_id:
        raise ValueError("batch id missing in submit meta")
    return batch_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO batch translation pipeline")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--source", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "build_requests",
            "submit_batch",
            "poll_batch",
            "download_results",
            "merge_results",
            "validate",
            "all",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)

    try:
        config = build_config(args)
    except Exception as e:  # noqa: BLE001
        print(f"config error: {e}")
        return 2

    try:
        if args.step == "build_requests":
            build_requests(config)
        elif args.step == "submit_batch":
            submit_batch(config)
        elif args.step == "poll_batch":
            batch_id = args.batch_id.strip() or infer_batch_id_from_submit_meta(config)
            poll_batch(config, batch_id=batch_id, interval_sec=int(args.poll_interval))
        elif args.step == "download_results":
            batch_id = args.batch_id.strip() or infer_batch_id_from_submit_meta(config)
            download_results(config, batch_id=batch_id)
        elif args.step == "merge_results":
            merge_results(config)
        elif args.step == "validate":
            validate(config)
        elif args.step == "all":
            build_requests(config)
            batch_id = submit_batch(config)
            print(f"submit done. batch_id={batch_id}")
        else:
            print("unknown step")
            return 2
    except Exception as e:  # noqa: BLE001
        print(f"pipeline error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
