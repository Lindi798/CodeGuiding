from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = "/home/liyongqi/project/personalized_grpo_tutor/data/process/grpo_batch"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(rows: Iterable[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_submit_meta(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid submit meta format: {path}")
    return payload


def get_batch_status(*, base_url: str, api_key: str, batch_id: str) -> Dict:
    url = base_url.rstrip("/") + f"/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Query batch status failed: {resp.status_code} {resp.text[:500]}")
    return resp.json()


def wait_for_batch_completion(
    *,
    base_url: str,
    api_key: str,
    batch_id: str,
    poll_interval_sec: int,
    max_wait_sec: int,
) -> Dict:
    start = time.time()
    while True:
        payload = get_batch_status(base_url=base_url, api_key=api_key, batch_id=batch_id)
        status = str(payload.get("status", "")).lower()
        if status in {"completed", "failed", "expired", "cancelled"}:
            return payload
        if max_wait_sec > 0 and (time.time() - start) > max_wait_sec:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}, last status={status}")
        print(f"batch {batch_id} status={status}, waiting {poll_interval_sec}s...")
        time.sleep(max(1, poll_interval_sec))


def download_file_content(*, base_url: str, api_key: str, file_id: str) -> str:
    url = base_url.rstrip("/") + f"/files/{file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=120)
    if resp.status_code >= 300:
        raise RuntimeError(f"Download file content failed: {resp.status_code} {resp.text[:500]}")
    return resp.text


def parse_result_line(row: Mapping[str, object]) -> Optional[tuple[str, str]]:
    custom_id = str(row.get("custom_id", "")).strip()
    if not custom_id:
        return None

    response = row.get("response")
    if not isinstance(response, Mapping):
        return None

    status_code = int(response.get("status_code", 0) or 0)
    if status_code >= 300:
        return None

    body = response.get("body")
    if not isinstance(body, Mapping):
        return None

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        return None

    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        return None

    content = message.get("content")
    if not isinstance(content, str):
        return None

    translated_question = ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, Mapping):
            translated_question = str(parsed.get("question", "")).strip()
    except Exception:
        translated_question = content.strip()

    if not translated_question:
        return None

    return custom_id, translated_question


def merge_results(
    *,
    manifest_rows: List[Dict],
    batch_result_rows: List[Dict],
) -> tuple[List[Dict], List[Dict]]:
    manifest_by_id: Dict[str, Dict] = {}
    for row in manifest_rows:
        custom_id = str(row.get("custom_id", "")).strip()
        if custom_id:
            manifest_by_id[custom_id] = row

    translated_by_id: Dict[str, str] = {}
    for row in batch_result_rows:
        parsed = parse_result_line(row)
        if parsed is None:
            continue
        custom_id, question_zh = parsed
        translated_by_id[custom_id] = question_zh

    merged: List[Dict] = []
    missing: List[Dict] = []

    for custom_id, item in manifest_by_id.items():
        translated = translated_by_id.get(custom_id, "").strip()
        if not translated:
            missing.append({"custom_id": custom_id, "reason": "no_translation_in_batch_result"})
            continue

        merged.append(
            {
                "instruction": item.get("instruction", ""),
                "question": translated,
                "input_output": item.get("input_output", {}),
                "meta": {
                    "custom_id": custom_id,
                    "source_index": item.get("source_index"),
                    "question_en": item.get("question_en", ""),
                },
            }
        )

    return merged, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download completed batch output and merge with manifest to build final grpo.jsonl"
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submit-meta", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--wait", action="store_true", help="Wait until batch reaches terminal state")
    parser.add_argument("--poll-interval-sec", type=int, default=20)
    parser.add_argument("--max-wait-sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--batch-id", type=str, default=None, help="Optional override for batch id")
    parser.add_argument("--base-url", type=str, default=None, help="Optional override for base url")
    parser.add_argument("--output-file", type=str, default=None, help="Optional override for final grpo jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    out_dir = Path(args.output_dir)
    submit_meta_path = Path(args.submit_meta) if args.submit_meta else (out_dir / "batch_submit_result.json")
    manifest_path = Path(args.manifest) if args.manifest else (out_dir / "manifest.jsonl")

    if not submit_meta_path.exists():
        raise FileNotFoundError(f"submit meta not found: {submit_meta_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    submit_meta = load_submit_meta(submit_meta_path)
    batch_obj = submit_meta.get("batch", {})
    if not isinstance(batch_obj, Mapping):
        raise ValueError("Invalid batch object in submit meta")

    base_url = str(args.base_url or submit_meta.get("base_url") or "").strip()
    if not base_url:
        raise ValueError("Missing base_url. Provide --base-url or ensure submit meta contains base_url")

    batch_id = str(args.batch_id or batch_obj.get("id") or "").strip()
    if not batch_id:
        raise ValueError("Missing batch id. Provide --batch-id or ensure submit meta contains batch.id")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key env: {args.api_key_env}")

    current_batch = batch_obj
    status = str(current_batch.get("status", "")).lower()
    if args.wait and status not in {"completed", "failed", "expired", "cancelled"}:
        current_batch = wait_for_batch_completion(
            base_url=base_url,
            api_key=api_key,
            batch_id=batch_id,
            poll_interval_sec=int(args.poll_interval_sec),
            max_wait_sec=int(args.max_wait_sec),
        )
        status = str(current_batch.get("status", "")).lower()

    if status != "completed":
        raise RuntimeError(f"Batch is not completed. current status={status}")

    output_file_id = str(current_batch.get("output_file_id") or "").strip()
    if not output_file_id:
        raise RuntimeError("Batch completed but output_file_id is empty")

    result_text = download_file_content(
        base_url=base_url,
        api_key=api_key,
        file_id=output_file_id,
    )

    batch_result_path = out_dir / "batch_result.jsonl"
    batch_result_path.write_text(result_text, encoding="utf-8")

    batch_result_rows = read_jsonl(batch_result_path)
    manifest_rows = read_jsonl(manifest_path)
    merged_rows, missing_rows = merge_results(
        manifest_rows=manifest_rows,
        batch_result_rows=batch_result_rows,
    )

    final_output_path = Path(args.output_file) if args.output_file else (out_dir / "grpo.jsonl")
    write_jsonl(merged_rows, final_output_path)

    missing_path = out_dir / "grpo_missing.jsonl"
    write_jsonl(missing_rows, missing_path)

    status_dump_path = out_dir / "batch_status_latest.json"
    status_dump_path.write_text(json.dumps(dict(current_batch), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"batch_id: {batch_id}")
    print(f"status: {status}")
    print(f"output_file_id: {output_file_id}")
    print(f"downloaded result: {batch_result_path}")
    print(f"final grpo: {final_output_path}")
    print(f"merged: {len(merged_rows)}")
    print(f"missing: {len(missing_rows)}")
    print(f"missing file: {missing_path}")


if __name__ == "__main__":
    main()
