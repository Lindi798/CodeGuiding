import json
import os
from json import JSONDecodeError
from typing import Any, List, Tuple

INPUT_JSON_PATH = "/home/liyongqi/project/personalized_grpo_tutor/data/apps/apps_train.jsonl"
OUTPUT_JSON_PATH = "/home/liyongqi/project/personalized_grpo_tutor/data/process/apps_train_process.jsonl"
BAD_RECORD_LOG_PATH = OUTPUT_JSON_PATH + ".bad_records.log"


def trim_input_output(input_output):
    """提取 input_output 的前三项；不足三项则全部保留。"""
    if isinstance(input_output, list):
        return input_output[:3]
    return input_output


def _strip_control_chars(text: str) -> str:
    """移除会导致 JSON 解析失败的控制字符，保留换行/回车/制表。"""
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return "".join(ch for ch in text if ch in "\n\r\t" or ord(ch) >= 32)


def _is_likely_incomplete_json(err: JSONDecodeError) -> bool:
    msg = err.msg.lower()
    keywords = [
        "unterminated",
        "expecting value",
        "expecting ',' delimiter",
        "expecting property name enclosed in double quotes",
    ]
    return any(k in msg for k in keywords)


def _parse_jsonl_tolerant(text: str) -> Tuple[List[Any], List[str]]:
    """
    容错 JSONL 解析：
    - 允许控制字符（strict=False）
    - 允许单条记录跨多行并自动拼接
    - 对无法修复的坏样本跳过，并返回错误信息
    """
    records: List[Any] = []
    bad_messages: List[str] = []

    lines = text.splitlines()
    buffer = ""
    start_line = 0

    for i, raw_line in enumerate(lines, start=1):
        line = _strip_control_chars(raw_line).strip()
        if not line and not buffer:
            continue

        if buffer:
            buffer = f"{buffer}\n{line}"
        else:
            buffer = line
            start_line = i

        try:
            obj = json.loads(buffer, strict=False)
            records.append(obj)
            buffer = ""
        except JSONDecodeError as e:
            if _is_likely_incomplete_json(e) and i < len(lines):
                continue

            bad_messages.append(f"行 {start_line}-{i} 解析失败: {e}")
            buffer = ""

    if buffer:
        try:
            obj = json.loads(buffer, strict=False)
            records.append(obj)
        except JSONDecodeError as e:
            bad_messages.append(f"行 {start_line}-{len(lines)} 解析失败: {e}")

    return records, bad_messages


def read_json_or_jsonl(path):
    """
    兼容读取 JSON / JSONL：
    - JSONL: 每行一个 JSON 对象（容错）
    - JSON: 顶层可为 list 或 dict
    返回: (raw_data, bad_messages)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = _strip_control_chars(f.read())

    # 先尝试按标准 JSON 解析
    try:
        return json.loads(text, strict=False), []
    except JSONDecodeError:
        pass

    records, bad_messages = _parse_jsonl_tolerant(text)
    if not records:
        preview = bad_messages[0] if bad_messages else "未知错误"
        raise ValueError(f"输入文件无法解析为 JSON 或 JSONL。示例错误: {preview}")
    return records, bad_messages


def extract_records(raw_data):
    """兼容常见数据结构并执行字段抽取。"""
    if isinstance(raw_data, list):
        records = raw_data
    elif isinstance(raw_data, dict):
        if "data" in raw_data and isinstance(raw_data["data"], list):
            records = raw_data["data"]
        elif "samples" in raw_data and isinstance(raw_data["samples"], list):
            records = raw_data["samples"]
        elif "items" in raw_data and isinstance(raw_data["items"], list):
            records = raw_data["items"]
        else:
            raise ValueError("无法识别数据列表位置，请确认 JSON 结构（支持 list / data / samples / items）。")
    else:
        raise ValueError("输入 JSON 顶层必须是 list 或 dict。")

    extracted = []
    for item in records:
        if not isinstance(item, dict):
            continue

        extracted_item = {
            "question": item.get("question"),
            "solutions": item.get("solutions"),
            "input_output": trim_input_output(item.get("input_output")),
            "starter_code": item.get("starter_code"),
        }
        extracted.append(extracted_item)

    return extracted


def write_jsonl(records, path):
    """按 JSONL 写出：每行一个 JSON 对象。"""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_JSON_PATH}")

    raw_data, bad_messages = read_json_or_jsonl(INPUT_JSON_PATH)
    extracted_data = extract_records(raw_data)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    write_jsonl(extracted_data, OUTPUT_JSON_PATH)

    if bad_messages:
        with open(BAD_RECORD_LOG_PATH, "w", encoding="utf-8") as f:
            for msg in bad_messages:
                f.write(msg + "\n")
        print(f"警告: 跳过 {len(bad_messages)} 条异常样本")
        print(f"异常日志: {BAD_RECORD_LOG_PATH}")

    print(f"提取完成，共 {len(extracted_data)} 条样本")
    print(f"输出文件: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
