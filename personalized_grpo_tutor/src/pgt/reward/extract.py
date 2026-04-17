from __future__ import annotations

import json
import re
from typing import Optional


_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_CODE_BLOCK_WITH_LANG_RE = re.compile(r"```([a-zA-Z0-9_+\-]*)\s*(.*?)```", re.DOTALL)
_COMPLEXITY_RE = re.compile(r"Time\s*:\s*(O\([^\n]+\))", re.IGNORECASE)
_STEP_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_FINAL_CODE_MARKER = "【最终代码（可运行）】"


def _extract_text_payload(raw_text: str) -> str:
    text = raw_text or ""
    stripped = text.strip()
    if not stripped:
        return ""

    # If input is a JSON object like {"text": "...", "meta": {...}}, use the text field.
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                return obj["text"]
        except Exception:
            pass

    return text


def extract_c_code(text: str) -> str:
    _, code = extract_code_with_lang(text, prefer_lang="c")
    return code


def extract_code_with_lang(text: str, prefer_lang: str = "python") -> tuple[str, str]:
    payload = _extract_text_payload(text)
    if not payload:
        return "", ""

    pref = (prefer_lang or "").strip().lower()

    def pick_from(region_text: str) -> tuple[str, str]:
        matches = list(_CODE_BLOCK_WITH_LANG_RE.finditer(region_text))
        if not matches:
            return "", ""

        if pref:
            preferred = [m for m in matches if (m.group(1) or "").strip().lower().startswith(pref)]
            if preferred:
                chosen = preferred[-1]
                return (chosen.group(1) or "").strip().lower(), (chosen.group(2) or "").strip()

        chosen = matches[-1]
        return (chosen.group(1) or "").strip().lower(), (chosen.group(2) or "").strip()

    search_text = payload
    marker_pos = payload.find(_FINAL_CODE_MARKER)
    if marker_pos != -1:
        search_text = payload[marker_pos + len(_FINAL_CODE_MARKER):]

    lang, code = pick_from(search_text)
    if code:
        return lang, code

    if search_text is not payload:
        return pick_from(payload)

    return "", ""


def extract_step_count(text: str) -> int:
    if not text:
        return 0
    return len(_STEP_RE.findall(text))


def extract_claimed_time_complexity(text: str) -> Optional[str]:
    m = _COMPLEXITY_RE.search(text or "")
    if not m:
        return None
    return m.group(1).replace(" ", "")


def estimate_time_complexity_from_code(code: str) -> str:
    if not code:
        return "O(?)"

    normalized = code.lower()
    loop_count = len(re.findall(r"\b(for|while)\b", normalized))

    if loop_count == 0:
        return "O(1)"
    if loop_count == 1:
        return "O(n)"
    return "O(n^2)"
