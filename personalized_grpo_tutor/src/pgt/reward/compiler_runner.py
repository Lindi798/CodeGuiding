from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping


@dataclass
class CompileRunResult:
    compiled: bool
    passed: int
    total: int
    stderr: str

    @property
    def pass_ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def _run_with_input(exe_path: Path, stdin_text: str, timeout_sec: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(exe_path)],
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def compile_and_run_c_code(code: str, tests: Iterable[Mapping[str, str]], timeout_sec: float = 1.5) -> CompileRunResult:
    if not code.strip():
        return CompileRunResult(compiled=False, passed=0, total=len(list(tests)), stderr="empty code")

    tests_list: List[Mapping[str, str]] = list(tests)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        c_file = tmp_dir / "main.c"
        exe_file = tmp_dir / ("main.exe" if os.name == "nt" else "main")
        c_file.write_text(code, encoding="utf-8")

        compile_cmd = ["gcc", str(c_file), "-O2", "-std=c11", "-o", str(exe_file)]
        cp = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
        if cp.returncode != 0:
            return CompileRunResult(compiled=False, passed=0, total=len(tests_list), stderr=cp.stderr[-1000:])

        passed = 0
        stderr_all: List[str] = []
        for t in tests_list:
            try:
                run_p = _run_with_input(exe_file, t.get("input", ""), timeout_sec=timeout_sec)
                expected = (t.get("output", "") or "").strip()
                actual = (run_p.stdout or "").strip()
                if run_p.returncode == 0 and actual == expected:
                    passed += 1
                else:
                    stderr_all.append(run_p.stderr[-300:])
            except subprocess.TimeoutExpired:
                stderr_all.append("timeout")

        return CompileRunResult(
            compiled=True,
            passed=passed,
            total=len(tests_list),
            stderr="\n".join(stderr_all)[-1000:],
        )
