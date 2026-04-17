from __future__ import annotations

import argparse
import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pgt.training.grpo_trainer import run_grpo


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO for personalized tutor")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT)
    run_grpo(args.config)
