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

from pgt.config import load_yaml_with_extends
from pgt.data import read_jsonl
from pgt.reward.reward_model import compute_reward


DEMO_COMPLETION = """[Reasoning]
1. 读取输入 n。
2. 初始化 a=0,b=1，循环更新。
3. 当 n==0 输出 0，否则输出第 n 项。
[Complexity]
Time: O(n)
Space: O(1)
[Code]
```c
#include <stdio.h>

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n == 0) {
        printf("0\\n");
        return 0;
    }
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int c = a + b;
        a = b;
        b = c;
    }
    printf("%d\\n", n == 1 ? 1 : b);
    return 0;
}
```
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reward on a sample")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(ROOT)

    cfg = load_yaml_with_extends(args.config)
    rl_data = read_jsonl(cfg["paths"]["rl_data"])

    sample = rl_data[args.index]
    bd = compute_reward(DEMO_COMPLETION, sample, cfg["reward"])

    print("Reward breakdown:")
    print(f"accuracy: {bd.accuracy:.4f}")
    print(f"steps: {bd.steps:.4f}")
    print(f"personalized: {bd.personalized:.4f}")
    print(f"reflective: {bd.reflective:.4f}")
    print(f"anti_hacking: {bd.anti_hacking:.4f}")
    print(f"total: {bd.total:.4f}")
