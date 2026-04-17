# Personalized GRPO Tutor (Qwen-7B)

本项目实现一个面向编程教育场景的后训练范式：

1. 冷启动 SFT（分步讲解 + 最终代码）
2. 基于 GRPO 的强化学习（多维奖励）
3. 防 Reward Hacking 的反作弊奖励设计

项目默认基座模型：`Qwen/Qwen2.5-7B-Instruct`

## 1. 功能概览

- 单卡 4090 可运行的 QLoRA 微调方案
- 针对 C 语言 OJ 题的可执行正确性奖励（GCC + 测试用例）
- 逻辑粒度奖励（步骤覆盖）
- 个性化奖励（按学生水平约束解释与代码复杂度）
- Reflective Reward（复杂度分析与代码结构一致性校验）
- Reward Hacking 保护项（只追求最终答案但过程失真时扣分）

## 2. 目录结构

```text
personalized_grpo_tutor/
  configs/
    base.yaml
    sft.yaml
    grpo.yaml
  data/
    sample_sft.jsonl
    sample_rl.jsonl
  scripts/
    run_sft.py
    run_grpo.py
    eval_reward.py
  src/pgt/
    config.py
    data.py
    prompts.py
    reward/
      extract.py
      compiler_runner.py
      reward_model.py
    training/
      utils.py
      sft_trainer.py
      grpo_trainer.py
```

## 3. 安装

```bash
pip install -r requirements.txt
```

建议 CUDA 版本与 PyTorch 对齐，并确保本机可调用 `gcc`。

## 4. 快速开始

### 4.1 SFT

```bash
python scripts/run_sft.py --config configs/sft.yaml
```

### 4.2 GRPO

```bash
python scripts/run_grpo.py --config configs/grpo.yaml
```

### 4.3 奖励函数调试

```bash
python scripts/eval_reward.py --config configs/grpo.yaml --index 0
```

## 5. 数据格式

### 5.1 SFT 数据（JSONL）

每行一个样本，字段如下：

```json
{
  "question": "给定 n，输出 1..n 的和",
  "student_level": "beginner",
  "reasoning_steps": [
    "识别输入输出",
    "确定循环变量",
    "累加并输出"
  ],
  "answer_code": "#include <stdio.h>\\nint main(){...}"
}
```

### 5.2 RL 数据（JSONL）

```json
{
  "problem": "输入 n，输出斐波那契第 n 项",
  "student_level": "beginner",
  "tests": [
    {"input": "5\\n", "output": "5\\n"},
    {"input": "10\\n", "output": "55\\n"}
  ],
  "target_step_count": 4
}
```

## 6. 面试讲述建议

- 冷启动阶段强调“可解释推理模板”的收敛作用。
- GRPO 阶段强调“组内相对优势”在单卡下的显存友好性。
- 奖励设计强调“正确性 + 过程一致性 + 个性化”的多目标平衡。
- 反作弊强调“Complexity Reflection”和“逻辑步骤覆盖”两个闸门。

## 7. 注意事项

- 本项目提供可运行训练骨架，数据规模与超参请根据显存和时长扩展。
- 如果 `trl` 版本变更导致 API 差异，请将版本固定在 `requirements.txt` 中建议范围。
