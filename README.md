# CodeLLM
# Personalized GRPO Tutor (Qwen-7B)

## 项目概述

**Personalized GRPO Tutor** 是一个基于强化学习的个性化算法教学系统，通过两阶段训练（SFT + GRPO）让 LLM 学会用"步进式教学"的方式引导学生解决算法题。

该项目针对 C 语言 OJ 竞赛题设计，支持多学生水平个性化讲解，并集成代码执行验证、复杂度分析引导等教学能力。

---

## 项目背景

在传统在线编程教育中，大多数 AI 辅导系统直接给出完整答案，学生获不到启发。本项目的目标是：

1. **格式规范**：教会模型遵循"【第一步：题意解读】→【第二步：逻辑推导】→【第三步：分步代码】→【第四步：复杂度分析】"的教学框架
2. **代码正确性**：通过 GRPO 强化学习，优化生成代码的执行准确率和步骤匹配度
3. **奖励对齐**：构建多维度奖励函数（代码准确率、步骤数量、通过奖励），防止 Reward Hacking

---

## 技术方法

### 1. SFT 阶段（监督微调）- 教格式

- **目标**：通过监督学习，让模型学会"步进式讲解"的规范格式
- **数据来源**：使用 Qwen 3.5 API 将原始 APPS 题目转化为步进式教学样本
- **约束机制**：
  - 回复前 50% 禁止出现完整代码
  - 必须分 4 个教学阶段，每个代码块限制 ≤15 行
  - 强制在末尾给出【最终代码（可运行）】

### 2. GRPO 阶段（强化学习）- 优化正确性

- **基础**：GRPO（Group Reward Policy Optimization）
- **奖励函数**（多维度）：
  ```
  total_reward = w_accuracy × code_accuracy 
               + w_steps × step_ratio 
               + pass_bonus (accuracy >= 99%)
  ```
- **配置参数**：
  - `w_accuracy: 1.2` - 代码执行准确率权重
  - `w_steps: 0.4` - 步骤数匹配权重
  - `pass_bonus: 0.1` - 完全通过的额外奖励

### 3. 训练流程

```
APPS 数据集
    ↓
[SFT 数据处理] → 调用 Qwen API 生成步进式讲解
    ↓
apps_train_sft_stepwise.jsonl (SFT 训练数据)
    ↓
[SFT 训练] → Qwen-7B + LoRA 微调
    ↓
sft_adapter (已保存)
    ↓
[GRPO 数据准备] → 提取优质题目和测试用例
    ↓
grpo.jsonl (GRPO 训练数据)
    ↓
[GRPO 训练] → 加载 SFT Adapter + GRPO 优化
    ↓
grpo_adapter (最终模型)
```

---

## 项目数据来源

### 基础数据集：APPS (Algorithm Problem Programming Set)

- **来源**：[APPS: BENCHMARK FOR CODE GENERATION FROM NATURAL LANGUAGE](https://www.kaggle.com/datasets/google-code-competitions/code-competitions)
- **数据规模**：
  - 训练集：~10,000 个编程问题
  - 测试集：题目分类覆盖离散数学、计算几何、图论等
- **数据格式**：每个问题包含题目描述、多个参考解答、输入输出测试用例
- **语言覆盖**：Python, C, C++, Java 等

### 数据流水线

#### Step 1: 原始 APPS 数据处理 (`data/extract.py`)

从 Kaggle APPS 数据集中提取并标准化数据结构：

```
输入：apps/apps_train.jsonl (raw data)
处理流程：
  ├─ 容错 JSON/JSONL 解析（处理编码、控制字符）
  ├─ 提取题目、多个参考代码、前 3 条测试用例
  ├─ 重构为统一格式（instruction, input, output, tests）
  └─ 生成中间文件
输出：data/process/apps_train_process.jsonl
```

#### Step 2: SFT 步进式讲解生成 (`data/build_sft_stepwise.py`)

调用 Qwen 3.5 API 将原始问题和参考代码转化为步进式教学样本：

```
输入：apps_train_process.jsonl
处理流程：
  ├─ 逐条读取题目和参考代码
  ├─ 调用 Qwen API（Model: qwen3.5-plus-2026-02-15）
  │   使用精心设计的 SYSTEM_PROMPT 约束：
  │   - 禁止开篇代码
  │   - 强制 4 阶段教学框架
  │   - 代码片段命名规范（【局部实现-1】等）
  │   - 末尾必须有【最终代码（可运行）】
  ├─ 实时写入（防止中断丢失）
  └─ 失败重试（最多 3 次）
输出：data/process/apps_train_sft_stepwise.jsonl

配置参数：
  - TEMPERATURE: 0.3 (保持高质量一致性)
  - REQUEST_TIMEOUT: 150s
  - SKIP_ROWS: 370 (支持断点续传)
  - PROCESS_LIMIT: 100 (当前处理 100 条)
```

#### Step 3: GRPO 数据准备 (`data/process/grpo_batch/grpo.jsonl`)

从处理后的 SFT 数据中提取用于强化学习的数据：

```
字段说明：
  - problem: 题目描述（用于生成模型输入）
  - student_level: 学生水平 (beginner/intermediate/advanced)
  - target_step_count: 目标讲解步数（评估步骤匹配度）
  - tests: List[{input, output}] - 用于代码执行验证
  - prompt: 完整的用户提示词
  - system: 根据 student_level 定制的系统提示词
```

---

## 文件结构与说明

```
personalized_grpo_tutor/
├── README.md                              # 项目说明文档
├── requirements.txt                       # Python 依赖（见下表）
│
├── configs/                               # 训练配置文件（YAML）
│   ├── base.yaml                          # 基础配置（共同参数）
│   │   ├─ model.base_model: Qwen2.5-7B-Instruct
│   │   ├─ model.use_4bit: false (标准 LoRA 微调)
│   │   ├─ lora.r: 32 (LoRA rank)
│   │   └─ lora.target_modules: [q_proj, k_proj, ...] (7 个)
│   ├── sft.yaml                           # SFT 阶段配置
│   │   ├─ num_train_epochs: 1
│   │   ├─ per_device_train_batch_size: 1
│   │   ├─ gradient_accumulation_steps: 8
│   │   ├─ learning_rate: 2.0e-4
│   │   ├─ save_steps: 100
│   │   └─ paths.sft_data: apps_train_sft_stepwise.jsonl
│   └── grpo.yaml                          # GRPO 阶段配置
│       ├─ num_train_epochs: 1
│       ├─ per_device_train_batch_size: 1
│       ├─ gradient_accumulation_steps: 4
│       ├─ learning_rate: 1.0e-5
│       ├─ num_generations: 8 (每个 prompt 生成 8 个候选)
│       ├─ beta: 0.04 (KL 散度系数)
│       ├─ reward.w_accuracy: 1.2
│       ├─ reward.w_steps: 0.4
│       └─ reward.pass_bonus: 0.1
│
├── data/                                  # 数据处理模块
│   ├── extract.py                         # APPS 原始数据提取与标准化
│   │   ├─ read_json_or_jsonl: 容错 JSON/JSONL 读取
│   │   ├─ trim_input_output: 提取前 3 条测试用例
│   │   └─ 处理编码和控制字符问题
│   │
│   ├── build_sft_stepwise.py              # 调用 Qwen API 生成 SFT 数据
│   │   ├─ 使用精心设计的 SYSTEM_PROMPT
│   │   ├─ 支持断点续传（SKIP_ROWS, APPEND_OUTPUT）
│   │   ├─ 实时写入机制（REALTIME_WRITE）
│   │   ├─ 失败重试机制（MAX_RETRIES=3）
│   │   └─ 生成 JSON 格式结果（chinese_question, teaching_output）
│   │
│   ├── process/                           # 处理后的数据文件
│   │   ├── apps_train_process.jsonl       # 标准化的 APPS 数据（中间数据）
│   │   ├── apps_train_sft_stepwise.jsonl  # SFT 训练数据（100 条样本演示版）
│   │   └── grpo_batch/
│   │       └── grpo.jsonl                 # GRPO 训练数据
│   │
│   ├── apps/                              # 原始 APPS 数据下载目录
│   │   └── apps_train.jsonl
│   │
│   └── download_apps_v2.py                # APPS 数据集下载脚本
│
├── scripts/                               # 核心训练和推理脚本
│   ├── run_sft.py                         # SFT 训练入口
│   │   └─ 执行：python scripts/run_sft.py --config configs/sft.yaml
│   │
│   ├── run_grpo.py                        # GRPO 训练入口
│   │   └─ 执行：python scripts/run_grpo.py --config configs/grpo.yaml
│   │
│   ├── infer_apps_sample.py               # 推理脚本（APPS 样本）
│   ├── infer_grpo_test.py                 # 推理脚本（GRPO 测试集）
│   ├── eval_reward.py                     # 离线评估奖励函数
│   ├── grpo_pipeline.py                   # 完整 GRPO 流程（SFT->GRPO->评估）
│   ├── download_merge_grpo_batch.py       # GRPO 数据批量准备脚本
│   └── download_apps_v2.py                # APPS 数据下载
│
├── src/pgt/                               # 核心库（Python 包）
│   ├── __init__.py
│   │
│   ├── config.py                          # YAML 配置加载器
│   │   ├─ load_yaml_with_extends: 支持配置继承（sft.yaml extends base.yaml）
│   │   └─ _deep_update: 递归配置合并
│   │
│   ├── data.py                            # 数据集构建
│   │   ├─ read_jsonl: 读取 JSONL 文件
│   │   ├─ build_sft_dataset: 构建 SFT 数据集（Dataset 对象）
│   │   ├─ build_rl_dataset: 构建 RL 数据集
│   │   └─ _build_tests_from_input_output: 测试用例标准化
│   │
│   ├── prompts.py                         # Prompt 模板管理
│   │   ├─ build_system_prompt: 根据学生水平生成系统提示
│   │   │   ├─ beginner: 清晰短句，基础循环数组
│   │   │   ├─ intermediate: 允许常见优化
│   │   │   └─ advanced: 允许高阶技巧，需要解释复杂度
│   │   └─ build_user_prompt: 格式化用户提示词
│   │
│   ├── reward/                            # 奖励函数模块
│   │   ├── __init__.py
│   │   │
│   │   ├── reward_model.py                # 核心奖励计算逻辑
│   │   │   ├─ _run_python_tests: 运行 Python 代码并与预期输出对比
│   │   │   ├─ compute_reward: 计算单个样本的多维奖励
│   │   │   │   ├─ accuracy: 代码执行准确率 (passed_tests / total_tests)
│   │   │   │   ├─ steps: 步骤匹配度 (min(step_count / target, 1.0))
│   │   │   │   └─ total: 加权组合奖励
│   │   │   ├─ RewardBreakdown: 奖励分解信息
│   │   │   └─ build_grpo_reward_func: 构建 GRPO 训练用的奖励函数
│   │   │
│   │   ├── extract.py                    # 代码和步骤提取
│   │   │   ├─ extract_code_with_lang: 从回复中提取代码（识别语言）
│   │   │   └─ extract_step_count: 统计【局部实现-N】的个数
│   │   │
│   │   └── compiler_runner.py             # C/C++ 代码编译与运行
│   │       ├─ compile_and_run_c_code: 编译 C 代码并执行测试
│   │       └─ 使用 GCC + 沙箱隔离
│   │
│   └── training/                          # 训练逻辑模块
│       ├── __init__.py
│       │
│       ├── sft_trainer.py                 # SFT 训练器
│       │   ├─ run_sft: SFT 训练主函数
│       │   ├─ 使用 HuggingFace Transformers + TRL SFTTrainer
│       │   ├─ 支持在线评估（sft_online_eval）
│       │   └─ W&B 日志集成
│       │
│       ├── grpo_trainer.py                # GRPO 训练器
│       │   ├─ run_grpo: GRPO 训练主函数
│       │   ├─ 加载 SFT Adapter
│       │   ├─ 生成多个候选响应
│       │   ├─ 计算优势估计（Advantage = reward - baseline）
│       │   └─ W&B 日志记录（grpo_reward/accuracy, grpo_reward/total_mean 等）
│       │
│       └── utils.py                       # 通用工具函数
│           ├─ 日志配置
│           ├─ 设备管理
│           └─ 其他辅助函数
│
├── outputs/                               # 训练输出目录
│   ├── sft_qwen7b_lora/                  # SFT 阶段输出
│   │   ├── checkpoint-XXX/               # 保存的检查点
│   │   ├── adapter_config.json           # LoRA 配置
│   │   ├── adapter_model.bin             # LoRA 权重
│   │   └── training_args.bin             # 训练参数
│   │
│   ├── grpo_qwen7b_lora/                 # GRPO 阶段输出
│   │   ├── checkpoint-XXX/               # 保存的检查点
│   │   └── ...
│   │
│   ├── base_test_predictions.jsonl       # 基础模型测试预测
│   ├── sft_test_predictions.jsonl        # SFT 模型测试预测
│   └── grpo_test_predictions.jsonl       # GRPO 模型测试预测
│
└── wandb/                                 # Weights & Biases 日志
    ├── latest-run/                        # 最新运行
    └── run-XXXXXXXX-XXXXXXXX/             # 历史运行记录
        ├── run-XXXXXXXX.wandb             # 日志二进制
        ├── files/
        │   ├── config.yaml                # 当次训练配置
        │   ├── requirements.txt           # 依赖版本记录
        │   └── wandb-summary.json         # 最终指标汇总
        └── logs/
```

---

## 关键脚本说明

### 1. 数据处理脚本

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `data/extract.py` | APPS 数据标准化 | `apps/apps_train.jsonl` | `data/process/apps_train_process.jsonl` |
| `data/build_sft_stepwise.py` | 调用 API 生成 SFT 讲解 | `apps_train_process.jsonl` | `apps_train_sft_stepwise.jsonl` |
| `scripts/download_apps_v2.py` | 下载 APPS 数据集 | - | `data/apps/` |
| `scripts/download_merge_grpo_batch.py` | 批量准备 GRPO 数据 | `apps_train_sft_stepwise.jsonl` | `grpo_batch/grpo.jsonl` |

### 2. 训练脚本

| 脚本 | 功能 | GPU 需求 |
|------|------|---------|
| `scripts/run_sft.py` | 执行 SFT 微调 | 单卡 4090 (≈22GB) |
| `scripts/run_grpo.py` | 执行 GRPO 强化学习 | 单卡 4090 (≈22GB) |
| `scripts/grpo_pipeline.py` | 完整流程：SFT→GRPO→评估 | 单卡 4090 |

### 3. 评估与推理

| 脚本 | 功能 |
|------|------|
| `scripts/eval_reward.py` | 离线计算模型输出的奖励值 |
| `scripts/infer_apps_sample.py` | 在 APPS 样本上进行推理 |
| `scripts/infer_grpo_test.py` | 在 GRPO 测试集上进行推理 |

---

## 安装与启动

### 1. 环境要求

- **Python**: 3.10+
- **GPU**: NVIDIA RTX 4090 (24GB) 或等价显存
- **CUDA**: 11.8+ 
- **cuDNN**: 8.0+

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/personalized_grpo_tutor.git
cd personalized_grpo_tutor

# 创建虚拟环境（或者用conda）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置模型与数据路径

编辑 `configs/base.yaml`，设置以下路径：

```yaml
model:
  base_model: /path/to/Qwen2.5-7B-Instruct  # 本地模型路径

paths:
  output_dir: outputs
  sft_data: data/process/apps_train_sft_stepwise.jsonl
  rl_data: data/process/grpo_batch/grpo.jsonl
```

### 4. 获取数据

#### 方式 A：下载原始 APPS 数据集

```bash
python scripts/download_apps_v2.py
```

#### 方式 B：使用已处理的演示数据

项目已包含处理好的演示数据 `data/process/apps_train_sft_stepwise.jsonl`（100 条样本）

### 5. 配置 W&B（可选）

```bash
wandb login  # 输入你的 W&B API Key
```

编辑 `configs/base.yaml`：

```yaml
wandb:
  enabled: true
  project: personalized_grpo_tutor
  entity: your_wandb_username
  mode: online
```

### 6. 启动训练

#### 仅 SFT 训练

```bash
python scripts/run_sft.py --config configs/sft.yaml
```

**预期时间**：单卡 4090，约 30 分钟

#### SFT + GRPO 完整流程

```bash
python scripts/grpo_pipeline.py --config configs/grpo.yaml
```

#### 仅 GRPO 训练（假设已有 SFT Adapter）

```bash
# 首先确保 configs/grpo.yaml 中的 `init_adapter_path` 指向已训练的 SFT Adapter
python scripts/run_grpo.py --config configs/grpo.yaml
```

### 7. 推理与评估

运行推理：

```bash
python scripts/infer_grpo_test.py --model_path outputs/grpo_qwen7b_lora/checkpoint-50
```

离线评估奖励：

```bash
python scripts/eval_reward.py --predictions_file outputs/grpo_test_predictions.jsonl
```

---

## 算力消耗

### 硬件配置

| 项目 | 规格 | 显存使用 |
|------|------|--------|
| GPU | NVIDIA RTX 4090 (单卡) | ~22-24GB |
| 数据 | 演示版 ~500 条 | - |
| | 完整版 ~5000 条 | - |

### 训练时间估算（单卡 4090）

| 阶段 | 数据量 | Batch Size | 时间 |
|------|-------|-----------|------|
| **SFT** | 100 条 | 1 → accum 8 | ~30 min |
| | 5000 条 | 1 → accum 8 | ~12 hours |
| **GRPO** | 100 条 | 1 → accum 4, 生成 8 个 | ~1 hour |
| | 5000 条 | 1 → accum 4, 生成 8 个 | ~2-3 days |

### 显存占用

- **基础模型** (Qwen-7B, bf16)：~15GB
- **LoRA Adapter**：~1GB
- **批次 + 梯度**：~6-8GB
- **总计**：~22-24GB（4090 足够）

---

## 训练结果

### 训练指标

以下是 SFT + GRPO 训练阶段的典型指标曲线：

#### 1. SFT 阶段指标

| 指标 | 描述 | 初值 | 最终值 |
|------|------|------|--------|
| **train/loss** | 训练损失 | ~1.5 | ~0.75-0.85 |
| **train/mean_token_accuracy** | Token 级准确率 | ~0.65 | ~0.78 |
| **train/num_tokens** | 已处理的 token 数 | ~50K | ~450K |

#### 2. GRPO 阶段指标

| 指标 | 描述 | 含义 |
|------|------|------|
| **grpo_reward/accuracy_mean** | 平均代码执行准确率 | 多少比例的测试用例通过 |
| **grpo_reward/accuracy_pass_rate** | 完全通过率（准确率≥99%） | 代码完全正确的样本比例 |
| **grpo_reward/steps_mean** | 步骤数匹配度 | 生成步数与目标步数的符合度 |
| **grpo_reward/total_mean** | 综合奖励值 | 最终的加权奖励得分 |

### 典型训练曲线

由 Weights & Biases 生成的训练曲线：

![Training Curves - Loss](./wandb_outputs/train_loss.png)
![Training Curves - Accuracy](./wandb_outputs/train_accuracy.png)
![Training Curves - GRPO Reward](./wandb_outputs/grpo_reward.png)

**数值范例**（来自实际运行）：

```
SFT 阶段结束：
  - train/loss: 0.82
  - train/mean_token_accuracy: 0.778

GRPO 阶段结束：
  - grpo_reward/accuracy_mean: 0.45
  - grpo_reward/accuracy_pass_rate: 0.12
  - grpo_reward/steps_mean: 0.68
  - grpo_reward/total_mean: 0.82
```


---

## 项目进度与改进方向

### 已完成

✅ **格式规范训练完成**
- 模型已学会遵循"【第一步】→【第二步】→【第三步】→【第四步】"的教学框架
- 能正确分段给出局部代码实现（【局部实现-1】、【局部实现-2】等）
- 末尾正确输出【最终代码（可运行）】

### 进行中 / 待改进

🔄 **数据质量提升**
- **当前状态**：演示数据集 100 条（使用 Qwen API 生成）
- **问题**：Qwen API 生成的讲解质量参差不齐，部分样本不符合格式约束
- **方案**：
  - 手工标注 500 条高质量讲解样本作为黄金数据集
  - 迭代 SYSTEM_PROMPT，严格约束生成格式
  - 通过 GPT-4/Claude 进行反复验证和过滤

🔄 **数据集规模扩大** 
- **当前规模**：500 条有效样本（SFT 训练数据）
- **目标规模**：5000 条
- **方案**：
  - 批量调用 Qwen API 生成初步讲解
  - 自动化过滤（检查格式、代码块数量、是否包含完整代码等）
  - 人工抽样验证

⚠️ **代码格式问题（关键）**
- **问题**：模型生成的代码必须是 **OJ 格式** 而非 LeetCode 格式
  - ❌ **LeetCode 格式**：`class Solution: def solve(...)` → 沙箱判定全 0
  - ✅ **OJ 格式**：`n = int(input()); ... print(result)` → 正确执行
- **解决方案**：
  - 更新 SYSTEM_PROMPT，明确指出"代码必须读取标准输入，输出到标准输出"
  - 在 Reward 函数中增加"格式检查"预处理（自动转换或拒绝错误格式）
  - 扩大高质量 OJ 格式样本数量

🔄 **GRPO 优化**
- **目标**：进一步提升代码执行准确率和步骤匹配度
- **方案**：
  - 调整 `beta` 参数（KL 散度权重）以平衡探索与优化
  - 增加奖励函数的"步骤完整性"权重
  - 实现 Reward Hacking 检测机制（识别通过取巧方式高分的样本）

---

## 配置文件详解

### base.yaml

```yaml
project_name: personalized_grpo_tutor  # 项目名称

wandb:
  enabled: true                          # 是否启用 W&B 日志
  project: personalized_grpo_tutor       # W&B 项目名
  entity: null                           # W&B 团队名（可选）
  mode: online                           # online / offline / disabled

model:
  base_model: /path/to/Qwen2.5-7B-Instruct  # 基础模型路径
  max_length: 2048                       # 最大序列长度
  use_4bit: false                        # 使用标准 LoRA (fp16)

lora:
  r: 32                                  # LoRA rank
  alpha: 64                              # LoRA alpha = 2 × r
  dropout: 0.05                          # LoRA Dropout 概率
  target_modules:                        # 目标层
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

paths:
  output_dir: outputs
  sft_data: data/sample_sft.jsonl
  rl_data: data/sample_rl.jsonl
```

### sft.yaml (extends base.yaml)

```yaml
extends: base.yaml  # 继承 base.yaml 所有配置

sft:
  output_subdir: sft_qwen7b_lora         # 输出子目录
  num_train_epochs: 1                    # 训练轮数
  per_device_train_batch_size: 1         # 每卡 batch size
  gradient_accumulation_steps: 8         # 梯度累积步数（有效 BS = 1 × 8 = 8）
  learning_rate: 2.0e-4                  # 学习率
  logging_steps: 5                       # 日志间隔
  save_steps: 100                        # 保存检查点间隔
  warmup_ratio: 0.03                     # 预热比例
  max_seq_length: 2048                   # 序列最大长度

lora:
  r: 32                                  # 覆盖 base.yaml 的 r
  alpha: 64

paths:
  sft_data: /path/to/apps_train_sft_stepwise.jsonl

sft_online_eval:
  enabled: true                          # 是否启用在线评估
  every_n_steps: 20                      # 每 N 步评估一次
  num_samples: 6                         # 评估样本数
  max_new_tokens: 256                    # 生成最大 token 数
```

### grpo.yaml (extends base.yaml)

```yaml
extends: base.yaml

grpo:
  output_subdir: grpo_qwen7b_lora
  init_adapter_path: /path/to/sft_qwen7b_lora  # 从 SFT Adapter 初始化
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-5                  # GRPO 学习率更低
  logging_steps: 2
  save_steps: 50
  max_prompt_length: 768                 # 问题最大长度
  max_completion_length: 512             # 生成答案最大长度
  num_generations: 8                     # 每个 prompt 生成 8 个候选
  beta: 0.04                             # KL 散度系数（防止偏离太远）
  reward_log_every: 1                    # 每步记录奖励日志

reward:
  code_language: python                  # 目标编程语言
  w_accuracy: 1.2                        # 准确率权重
  w_steps: 0.4                           # 步数权重
  pass_bonus: 0.1                        # 通过奖励加分

paths:
  rl_data: /path/to/grpo_batch/grpo.jsonl
```

---

## 常见问题

### Q1: 模型下载慢怎么办？

**A**: 使用国内镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

### Q2: 显存不足报错

**A**: 调整 batch size 和梯度累积步数：

```yaml
per_device_train_batch_size: 1  # 已是最小
gradient_accumulation_steps: 4  # 继续减少
```

或减少 LoRA rank 和 alpha 参数：

```yaml
lora:
  r: 16          # 从 32 降至 16
  alpha: 32      # 从 64 降至 32
```

### Q3: Reward 函数如何验证生成的代码？

**A**: 
- **Python 代码**：临时文件中运行，对比输出
- **C 代码**：GCC 编译后运行，沙箱隔离执行
- 所有输出与预期测试用例 `.strip()` 后逐字对比

### Q4: 如何自定义奖励函数？

**A**: 编辑 `src/pgt/reward/reward_model.py` 中的 `compute_reward` 函数，修改权重或添加新维度。

### Q5: W&B 如何集成？

**A**: 项目自动集成，无需额外代码。只需 `wandb login` 然后在 `configs/base.yaml` 中启用：

```yaml
wandb:
  enabled: true
  mode: online
```

---

## 参考资源

- **TRL 文档**：https://huggingface.co/docs/trl/
- **PEFT (LoRA)**：https://github.com/huggingface/peft
- **APPS 数据集**：https://huggingface.co/datasets/codeparrot/apps
- **Qwen 模型**：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

---

## 许可证

MIT License - 详见 LICENSE 文件

---

## 联系方式

有问题或建议？欢迎提交 Issue 或 PR！

**Email**: 20040015lin@gmail.com  
**GitHub**: https://github.com/yourusername/personalized_grpo_tutor

---

**项目更新时间**：2026年4月17日  
**最后维护者**：@Lindi798
