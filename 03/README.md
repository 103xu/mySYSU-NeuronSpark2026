# NS-2026-03 防线之内：Prompt Injection 与内容安全检测

## 一、任务概述

本项目的目标是检测 **Prompt Injection** 及各类 AI 安全威胁，对用户输入进行六分类：

| 类别 | 含义 | risk_score 参考 |
|------|------|----------------|
| `benign` | 正常请求 | 0.00 ~ 0.30 |
| `prompt_injection` | 覆盖/泄露系统指令 | 0.70 ~ 1.00 |
| `data_exfiltration` | 诱导泄露敏感数据 | 0.75 ~ 1.00 |
| `tool_abuse` | 越权调用工具 | 0.70 ~ 1.00 |
| `jailbreak` | 解除安全限制 | 0.75 ~ 1.00 |
| `ambiguous` | 意图可疑但不确定 | 0.35 ~ 0.65 |

## 二、技术方案

采用 **「规则系统 + Transformer 分类 + risk_score 融合」** 混合架构：

```
输入文本 → 文本拼接 → 规则检测 → Transformer → 概率融合 → 风险校准 → results.json
```

### 2.1 规则系统 (rules.py)

- **4 类高风险检测**：prompt_injection / data_exfiltration / tool_abuse / jailbreak
- **100+ 正则模式**：覆盖英文、中文、中英混合攻击模式
- **加权匹配**：核心 pattern 权重高，辅助 pattern 权重低
- **组合加分**：多关键词同时出现时额外加分
- **Unicode 标准化**：NFKC 规范化防止同形字符混淆
- **混淆检测**：零宽字符、非ASCII比例、异常空格

### 2.2 模型部分

- **基座**：Qwen/Qwen2.5-1.5B-Instruct
- **微调**：LoRA (rank=16, alpha=32)
- **分类头**：6 类 SequenceClassification
- **精度**：bf16 训练
- **最大长度**：1024 tokens

### 2.3 融合策略

```
final_score = 0.7 * model_prob + 0.3 * rule_score
risk_score  = P(prompt_injection) + P(data_exfiltration) + P(tool_abuse) + P(jailbreak)
```

### 2.4 ambiguous 判定

当 `top1_score - top2_score < 0.15` 且满足以下条件之一时，归为 ambiguous：
- top1 本身就是 ambiguous 类别
- 两个高分类别都是高风险类别
- top1 置信度不足 0.4

### 2.5 数据增强

- **Unicode 对抗增强**：同形字符替换（如 a→а）
- **中英混合增强**：随机替换中文关键词为英文
- **拼写扰动增强**：1337 风格替换（如 bypass→byp4ss）
- **增强倍数**：对高风险类别额外生成 2 倍样本

## 三、项目结构

```
NS-2026-03/
├── train.py          # 训练脚本
├── predict.py        # 推理脚本
├── rules.py          # 规则检测系统
├── dataset.py        # 数据处理与增强
├── utils.py          # 工具函数
├── requirements.txt  # 依赖列表
├── README.md         # 本文档
├── train.jsonl       # 训练数据
├── test.jsonl        # 测试数据
├── results.json      # 预测结果
└── outputs/
    └── final_model/  # 训练好的模型
```

## 四、输入格式

模型输入使用多字段拼接（不是仅 user_input）：

```
[ SYSTEM ]
{system_prompt}

[ TOOLS ]
工具名: {name}, 描述: {description}
...

[ HISTORY ]
{conversation_history}

[ USER ]
{user_input}
```

## 五、环境搭建

```bash
# 1. 创建虚拟环境 (Python >= 3.10)
python -m venv .venv

# 2. 激活环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 六、训练

```bash
# 标准训练（train/test split）
python train.py

# K-Fold 训练
python train.py --kfold

# 自定义参数
python train.py --epochs 5 --no-augment

# 使用 HF 镜像（国内）
HF_ENDPOINT=https://hf-mirror.com python train.py
```

## 七、推理

```bash
# 标准推理
python predict.py

# 指定数据路径
python predict.py --test_data /path/to/test.jsonl --output results.json

# 使用 HF 镜像（国内）
HF_ENDPOINT=https://hf-mirror.com python predict.py
```

## 八、输出格式

```json
[
  {
    "id": "te_00001",
    "label": "prompt_injection",
    "risk_score": 0.92
  }
]
```

- `id`: 测试样本 ID
- `label`: 预测类别（6 选 1）
- `risk_score`: 风险分数（0~1）

## 九、模型结构

```
Qwen2.5-1.5B-Instruct (1.54B params)
├── LoRA Adapters (18.5M trainable, 1.18%)
│   ├── q_proj (rank=16)
│   ├── k_proj (rank=16)
│   ├── v_proj (rank=16)
│   ├── o_proj (rank=16)
│   ├── gate_proj (rank=16)
│   ├── up_proj (rank=16)
│   └── down_proj (rank=16)
└── Classification Head (6 classes)
```

## 十、风险分类解释

| 类别 | 典型攻击模式 |
|------|-------------|
| prompt_injection | "ignore previous instructions", "忽略规则", "打印系统提示词" |
| data_exfiltration | "export all records", "显示密钥", "导出数据库" |
| tool_abuse | "bypass confirmation", "跳过审批", "冒充其他用户" |
| jailbreak | "DAN mode", "pretend to be unrestricted", "解除安全限制" |
| ambiguous | 存在可疑意图但不足以明确判定 |
| benign | 正常的、合规的用户请求 |

## 十一、后续优化方向

1. **模型升级**：使用 Qwen2.5-7B 或更大模型
2. **集成学习**：多模型投票/stacking
3. **对比学习**：增强类别间区分度
4. **主动学习**：针对 ambiguous 样本进行人工标注
5. **Online Learning**：持续学习新的攻击模式
6. **LLM-as-Judge**：使用更大模型做二次审核
7. **更多增强策略**：同义替换、句式变换、对抗样本生成
8. **阈值自适应**：根据业务场景自动调整决策阈值
