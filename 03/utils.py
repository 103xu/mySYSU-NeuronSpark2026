"""
通用工具函数
- 标签映射
- 结果保存与验证
- 随机种子设置
"""

import json
import random
import os

import numpy as np
import torch

# ========================
# 标签映射
# ========================
LABEL2ID = {
    "benign": 0,
    "prompt_injection": 1,
    "data_exfiltration": 2,
    "tool_abuse": 3,
    "jailbreak": 4,
    "ambiguous": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# 合法标签集合
VALID_LABELS = set(LABEL2ID.keys())

# 高风险类别（用于 risk_score 计算）
HIGH_RISK_LABELS = ["prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"]

# 标签对应的 risk_score 参考范围（来自 label_schema.json）
LABEL_RISK_HINT = {
    "benign": (0.00, 0.30),
    "prompt_injection": (0.70, 1.00),
    "data_exfiltration": (0.75, 1.00),
    "tool_abuse": (0.70, 1.00),
    "jailbreak": (0.75, 1.00),
    "ambiguous": (0.35, 0.65),
}


def set_seed(seed: int = 42):
    """固定随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def validate_results(results: list) -> bool:
    """
    验证结果格式是否正确
    每个元素必须包含 id (str), label (合法类别), risk_score (0-1 的 float)
    """
    if not isinstance(results, list):
        print("[ERROR] results 必须是 list")
        return False

    for i, item in enumerate(results):
        if "id" not in item:
            print(f"[ERROR] 第 {i} 项缺少 id")
            return False
        if "label" not in item:
            print(f"[ERROR] 第 {i} 项缺少 label")
            return False
        if "risk_score" not in item:
            print(f"[ERROR] 第 {i} 项缺少 risk_score")
            return False
        if item["label"] not in VALID_LABELS:
            print(f"[ERROR] label={item['label']} 不是合法类别")
            return False
        if not (0.0 <= item["risk_score"] <= 1.0):
            print(f"[ERROR] risk_score={item['risk_score']} 超出 0~1 范围")
            return False

    print(f"[OK] 共 {len(results)} 条结果，格式验证通过")
    return True


def save_results(results: list, path: str = "results.json"):
    """保存结果到 JSON 文件，包含格式验证"""
    if validate_results(results):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[OK] 结果已保存到 {path}")
        return True
    else:
        print(f"[ERROR] 格式验证失败，未保存")
        return False
