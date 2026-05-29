"""
数据处理模块
- 多字段拼接（system + tools + history + user_input）
- 数据加载
- 数据增强（Unicode 对抗、中英混合、拼写扰动）
"""

import json
import random
import copy

from utils import LABEL2ID

# ========================
# 文本拼接
# ========================

def build_text(sample: dict) -> str:
    """
    将 system / tools / history / user_input 拼接为统一文本
    顺序和分隔符与 label_schema 描述一致
    """
    # 工具列表格式化为文本
    tools_text = ""
    if isinstance(sample.get("tools"), list):
        tool_parts = []
        for t in sample["tools"]:
            if isinstance(t, dict):
                tool_parts.append(
                    f"工具名: {t.get('name', '')}, 描述: {t.get('description', '')}"
                )
            else:
                tool_parts.append(str(t))
        tools_text = "\n".join(tool_parts)

    text = f"""[ SYSTEM ]
{sample.get('system', '')}

[ TOOLS ]
{tools_text}

[ HISTORY ]
{sample.get('history', '')}

[ USER ]
{sample.get('user_input', '')}"""

    return text


# ========================
# 数据加载
# ========================

def load_jsonl(path: str, with_label: bool = True) -> list:
    """
    从 JSONL 文件加载数据
    - with_label=True: 加载训练数据（含 label）
    - with_label=False: 加载测试数据（无 label）
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample["text"] = build_text(sample)
            if with_label and "label" in sample:
                sample["label_id"] = LABEL2ID[sample["label"]]
            data.append(sample)
    return data


# ========================
# 数据增强
# ========================

# Unicode 同形字符映射（视觉上相似但编码不同）
UNICODE_CONFUSABLES = {
    "a": ["а", "à", "á"],  # Cyrillic a, à, á
    "e": ["е", "è", "é"],  # Cyrillic e, è, é
    "i": ["і", "ì", "í"],  # Cyrillic i, ì, í
    "o": ["о", "ò", "ó"],  # Cyrillic o, ò, ó
    "c": ["с", "ç"],             # Cyrillic s, ç
    "p": ["р"],                        # Cyrillic r
    "y": ["у"],                        # Cyrillic u
    "x": ["х"],                        # Cyrillic kh
}

# 中英混合替换映射
CN_EN_MIX = {
    "忽略": "ignore",
    "规则": "rules",
    "指令": "instructions",
    "系统": "system",
    "提示": "prompt",
    "限制": "restrictions",
    "安全": "safety",
    "删除": "delete",
    "文件": "file",
    "发送": "send",
    "邮件": "email",
    "绕过": "bypass",
    "权限": "permission",
    "管理员": "admin",
    "数据库": "database",
    "导出": "export",
    "泄露": "leak",
    "密钥": "key",
    "隐藏": "hidden",
}


def unicode_obfuscate(text: str, prob: float = 0.3) -> str:
    """
    Unicode 对抗增强：随机将部分字符替换为视觉相似的同形字符
    模拟攻击者使用的 Unicode 混淆技术
    """
    chars = list(text)
    for i, ch in enumerate(chars):
        lower = ch.lower()
        if lower in UNICODE_CONFUSABLES and random.random() < prob:
            chars[i] = random.choice(UNICODE_CONFUSABLES[lower])
    return "".join(chars)


def cn_en_mix(text: str, prob: float = 0.3) -> str:
    """
    中英混合增强：将中文关键词随机替换为英文对应词
    模拟实际攻击中常见的中英混杂手法
    """
    for cn, en in CN_EN_MIX.items():
        if cn in text and random.random() < prob:
            text = text.replace(cn, en, 1)
    return text


def spelling_perturbation(text: str, prob: float = 0.1) -> str:
    """
    拼写扰动增强：对英文单词进行随机字符替换或重复
    模拟攻击者故意拼错以绕过关键词过滤
    """
    perturbations = {
        "ignore": "ign0re",
        "bypass": "byp4ss",
        "system": "syst3m",
        "prompt": "pr0mpt",
        "admin": "4dm1n",
        "delete": "d3l3t3",
        "email": "3m41l",
        "file": "f1l3",
        "hidden": "h1dd3n",
        "security": "s3cur1ty",
        "access": "4cc3ss",
        "password": "p4ssw0rd",
        "token": "t0k3n",
    }
    for word, perturbed in perturbations.items():
        if word in text.lower() and random.random() < prob:
            text = text.lower().replace(word, perturbed, 1)
    return text


def augment_sample(sample: dict, strategies: list = None) -> dict:
    """
    对单条样本应用数据增强策略
    返回增强后样本的副本（保留原始 label）
    """
    if strategies is None:
        strategies = ["unicode", "cn_en", "spelling"]

    new_sample = copy.deepcopy(sample)
    text = new_sample["user_input"]

    if "unicode" in strategies:
        text = unicode_obfuscate(text, prob=random.uniform(0.2, 0.4))
    if "cn_en" in strategies:
        text = cn_en_mix(text, prob=random.uniform(0.2, 0.5))
    if "spelling" in strategies:
        text = spelling_perturbation(text, prob=random.uniform(0.1, 0.3))

    new_sample["user_input"] = text
    new_sample["text"] = build_text(new_sample)
    return new_sample


def augment_dataset(data: list, multiplier: int = 2) -> list:
    """
    对整个数据集进行增强
    - 原始数据全部保留
    - 对高危类别额外生成 multiplier 倍的增强样本
    """
    high_risk = {"prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"}
    augmented = list(data)

    for sample in data:
        if sample.get("label") in high_risk:
            for _ in range(multiplier):
                aug = augment_sample(sample)
                augmented.append(aug)

    random.shuffle(augmented)
    return augmented
