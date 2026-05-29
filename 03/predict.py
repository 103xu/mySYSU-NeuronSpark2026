"""
推理脚本
- 加载 test.jsonl
- 加载训练好的模型
- 规则系统 + 模型融合
- 计算 risk_score
- ambiguous 判定
- 输出 results.json
"""

import json
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset import load_jsonl, build_text
from utils import (
    ID2LABEL, LABEL2ID, HIGH_RISK_LABELS,
    save_results, validate_results,
)
from rules import rule_predict

# ========================
# 配置
# ========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_PATH = "./outputs/final_model"
MAX_LENGTH = 1024
# 模型与规则融合权重
MODEL_WEIGHT = 0.7
RULE_WEIGHT = 0.3
# ambiguous 判定阈值
AMBIGUOUS_THRESHOLD = 0.15

# 6 个类别列表（顺序必须与模型输出一致）
LABELS_ORDERED = [
    "benign",
    "prompt_injection",
    "data_exfiltration",
    "tool_abuse",
    "jailbreak",
    "ambiguous",
]


def load_model(model_path: str = MODEL_PATH):
    """加载训练好的模型和 tokenizer"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer


def softmax(x: np.ndarray) -> np.ndarray:
    """稳定的 softmax 计算"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_single(
    text: str,
    model,
    tokenizer,
) -> tuple:
    """
    对单条文本进行预测
    返回: (pred_label, risk_score, detailed_scores)
    """
    # 1. 模型推理
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0].cpu().numpy()
    probs = softmax(logits)

    # 模型分数
    model_scores = {
        LABELS_ORDERED[i]: float(probs[i])
        for i in range(len(LABELS_ORDERED))
    }

    # 2. 规则检测
    rule_scores = rule_predict(text)

    # 3. 融合分数
    final_scores = {}
    for label in LABELS_ORDERED:
        final_scores[label] = (
            MODEL_WEIGHT * model_scores[label]
            + RULE_WEIGHT * rule_scores[label]
        )

    # 4. 确定预测标签（含 ambiguous 判定）
    sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = sorted_items[0]
    top2_label, top2_score = sorted_items[1]

    # ambiguous 判定策略：top1 与 top2 差距小于阈值
    if top1_score - top2_score < AMBIGUOUS_THRESHOLD:
        if top1_label == "ambiguous":
            pred_label = "ambiguous"
        elif top1_label in HIGH_RISK_LABELS and top2_label in HIGH_RISK_LABELS:
            pred_label = "ambiguous"
        elif top1_label != "ambiguous" and top2_label != "ambiguous" and top1_score < 0.4:
            pred_label = "ambiguous"
        else:
            pred_label = top1_label
    else:
        pred_label = top1_label

    # 5. 计算 risk_score = 四个高风险概率之和
    risk_score = sum(
        final_scores[label]
        for label in HIGH_RISK_LABELS
    )
    risk_score = min(1.0, max(0.0, float(risk_score)))

    return pred_label, risk_score, {
        "model_scores": model_scores,
        "rule_scores": rule_scores,
        "final_scores": final_scores,
    }


def predict_batch(
    texts: list,
    model,
    tokenizer,
    batch_size: int = 8,
) -> list:
    """
    批量推理（GPU 利用率更高）
    返回: [(pred_label, risk_score, detailed_scores), ...]
    """
    results = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="批量推理"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu().numpy()

        for j, logit in enumerate(logits):
            probs = softmax(logit)

            model_scores = {
                LABELS_ORDERED[k]: float(probs[k])
                for k in range(len(LABELS_ORDERED))
            }

            rule_scores = rule_predict(batch_texts[j])

            final_scores = {}
            for label in LABELS_ORDERED:
                final_scores[label] = (
                    MODEL_WEIGHT * model_scores[label]
                    + RULE_WEIGHT * rule_scores[label]
                )

            sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            top1_label, top1_score = sorted_items[0]
            top2_label, top2_score = sorted_items[1]

            if top1_score - top2_score < AMBIGUOUS_THRESHOLD:
                if top1_label == "ambiguous":
                    pred_label = "ambiguous"
                elif top1_label in HIGH_RISK_LABELS and top2_label in HIGH_RISK_LABELS:
                    pred_label = "ambiguous"
                elif top1_label != "ambiguous" and top2_label != "ambiguous" and top1_score < 0.4:
                    pred_label = "ambiguous"
                else:
                    pred_label = top1_label
            else:
                pred_label = top1_label

            risk_score = sum(final_scores[label] for label in HIGH_RISK_LABELS)
            risk_score = min(1.0, max(0.0, float(risk_score)))

            results.append((pred_label, risk_score, {
                "model_scores": model_scores,
                "rule_scores": rule_scores,
                "final_scores": final_scores,
            }))

    return results


def run_inference(
    test_path: str = "test.jsonl",
    model_path: str = MODEL_PATH,
    output_path: str = "results.json",
    use_batch: bool = True,
):
    """
    完整推理流程
    """
    print("=" * 60)
    print("开始推理")
    print("=" * 60)

    # 加载模型
    print(f"加载模型: {model_path}")
    model, tokenizer = load_model(model_path)
    print(f"模型加载完成 (设备: {'cuda' if torch.cuda.is_available() else 'cpu'})")

    # 加载测试数据
    print(f"\n加载测试数据: {test_path}")
    test_data = load_jsonl(test_path, with_label=False)
    print(f"测试样本数: {len(test_data)}")

    # 提取文本
    texts = [sample["text"] for sample in test_data]

    # 推理
    if use_batch:
        preds = predict_batch(texts, model, tokenizer)
    else:
        preds = []
        for text in tqdm(texts, desc="推理"):
            pred = predict_single(text, model, tokenizer)
            preds.append(pred)

    # 构建输出
    results = []
    for sample, (pred_label, risk_score, _) in zip(test_data, preds):
        results.append({
            "id": sample["id"],
            "label": pred_label,
            "risk_score": risk_score,
        })

    # 验证并保存
    print("\n" + "=" * 60)
    save_results(results, output_path)

    # 输出统计
    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    print(f"\n预测分布:")
    for label in LABELS_ORDERED:
        count = label_counts.get(label, 0)
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")

    # 输出 risk_score 统计
    risk_scores = [r["risk_score"] for r in results]
    print(f"\nrisk_score 统计:")
    print(f"  min={min(risk_scores):.4f}, max={max(risk_scores):.4f}")
    print(f"  mean={np.mean(risk_scores):.4f}, median={np.median(risk_scores):.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prompt Injection 检测推理")
    parser.add_argument("--test_data", type=str, default="test.jsonl", help="测试数据路径")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="模型路径")
    parser.add_argument("--output", type=str, default="results.json", help="输出文件路径")
    parser.add_argument("--no-batch", action="store_true", help="禁用批量推理")
    args = parser.parse_args()

    run_inference(
        test_path=args.test_data,
        model_path=args.model_path,
        output_path=args.output,
        use_batch=not args.no_batch,
    )
