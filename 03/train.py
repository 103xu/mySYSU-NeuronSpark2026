"""
训练脚本
- 基于 Qwen2.5-1.5B-Instruct 做 LoRA 微调
- 6 分类任务：benign / prompt_injection / data_exfiltration / tool_abuse / jailbreak / ambiguous
- 支持 GroupKFold、数据增强、fp16 训练
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dataset import load_jsonl, augment_dataset
from utils import LABEL2ID, ID2LABEL, set_seed

# ========================
# 配置
# ========================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 1024
OUTPUT_DIR = "./outputs"
FINAL_MODEL_DIR = "./outputs/final_model"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
USE_FP16 = False  # 使用 bf16 替代 fp16，避免梯度缩放冲突
USE_BF16 = True
USE_KFOLD = False  # 数据集较小，默认使用 train/test split
KFOLD_SPLITS = 5
SEED = 42
USE_AUGMENTATION = True
AUGMENT_MULTIPLIER = 2


def load_or_prepare_tokenizer():
    """加载 tokenizer 并设置 pad_token"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """批量 tokenize"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


def compute_metrics(eval_pred):
    """评估指标：accuracy, macro F1, 各类别 recall"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")
    weighted_f1 = f1_score(labels, predictions, average="weighted")

    # 高风险类别 recall
    high_risk_ids = [
        LABEL2ID["prompt_injection"],
        LABEL2ID["data_exfiltration"],
        LABEL2ID["tool_abuse"],
        LABEL2ID["jailbreak"],
    ]
    hr_recalls = {}
    for label_name in ["prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"]:
        label_id = LABEL2ID[label_name]
        mask = labels == label_id
        if mask.sum() > 0:
            hr_recalls[f"recall_{label_name}"] = (
                predictions[mask] == label_id
            ).mean()
        else:
            hr_recalls[f"recall_{label_name}"] = 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **hr_recalls,
    }


def build_model(num_labels: int = 6):
    """构建 LoRA 模型"""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        trust_remote_code=True,
        dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32),
    )

    # 设置 pad_token_id
    base_model.config.pad_token_id = base_model.config.eos_token_id

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def train_single_fold(train_data, val_data, tokenizer, fold_name: str = ""):
    """
    训练单个 fold
    返回训练好的模型
    """
    # 构建 HuggingFace Dataset
    train_dataset = HFDataset.from_list(train_data)
    val_dataset = HFDataset.from_list(val_data)

    # Tokenize
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=[c for c in train_dataset.column_names if c not in ["label_id"]],
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=[c for c in val_dataset.column_names if c not in ["label_id"]],
    )

    # 重命名 label 列（Trainer 需要 "label" 列名）
    train_dataset = train_dataset.rename_column("label_id", "label")
    val_dataset = val_dataset.rename_column("label_id", "label")

    # 构建模型
    model = build_model(num_labels=6)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{fold_name}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=USE_FP16,
        bf16=USE_BF16,
        report_to="none",
        seed=SEED,
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n{'='*50}")
    print(f"开始训练: {fold_name}")
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")
    print(f"{'='*50}\n")

    trainer.train()

    # 评估
    eval_results = trainer.evaluate()
    print(f"\n{fold_name} 评估结果:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    return model, trainer


def train_with_split():
    """使用 train/test split 方式训练"""
    set_seed(SEED)

    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    # 加载数据
    data = load_jsonl("train.jsonl")

    # 数据增强
    if USE_AUGMENTATION:
        original_count = len(data)
        data = augment_dataset(data, multiplier=AUGMENT_MULTIPLIER)
        print(f"数据增强: {original_count} → {len(data)} (x{len(data)/original_count:.1f})")

    # 分层划分
    labels = [d["label"] for d in data]
    train_data, val_data = train_test_split(
        data,
        test_size=0.15,
        stratify=labels,
        random_state=SEED,
    )

    # 统计分布
    from collections import Counter
    print(f"\n训练集分布: {dict(Counter(d['label'] for d in train_data))}")
    print(f"验证集分布: {dict(Counter(d['label'] for d in val_data))}")

    # 加载 tokenizer
    tokenizer = load_or_prepare_tokenizer()

    # 训练
    model, trainer = train_single_fold(train_data, val_data, tokenizer, fold_name="split")

    # 保存最终模型（合并 LoRA 权重后保存完整模型）
    print(f"\n保存模型到 {FINAL_MODEL_DIR}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    # 保存分类报告（容错处理）
    try:
        save_classification_report(trainer, val_data, tokenizer)
    except Exception as e:
        print(f"分类报告生成失败 (可忽略): {e}")

    print(f"\n训练完成！模型已保存到 {FINAL_MODEL_DIR}")
    return model, tokenizer


def train_with_kfold():
    """使用 GroupKFold (StratifiedKFold) 方式训练"""
    set_seed(SEED)

    print("=" * 60)
    print(f"加载数据 (StratifiedKFold, splits={KFOLD_SPLITS})...")
    print("=" * 60)

    data = load_jsonl("train.jsonl")

    if USE_AUGMENTATION:
        data = augment_dataset(data, multiplier=AUGMENT_MULTIPLIER)

    labels = [d["label"] for d in data]
    label_ids = [LABEL2ID[l] for l in labels]

    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=SEED)
    tokenizer = load_or_prepare_tokenizer()

    models = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, label_ids)):
        train_fold = [data[i] for i in train_idx]
        val_fold = [data[i] for i in val_idx]

        model, trainer = train_single_fold(
            train_fold, val_fold, tokenizer, fold_name=f"fold_{fold}"
        )
        models.append(model)

    # 保存最后一个 fold 的模型（合并后保存）
    final_model = models[-1]
    merged_model = final_model.merge_and_unload()
    merged_model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print(f"\nK-Fold 训练完成！最终模型已保存到 {FINAL_MODEL_DIR}")
    return final_model, tokenizer


def save_classification_report(trainer, val_data, tokenizer):
    """生成并保存分类报告"""
    from sklearn.metrics import classification_report as cr

    # 收集预测和真实标签
    val_dataset = HFDataset.from_list(val_data)
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    val_dataset = val_dataset.rename_column("label_id", "label")

    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = cr(labels, preds, target_names=list(LABEL2ID.keys()), digits=4)

    with open(f"{OUTPUT_DIR}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n分类报告:\n{report}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kfold", action="store_true", help="使用 K-Fold 训练")
    parser.add_argument("--no-augment", action="store_true", help="不使用数据增强")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="训练轮数")
    parser.add_argument("--train_data", type=str, default="train.jsonl", help="训练数据路径")
    args = parser.parse_args()

    if args.no_augment:
        USE_AUGMENTATION = False
    if args.epochs:
        NUM_EPOCHS = args.epochs

    # 修改训练数据路径（如果需要）
    train_path = args.train_data

    # 修改 load_jsonl 的默认路径
    import dataset as ds
    original_load = ds.load_jsonl

    def patched_load(path, with_label=True):
        if path == "train.jsonl":
            path = train_path
        return original_load(path, with_label)

    ds.load_jsonl = patched_load

    if args.kfold:
        train_with_kfold()
    else:
        train_with_split()
