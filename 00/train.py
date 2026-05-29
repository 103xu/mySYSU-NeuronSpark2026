"""
NS-2026-00 文档图像分类 - 训练脚本
完整训练流程：数据加载 → 验证集划分 → 训练循环 → 保存最佳模型
"""
import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import DocumentDataset, train_transform, test_transform, label2id, id2label
from model import build_model

# ==================== 超参数 ====================

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.2          # 验证集占比
RANDOM_SEED = 42
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# ==================== 工具函数 ====================


def set_seed(seed=RANDOM_SEED):
    """固定随机种子，保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device():
    """自动检测可用设备：CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_accuracy(outputs, labels):
    """计算分类准确率"""
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct, len(labels)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """执行一个 epoch 的训练，返回平均 loss 和准确率"""
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="训练", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct, batch_size = compute_accuracy(outputs, labels)
        total_correct += correct
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """在验证集上评估模型，返回平均 loss 和准确率"""
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="验证", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct, batch_size = compute_accuracy(outputs, labels)
        total_correct += correct
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


# ==================== 主训练流程 ====================


def train(args):
    """完整训练流程"""
    set_seed()

    # --- 1. 检查 CUDA ---
    device = get_device()
    print(f"使用设备: {device}")

    # --- 2. 读取数据 ---
    train_csv = os.path.join(args.data_dir, "train.csv")
    dataset = DocumentDataset(
        csv_path=train_csv,
        data_dir=args.data_dir,
        transform=train_transform,
        is_test=False,
    )
    print(f"数据集总样本数: {len(dataset)}")

    # --- 3. 划分训练集 / 验证集 ---
    df = pd.read_csv(train_csv)
    labels = df["label"].values
    indices = np.arange(len(dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    print(f"训练集: {len(train_indices)} 张, 验证集: {len(val_indices)} 张")

    # 验证集使用 test_transform（无数据增强）
    train_dataset = Subset(dataset, train_indices)
    val_dataset = DocumentDataset(
        csv_path=train_csv,
        data_dir=args.data_dir,
        transform=test_transform,
        is_test=False,
    )
    val_dataset = Subset(val_dataset, val_indices)

    # --- 4. 创建 DataLoader ---
    num_workers = min(os.cpu_count() or 0, 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- 5. 构建模型 ---
    model = build_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # --- 6. 训练循环 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"开始训练 | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | LR: {LR}")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[保存] 最佳模型 -> {BEST_MODEL_PATH} (Val Acc: {best_val_acc:.4f})")

        print()

    # --- 7. 训练完成 ---
    print(f"{'='*60}")
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"模型保存至: {BEST_MODEL_PATH}")
    print(f"{'='*60}")


# ==================== 入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NS-2026 文档图像分类 Baseline 训练")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="d1b5a028-b288-4ab9-a872-64c6e12a9185",
        help="数据目录路径（包含 train.csv 和 images/）",
    )
    args = parser.parse_args()
    train(args)
