"""
NS-2026-00 文档图像分类 - 推理脚本
加载最佳模型 -> 在测试集上推理 -> 生成 results.csv
"""
import os
import argparse
import pandas as pd

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DocumentDataset, test_transform, id2label
from model import build_model

BATCH_SIZE = 32
BEST_MODEL_PATH = os.path.join("checkpoints", "best_model.pth")
RESULTS_PATH = "results.csv"


def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def predict(args):
    """加载模型并在测试集上推理，生成 results.csv"""
    device = get_device()
    print(f"使用设备: {device}")

    # --- 1. 构建模型并加载权重 ---
    model = build_model()
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"模型已加载: {BEST_MODEL_PATH}")

    # --- 2. 加载测试数据集 ---
    test_csv = os.path.join(args.data_dir, "test.csv")
    test_dataset = DocumentDataset(
        csv_path=test_csv,
        data_dir=args.data_dir,
        transform=test_transform,
        is_test=True,
    )
    print(f"测试集样本数: {len(test_dataset)}")

    num_workers = min(os.cpu_count() or 0, 4)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- 3. 推理 ---
    all_ids = []
    all_labels = []

    pbar = tqdm(test_loader, desc="推理")
    for images, sample_ids in pbar:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for sample_id, pred_idx in zip(sample_ids, preds):
            all_ids.append(sample_id)
            all_labels.append(id2label[pred_idx.item()])

    # --- 4. 生成 results.csv ---
    results_df = pd.DataFrame({"id": all_ids, "label": all_labels})
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"结果已保存至: {RESULTS_PATH}")
    print(f"共 {len(results_df)} 条预测记录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NS-2026 文档图像分类 Baseline 推理")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="d1b5a028-b288-4ab9-a872-64c6e12a9185",
        help="数据目录路径（包含 test.csv 和 images/）",
    )
    args = parser.parse_args()
    predict(args)
