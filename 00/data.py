"""
NS-2026-00 文档图像分类 - 数据集模块
提供 DocumentDataset、数据增强、标签映射等功能
"""
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ==================== 标签映射 ====================

label2id = {
    "invoice": 0,
    "receipt": 1,
    "schedule": 2,
    "poster": 3,
    "lab_note": 4,
    "notice": 5,
    "handwritten_note": 6,
    "form": 7,
    "meeting_minutes": 8,
    "grade_report": 9,
}

id2label = {v: k for k, v in label2id.items()}
NUM_CLASSES = len(label2id)

# ==================== 数据增强 ====================

# ImageNet 标准均值与标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==================== Dataset ====================

class DocumentDataset(Dataset):
    """
    文档图像分类数据集

    Args:
        csv_path: train.csv 或 test.csv 的路径
        data_dir: 图片根目录（CSV 中的 image 路径相对于此目录）
        transform: torchvision 数据增强
        is_test: True 表示测试模式（无标签），False 表示训练模式
    """

    def __init__(self, csv_path, data_dir=None, transform=None, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test

        # 如果未指定 data_dir，默认取 csv_path 所在目录
        if data_dir is None:
            data_dir = os.path.dirname(os.path.abspath(csv_path))
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # CSV 中的 image 列为相对路径，与 data_dir 拼接
        image_path = os.path.join(self.data_dir, row["image"])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # 测试模式：返回 (image_tensor, sample_id)
            sample_id = row["id"]
            return image, sample_id
        else:
            # 训练模式：返回 (image_tensor, label_index)
            label_name = row["label"]
            label = label2id[label_name]
            return image, label
