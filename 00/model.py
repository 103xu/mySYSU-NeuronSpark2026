"""
NS-2026-00 文档图像分类 - 模型模块
使用 timm 库加载 EfficientNet-B0 预训练模型
"""
import os
import warnings

# 国内用户默认使用 HuggingFace 镜像站，可通过环境变量覆盖
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn as nn
import timm

from data import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    构建 EfficientNet-B0 分类模型

    Args:
        num_classes: 分类类别数，默认 10
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        model: PyTorch 模型
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


if __name__ == "__main__":
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
