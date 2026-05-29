# NS-2026-00 文档图像分类 Baseline

NeuronSpark 2026 入门赛题 —— 10 分类校园文档图像分类 Baseline 实现。

## 环境安装

```bash
pip install -r requirements.txt
```

> **国内用户注意**：如果预训练权重下载失败，请在训练前设置 HuggingFace 镜像：
> ```bash
> # Windows PowerShell
> $env:HF_ENDPOINT='https://hf-mirror.com'
> # Linux / macOS
> export HF_ENDPOINT=https://hf-mirror.com
> ```
> 之后运行 `python train.py ...` 即可正常下载预训练权重。

## 项目结构

```
project/
├── data.py              # 数据集、数据增强、标签映射
├── model.py             # EfficientNet-B0 模型构建
├── train.py             # 训练脚本
├── predict.py           # 推理脚本
├── requirements.txt     # 依赖列表
├── README.md            # 本文件
├── checkpoints/         # 模型保存目录
│   └── best_model.pth   # 最佳模型权重
└── results.csv          # 推理结果
```

## 数据准备

确保数据目录包含以下文件结构：

```
data_dir/
├── train.csv
├── test.csv
├── label_map.json
└── images/
    ├── train/
    │   └── *.jpg
    └── test/
        └── *.jpg
```

## 训练

```bash
python train.py --data_dir <数据目录路径>
```

示例：

```bash
python train.py --data_dir d1b5a028-b288-4ab9-a872-64c6e12a9185
```

训练参数：
- Batch Size: 32
- Epochs: 10
- Learning Rate: 1e-4
- Optimizer: AdamW
- Loss: CrossEntropyLoss
- 验证集比例: 20%（stratify 分层划分）

训练过程会输出每个 epoch 的 Train Loss、Train Acc、Val Acc，并自动保存验证准确率最高的模型至 `checkpoints/best_model.pth`。

## 推理

```bash
python predict.py --data_dir <数据目录路径>
```

示例：

```bash
python predict.py --data_dir d1b5a028-b288-4ab9-a872-64c6e12a9185
```

推理完成后生成 `results.csv`，格式如下：

| id | label |
|----|-------|
| ns26_xxx | invoice |

## Baseline 原理

### 模型选型

使用 **EfficientNet-B0** 作为 backbone，主要原因：

- 通过 NAS（神经架构搜索）在精度与效率之间取得优秀平衡
- B0 是最轻量的版本，适合快速实验和 baseline 迭代
- ImageNet 预训练权重提供了良好的特征提取基础

### 数据增强

训练时使用以下增强策略提升模型泛化能力：

- **RandomRotation(±15°)**：模拟文档拍摄角度变化
- **ColorJitter**：模拟不同光照条件
- **RandomHorizontalFlip**：增加样本多样性
- **Normalize**：使用 ImageNet 均值/标准差归一化

### 训练策略

- **AdamW 优化器**：带权重衰减的 Adam，收敛更稳定
- **分层验证集划分（stratify）**：保证各类别在训练/验证集中比例一致
- **最佳模型保存**：根据验证集准确率自动保存最优 checkpoint

### 后续升级方向

- **OCR 特征融合**：提取文档中的文字信息作为辅助特征
- **TTA（测试时增强）**：推理时多次增强取平均，提升预测稳定性
- **更大的 backbone**：EfficientNet-B3/B4 或 ViT 系列
- **多模态模型**：结合文本 OCR + 图像 Layout 信息
- **Mixup / CutMix**：更高级的数据增强策略
