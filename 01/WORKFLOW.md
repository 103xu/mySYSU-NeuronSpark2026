# AI 比赛 Baseline 开发标准流程 (SOP)

> 基于 PyTorch 的 AI 竞赛 baseline 快速搭建与迭代标准操作流程。
> 适用场景：图像分类 / OCR / NLP / 多模态等 AI 比赛任务。

---

## 0. Python 虚拟环境标准搭建

### 0.1 为什么必须使用虚拟环境

| 原因 | 说明 |
|---|---|
| **避免依赖冲突** | 不同项目可能依赖不同版本的 torch / numpy，全局安装会互相覆盖 |
| **比赛隔离** | 每道赛题的依赖独立管理，互不干扰 |
| **方便复现** | `requirements.txt` + 虚拟环境 = 一键复现训练环境 |
| **防止污染全局 Python** | 不会因比赛项目破坏系统 Python 环境 |

### 0.2 标准项目初始化流程

```bash
# 1. 创建项目目录
mkdir project_name
cd project_name

# 2. 创建虚拟环境
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate

# 3. 升级 pip
pip install --upgrade pip
```

### 0.3 安装依赖

```bash
# 方法一：从 requirements.txt 安装
pip install -r requirements.txt

# 方法二：手动安装核心依赖
pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm

# 导出当前环境依赖（供他人复现）
pip freeze > requirements.txt
```

### 0.4 CUDA / GPU 环境检查

```python
import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 型号:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
    # 快速 GPU 张量测试
    x = torch.randn(100, 100).cuda()
    print("GPU 张量测试: OK")
else:
    print("当前为 CPU 模式，训练速度较慢")
```

**关键注意事项：**

- PyTorch 版本必须匹配 CUDA 版本（如 `torch==2.11.0+cu128` 对应 CUDA 12.8）
- RTX 50 系列 (Blackwell, sm_120) 需要 PyTorch >= 2.7，建议使用最新稳定版 + CUDA 12.8
- 如使用 `uv` 管理项目，需在 `pyproject.toml` 中配置 PyTorch CUDA 源：

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

- 国内用户建议配置 HuggingFace 镜像（在代码中自动设置或手动设置环境变量）：

```python
import os
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 0.5 推荐基础依赖

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
Pillow>=9.0.0
tqdm>=4.64.0
opencv-python>=4.8.0
```

### 0.6 标准工程初始化 Checklist

```
[ ] 创建项目目录
[ ] 创建虚拟环境 (python -m venv venv)
[ ] 激活虚拟环境
[ ] 安装依赖 (pip install -r requirements.txt)
[ ] 验证 GPU (python -c "import torch; print(torch.cuda.is_available())")
[ ] 验证预训练权重可下载 (python -c "import timm; timm.create_model('efficientnet_b0', pretrained=True)")
[ ] 创建项目文件结构 (data.py / model.py / train.py / predict.py)
[ ] 跑通端到端训练流程 (1 batch 即可)
[ ] 跑通端到端推理流程 (生成 results.csv)
[ ] 开始正式开发
```

---

## 1. 比赛题分析流程

### 1.1 标准问题分析 Checklist

拿到新题目时，逐项确认：

```
[ ] 任务类型是什么？
    - 图像分类 / 目标检测 / OCR / NLP / 多模态 / 回归

[ ] 输入是什么？
    - 图像 (jpg/png) / 文本 (csv/txt) / 结构化数据 / 混合

[ ] 输出是什么？
    - 单标签分类 / 多标签分类 / 文本序列 / 数值

[ ] 类别数 (num_classes)？
    - N 分类，N = ?

[ ] 类别是否均衡？
    - 使用 df['label'].value_counts() 检查
    - 不均衡 → 考虑 class_weight 或 oversampling

[ ] 数据规模？
    - 训练集样本数
    - 测试集样本数
    - 图片尺寸是否统一

[ ] 是否需要 OCR？
    - 文档 / 票据 / 车牌 / 招牌 → 大概率需要

[ ] 是否需要 NLP？
    - 文本分类 / 情感分析 / 命名实体识别

[ ] 适合什么 backbone？
    - 通用图像 → EfficientNet / ResNet
    - 文档图像 → EfficientNet + OCR 特征
    - 细粒度 → ConvNeXt / ViT
    - 文本 → BERT / RoBERTa
```

### 1.2 快速数据分析脚本

```python
import pandas as pd

df = pd.read_csv("train.csv")
print("样本数:", len(df))
print("类别数:", df["label"].nunique())
print("类别分布:\n", df["label"].value_counts())
print("缺失值:\n", df.isnull().sum())
```

---

## 2. 标准项目结构

```
project/
├── data.py              # Dataset 类 + transform + 标签映射
├── model.py             # 模型构建函数
├── train.py             # 训练主脚本
├── predict.py           # 推理主脚本
├── requirements.txt     # Python 依赖
├── README.md            # 项目说明
├── WORKFLOW.md          # 本文件（开发 SOP）
│
├── checkpoints/         # 模型权重保存目录
│   └── best_model.pth
│
├── configs/             # （可选）配置文件
│   └── baseline.yaml
│
└── outputs/             # （可选）预测结果
    └── results.csv
```

### 2.1 各文件职责与协作关系

| 文件 | 职责 | 被谁调用 |
|---|---|---|
| `data.py` | Dataset 定义、数据增强、标签映射 (`label2id`, `id2label`) | `train.py`, `predict.py` |
| `model.py` | 模型构建 (`build_model()`)，返回 `nn.Module` | `train.py`, `predict.py` |
| `train.py` | 数据加载 → 训练 → 验证 → 保存最佳模型 | 独立运行 |
| `predict.py` | 加载模型 → 推理 → 生成 `results.csv` | 独立运行 |

### 2.2 为什么这样拆分

- **单一职责**：Dataset 只管数据，model 只管网络，train/predict 只管流程
- **可复用**：`data.py` 和 `model.py` 被 train 和 predict 共同引用，无需重复代码
- **可替换**：换 backbone 只改 `model.py`，换数据增强只改 `data.py`
- **可调试**：每个模块可独立单元测试

---

## 3. Dataset 标准模板

### 3.1 核心三方法

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, csv_path, data_dir=None, transform=None, is_test=False):
        """
        Args:
            csv_path:  标注文件路径
            data_dir:  图片根目录（CSV 中 image 路径相对此目录）
            transform: torchvision 数据增强
            is_test:   True=测试模式（无标签），False=训练模式
        """
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.data_dir, row["image"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, row["id"]          # 测试: (tensor, id)
        else:
            return image, label2id[row["label"]]  # 训练: (tensor, label)
```

### 3.2 transform 设计原则

```python
from torchvision import transforms

# ImageNet 标准归一化参数（使用预训练权重时必须）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 训练 transform：包含数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 统一尺寸
    transforms.RandomRotation(degrees=15),   # 随机旋转（模拟拍摄角度）
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # 颜色抖动（模拟光照变化）
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.ToTensor(),                   # [0,255] -> [0,1] + HWC -> CHW
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # 标准化
])

# 验证/测试 transform：仅 resize + 归一化，不做增强
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

**关键原则：**
- 验证集和测试集**绝不**使用数据增强（RandomRotation/ColorJitter 等）
- 使用预训练权重时，Normalize 参数必须与预训练时一致（ImageNet 参数）
- ToTensor 必须在 Normalize 之前

---

## 4. DataLoader 工作流

### 4.1 数据流

```
CSV 文件 → Dataset.__getitem__() → 单张 (tensor, label)
                                        ↓
                              DataLoader 收集 batch_size 张
                                        ↓
                               (B, C, H, W) + (B,) tensor
                                        ↓
                                  .to(device) GPU
                                        ↓
                                    Model
```

### 4.2 标准 DataLoader 配置

```python
from torch.utils.data import DataLoader

# 训练：shuffle=True，打乱顺序
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,              # 训练必须打乱
    num_workers=4,              # 多进程加载，提高 GPU 利用率
    pin_memory=True,            # 锁页内存，加速 CPU→GPU 传输
)

# 验证/测试：shuffle=False，保持顺序
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,             # 验证/测试不打乱
    num_workers=4,
    pin_memory=True,
)
```

### 4.3 参数说明

| 参数 | 训练 | 验证/测试 | 说明 |
|---|---|---|---|
| `shuffle` | `True` | `False` | 训练打乱防止模型记忆顺序 |
| `num_workers` | 2-8 | 2-8 | 一般设为 CPU 核心数，≤ 8 |
| `pin_memory` | `True` | `True` | GPU 训练时开启，加速数据传输 |
| `drop_last` | `False` | `False` | 是否丢弃最后不完整的 batch |

---

## 5. 模型选择策略

### 5.1 Baseline 模型选择指南

| 任务类型 | 推荐 Baseline | 参数量 | 理由 |
|---|---|---|---|
| 通用图像分类 | `efficientnet_b0` | ~4M | 轻量、精度好、收敛快 |
| 高精度需求 | `efficientnet_b3` | ~10M | B0 的大哥，精度更高 |
| 细粒度分类 | `convnext_tiny` | ~28M | 更强的特征提取能力 |
| 文档/票据分类 | `efficientnet_b0` + OCR | ~4M | 先跑纯图像 baseline |
| Transformer 路线 | `vit_base_patch16_224` | ~86M | 需要更多数据 |
| 轻量/移动端 | `mobilenetv3_small` | ~2M | 极致轻量 |

### 5.2 模型构建标准函数

```python
import timm

def build_model(num_classes=10, pretrained=True):
    """
    构建分类模型
    自动处理预训练权重下载失败（国内网络问题）
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
```

### 5.3 选型原则

1. **Baseline 阶段**：选最成熟的轻量模型（EfficientNet-B0），先跑通流程
2. **提分阶段**：逐步尝试更大 backbone 或 Transformer
3. **不要一开始就上大模型**：调试慢、容易 OOM、不易定位问题

---

## 6. 训练标准流程

### 6.1 train.py 核心结构

```
读取 train.csv
    ↓
train_test_split (stratify 分层划分 80/20)
    ↓
创建 DataLoader (train + val)
    ↓
build_model() → model.to(device)
    ↓
定义 criterion (CrossEntropyLoss) + optimizer (AdamW)
    ↓
for epoch in range(EPOCHS):
    train_one_epoch()      # forward → loss → backward → step
    validate()             # model.eval() → 计算 val_acc
    if val_acc > best:     # 保存最佳模型
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
    ↓
训练完成，输出最佳 val_acc
```

### 6.2 标准训练代码骨架

```python
# 超参数
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4

# 模型
model = build_model(num_classes=10, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 训练循环
for epoch in range(1, EPOCHS + 1):
    # --- 训练阶段 ---
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # --- 验证阶段 ---
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            # 累计正确数

    # --- 保存最佳 ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
```

### 6.3 关键组件说明

| 组件 | 作用 | 常见选择 |
|---|---|---|
| `CrossEntropyLoss` | 多分类损失函数 | 内置 softmax + NLLLoss |
| `AdamW` | 自适应学习率优化器 | 比 SGD 收敛更快更稳定 |
| `CosineAnnealingLR` | 学习率衰减 | 提升后期精度 |
| `model.train()` | 开启 Dropout/BN 训练模式 | 训练前调用 |
| `model.eval()` | 关闭 Dropout/BN 训练模式 | 验证/推理前调用 |
| `torch.no_grad()` | 关闭梯度计算 | 验证/推理时节省显存 |

---

## 7. 推理标准流程

### 7.1 predict.py 核心结构

```
加载 best_model.pth
    ↓
model.eval()
    ↓
读取 test.csv → DataLoader (shuffle=False)
    ↓
for images, ids in test_loader:
    outputs = model(images)        # (B, num_classes)
    preds = torch.argmax(outputs, dim=1)  # (B,) 类别索引
    id2label[pred] → 类别字符串
    ↓
生成 results.csv
```

### 7.2 为什么测试集有这些特殊要求

| 要求 | 原因 |
|---|---|
| `shuffle=False` | 保持样本顺序，结果可复现 |
| `model.eval()` | 关闭 Dropout，BatchNorm 使用全局统计量 |
| `torch.no_grad()` | 不计算梯度，节省显存，加速推理 |
| `weights_only=True` (torch.load) | 安全加载，防止 pickle 注入攻击 |

### 7.3 标准推理代码骨架

```python
model = build_model(num_classes=10)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

results = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        for sample_id, pred_idx in zip(ids, preds.cpu()):
            results.append([sample_id, id2label[pred_idx.item()]])

pd.DataFrame(results, columns=["id", "label"]).to_csv("results.csv", index=False)
```

---

## 8. 比赛提交规范

### 8.1 提交前 Checklist

```
[ ] results.csv 行数 = 测试集样本数（不含 header）
[ ] results.csv 列名为 id,label（与赛题要求一致）
[ ] id 列的值与 test.csv 完全一致，无缺失无重复
[ ] label 列的值在 label_map.json 定义的范围内
[ ] 编码为 UTF-8 (df.to_csv(encoding='utf-8-sig'))
[ ] 第一行是 header (id,label)，数据从第二行开始
[ ] 文件中不包含 BOM 异常字符
```

### 8.2 提交验证脚本

```python
import pandas as pd

# 验证 results.csv 格式
result = pd.read_csv("results.csv")
test = pd.read_csv("test.csv")

assert list(result.columns) == ["id", "label"], "列名错误"
assert len(result) == len(test), f"行数不匹配: {len(result)} vs {len(test)}"
assert set(result["id"]) == set(test["id"]), "id 不匹配"

# 检查 label 是否合法
with open("label_map.json") as f:
    valid_labels = json.load(f)["labels"]
assert result["label"].isin(valid_labels).all(), "存在非法 label"

print("格式验证通过！")
```

---

## 9. Baseline 提分路线

### 9.1 三阶段提分路线图

```
                    ┌─────────────────────────┐
  第一阶段          │  跑通 Baseline           │
  (0.85+)          │  - 数据增强              │
                    │  - 预训练权重            │
                    │  - AdamW + scheduler     │
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
  第二阶段          │  稳定提分               │
  (+0.03~0.05)     │  - 5-Fold Cross Validation│
                    │  - TTA (测试时增强)      │
                    │  - 更大的 backbone       │
                    │  - label smoothing       │
                    │  - Mixup / CutMix        │
                    └───────────┬─────────────┘
                                ↓
                    ┌─────────────────────────┐
  第三阶段          │  冲击高分               │
  (+0.05+)         │  - OCR 特征融合          │
                    │  - 多模态 (文本+布局)    │
                    │  - Ensemble (多模型投票) │
                    │  - 伪标签 / 自训练       │
                    │  - Transformer backbone  │
                    └─────────────────────────┘
```

### 9.2 各阶段详细策略

**第一阶段：稳定 Baseline**
- 数据增强：RandomRotation + ColorJitter + RandomHorizontalFlip
- 优化器：AdamW (lr=1e-4, weight_decay=1e-4)
- 学习率调度：CosineAnnealingLR 或 ReduceLROnPlateau
- 目标：val_acc >= 0.85

**第二阶段：稳定提分**
- 5-Fold Cross Validation：5 折交叉训练，取平均
- TTA：推理时对每张图做 5 次增强取平均概率
- Label Smoothing：防止过拟合，提升泛化
- Mixup：两张图按比例混合，增强鲁棒性
- 更大 backbone：EfficientNet-B3/B4 或 ConvNeXt

**第三阶段：多模态 + Ensemble**
- OCR 特征：提取文档文字，作为额外特征输入
- 多模态融合：图像特征 + 文本特征 → 联合分类器
- 多模型 Ensemble：不同 backbone 的结果加权投票
- 伪标签：用高置信度预测结果扩充训练集

---

## 10. 通用 Debug 流程

### 10.1 常见问题速查表

| 现象 | 可能原因 | 排查/修复方法 |
|---|---|---|
| **loss 不下降** | 学习率太高/太低、数据未归一化 | 尝试 lr=1e-3,1e-4,1e-5；检查 Normalize |
| **loss=NaN** | 学习率太高、梯度爆炸 | 降低 lr；加 gradient clipping |
| **CUDA OOM** | batch_size 太大、模型太大 | 减小 batch_size；使用 gradient accumulation |
| **val_acc 很低** | 过拟合、标签映射错误 | 检查 label2id；检查 transform 是否一致 |
| **shape mismatch** | 模型输出与标签维度不对 | 检查 num_classes；print(outputs.shape) |
| **val_acc > train_acc** | transform 不一致（训练用了增强验证没用）| 验证集使用 test_transform |
| **预测全是同一类** | 类别不均衡、模型未收敛 | 检查 class distribution；增加 epochs |
| **预训练权重下载失败** | 网络限制（国内） | 设置 HF_ENDPOINT 镜像 |

### 10.2 Debug 黄金法则

1. **先 overfit 小数据集**：用 100 张图训练，确保能达到 100% train_acc
2. **逐模块验证**：Dataset 先跑通 → Model forward 先跑通 → 1 batch 训练先跑通
3. **print shape**：每步 print tensor.shape，确认维度
4. **检查 device**：model 和 tensor 必须在同一设备 (cuda/cpu)
5. **固定随机种子**：确保每次运行结果一致

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
```

---

## 11. 通用开发原则

### 11.1 AI 比赛工程铁律

```
1. 先跑通，再优化
   不要一开始就搞复杂模型/花式增强，先让 baseline 完整跑通

2. 先保证训练闭环
   Dataset → DataLoader → Model → Loss → Backward → Val → Save
   任何一环断掉都不要继续往下走

3. 先保证 results.csv 正确
   格式错误 = 提交失败 = 白训练

4. 先验证 Dataset
   print(dataset[0]) 确认 (tensor, label) 的 shape 和内容

5. 先 overfit 小数据集
   100 张图如果都不能 overfit，完整数据更不可能

6. 先做可复现工程
   seed 固定、版本锁定 (requirements.txt)、参数可追溯

7. 一行改动，一次验证
   不要同时改 3 个参数然后不知道哪个有效

8. 保存每一个有效 checkpoint
   不要只保留 best model，阶段性保存便于回溯
```

### 11.2 开发优先级

```
P0 (必须)     Dataset 正确 → 训练能跑 → 能生成 results.csv
P1 (重要)     验证集划分 → 保存 best model → 训练日志完整
P2 (优化)     scheduler → 数据增强调优 → 超参数搜索
P3 (进阶)     5-Fold → TTA → Ensemble → OCR 融合
```

---

## 12. 新题目启动模板

### 12.1 接到新比赛题目后的标准动作

```bash
# Step 1: 创建项目
mkdir competition_name && cd competition_name
python -m venv venv && source venv/bin/activate  # 或 Windows: venv\Scripts\activate
pip install torch torchvision timm pandas numpy scikit-learn pillow tqdm

# Step 2: 分析数据
python -c "
import pandas as pd
df = pd.read_csv('train.csv')
print('样本数:', len(df))
print('类别:', df['label'].nunique())
print(df['label'].value_counts())
"

# Step 3: 复制 baseline 模板文件
# 从本 WORKFLOW.md 对应的项目复制 data.py / model.py / train.py / predict.py
# 修改 data.py 中的 label2id
# 修改 train.py 中的 --data_dir 默认值

# Step 4: 跑通端到端
python train.py
python predict.py

# Step 5: 验证提交格式
python -c "
import pandas as pd
r = pd.read_csv('results.csv')
print('行数:', len(r))
print('列:', r.columns.tolist())
print(r.head())
"
```

### 12.2 自动化启动 Prompt 模板

以后给 Claude Code 的新题目描述可直接使用：

```
我有一道新的 AI 比赛题：

任务类型：[图像分类/OCR/NLP/多模态]
类别数：[N]
数据格式：[描述 CSV 列名和图片路径结构]
提交格式：[results.csv 的列名要求]

请按照 WORKFLOW.md 的 SOP，自动：
1. 分析数据
2. 搭建 baseline 项目 (data.py / model.py / train.py / predict.py)
3. 训练模型
4. 生成 results.csv
```

---

## 附录 A：常用命令速查

```bash
# 环境检查
python -c "import torch; print(torch.cuda.is_available())"

# 安装依赖
pip install -r requirements.txt

# 导出依赖
pip freeze > requirements.txt

# 训练
python train.py --data_dir ./data

# 推理
python predict.py --data_dir ./data

# 验证提交
python -c "import pandas as pd; df=pd.read_csv('results.csv'); print(len(df), df.head())"
```

## 附录 B：常用代码片段索引

| 需求 | 位置 |
|---|---|
| Dataset 标准模板 | 第 3 节 |
| train_transform / test_transform | 第 3.2 节 |
| model build 函数 | 第 5.2 节 |
| 训练循环骨架 | 第 6.2 节 |
| 推理循环骨架 | 第 7.3 节 |
| 提交验证脚本 | 第 8.2 节 |
| Debug 速查表 | 第 10.1 节 |
| set_seed 函数 | 第 10.2 节 |
