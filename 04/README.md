# NS-2026-04 分子炼金术：小分子性质预测

## 题目要求

根据小分子的 SMILES 表示预测其物理化学性质或活性指标。需从分子字符串中构造描述符、fingerprint 或图结构，输出每个测试分子的连续预测值。

### 数据规模

| 文件 | 规模 |
|------|------|
| train.csv | 8,500 条 |
| test.csv | 3,000 条 |
| hashed_smiles_train.npz | 8,500 × 2,048 |
| hashed_smiles_test.npz | 3,000 × 2,048 |

### 评分规则

| 指标 | 分值 |
|------|------|
| RMSE 得分 | 500 |
| Spearman 得分 | 250 |
| 格式合法性 | 50 |
| **总分** | **800** |

```
raw_score = 500 × max(0, 1 − RMSE / 0.80) + 250 × max(0, Spearman) + 50
if RMSE ≤ 0.25 and Spearman ≥ 0.96: score = 800 (gold band)
```

### 提交格式

- 文件名：`NS-2026-04-answer.zip`
- 包内含：`results.csv`（仅此一个文件）
- CSV 格式：`id,prediction`，必须覆盖全部 3,000 个测试分子

## 方案设计

### 特征工程

| 特征 | 维度 | 来源 |
|------|------|------|
| Hashed SMILES token fingerprint | 2,048 | 官方预提取 |
| Morgan fingerprint (r=2) | 2,048 | RDKit |
| Morgan fingerprint (r=3) | 2,048 | RDKit |
| MACCS keys | 167 | RDKit |
| Atom pair fingerprint | 2,048 | RDKit |
| Molecular descriptors | 217 | RDKit |
| 手工特征 | 21 | SMILES 规则 |
| **合计** | **8,597** | |

经方差过滤后保留 8,566 维。

### 模型

采用 **Scaffold-aware 6 折交叉验证**（1 折官方划分 + 5 折 K-Fold）。

| 模型 | 说明 |
|------|------|
| CatBoost | GPU 训练，depth=7, lr=0.015 |
| XGBoost | CPU (hist), depth=6, lr=0.015 |
| LightGBM | CPU, depth=7, lr=0.015 |
| Ridge | alpha=100, 标准化后训练 |
| MLP (sklearn) | 三层 256→128→64, ReLU, Adam |

### 集成

Ridge Stacking：将五个模型的 OOF 预测作为元特征，训练 Ridge 回归得到最终预测。

Meta 权重：XGB(0.40) > LGB(0.33) > Cat(0.25) > Ridge(0.04) > MLP(≈0)

## OOF 交叉验证结果

| 模型 | RMSE | Spearman | 得分 |
|------|------|----------|------|
| XGBoost | 0.255 | 0.959 | 630 |
| LightGBM | 0.256 | 0.959 | 630 |
| CatBoost | 0.266 | 0.954 | 622 |
| Ridge | 0.425 | 0.892 | 507 |
| MLP | 0.335 | 0.930 | 573 |
| **Stack 集成** | **0.252** | **0.959** | **632** |

## 项目结构

```
04/
├── 01_extract_features.py   # 特征提取脚本
├── 02_train_models.py       # 模型训练脚本（初版）
├── train_all.py              # 全模型训练 + Stacking 集成（终版）
├── explore_data.py           # 数据探索
├── X_train_features.npz      # 训练特征缓存
├── X_test_features.npz       # 测试特征缓存
├── results.csv               # 最终预测结果
├── NS-2026-04-answer.zip    # 提交文件
└── 2a9fa6fc-.../            # 原始数据目录
    ├── train.csv
    ├── test.csv
    ├── scaffold_split.json
    ├── features/
    │   ├── hashed_smiles_train.npz
    │   └── hashed_smiles_test.npz
    ├── example_submission/
    └── tools/
        ├── check_format.py
        └── make_features.py
```

## 运行方式

```bash
# 1. 特征提取（生成 X_train_features.npz / X_test_features.npz）
python 01_extract_features.py

# 2. 训练 + 预测 + 打包
python train_all.py
```

依赖：`numpy pandas scipy scikit-learn xgboost lightgbm catboost rdkit`
