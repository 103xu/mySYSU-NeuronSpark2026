# NS-2026-07 灵巧手闭环操作

## 赛题概述

- **赛道**: NS-2026-07 灵巧手非抓握式物体操作
- **难度**: Hard（满分 1500）
- **控制模式**: 闭环（closed-loop），最大 128 步/回合
- **评分公式**: `competitionScore = 1500 × (0.25 + 0.55 × p_i + 0.20 × g_i)`
  - `p_i`: 任务平均 score01
  - `g_i`: 能力层级最低 score01（按短板聚合）

## 观察空间

| 模态 | 维度 | 类型 | 说明 |
|------|------|------|------|
| `low_dim_state` | 14 | float32 | 物体相对位姿 (dx, dy, dθ) + 手部关节状态 |
| `tactile_heatmap_7x4` | 7×4 | float32 | 7 指 × 4 通道触觉热力图 |
| `tactile_image` | 7×8×8 | uint8 | 7 指 × 8×8 触觉图像 |
| `vision_grid` | 6×16×16 | uint8 | 俯视视觉网格 |
| `tactile_history` | 6×5 | float32 | 6 步滑动窗口触觉历史 |
| `contact_summary` | 4 | float32 | 接触覆盖率/最小接触/滑移风险/损伤风险 |
| `stage_context` | 4 | float32 | 阶段启用/当前阶段索引/阶段总数/完成比例 |

## 动作空间

| 维度 | 类别数 | 说明 |
|------|--------|------|
| `primitive` | 10 | brace / push / drag / pivot / roll / lift_edge / tap / stabilize / wait / finish |
| `finger` | 7 | thumb / index / middle / ring / pinky / palm / wrist |
| `force` | 连续 [0, 1] | 施加力的大小 |
| `direction` | 连续 2D | 归一化方向向量 |

## 任务类型

| 类型 | 说明 |
|------|------|
| `nonprehensile_relocation` | 将物体推到目标位姿 |
| `tool_use` | 使用工具轴操作物体 |
| `resource_sequence` | 有手指约束的多阶段操作 |

## 能力层级

| 层级 | 说明 |
|------|------|
| base_state_feedback | 基础状态反馈下的物体重定位 |
| tactile_slip_recovery | 触觉感知下的滑移恢复 |
| cross_condition_generalization | 跨物体/环境/场景的泛化 |

## 运行时约束

| 约束 | 限制 |
|------|------|
| GPU | 1× RTX 4090D 24GB |
| 单步推理 (`act`) | ≤ 2s |
| 批量推理 (`act_batch`) | ≤ 10s（batch ≤ 64） |
| 最大步数 | 128 步/回合 |
| Python | 3.12 |
| PyTorch | 2.12.0+cu130 |
| 仿真器 | MuJoCo 3.3.7 |
| 网络 | 禁止外部访问 |

## 方案设计

### 模型架构

多模态编码器 → 融合 MLP → FiLM 任务调制 → 多头输出

| 编码器 | 输入 | 结构 | 输出维度 |
|--------|------|------|----------|
| VisionGridEncoder | 6×16×16 | 3 层 CNN | 128 |
| TactileImageEncoder | 7×8×8 | 2 层 CNN | 128 |
| TactileHeatmapEncoder | 7×4 | MLP | 64 |
| LowDimStateEncoder | 14 | MLP | 128 |
| ContactSummaryEncoder | 4 | MLP | 32 |
| TactileHistoryEncoder | 6×5 | GRU | 64 |
| StageContextEncoder | 4 | MLP | 32 |
| TaskTypeEmbedding | 1 (int) | Embedding(3→32) | 32 |
| SensorStatusEncoder | 3 | MLP | 16 |

- **融合层**: Concat(624) → Linear(768) → ReLU → Dropout(0.15) → Linear(768) → ReLU → Dropout(0.15) → Linear(512)
- **FiLM 调制**: 任务类型嵌入对融合特征做 Feature-wise Linear Modulation
- **参数量**: 1.84M (FP32 ≈ 7.2MB)

### 训练

- **数据**: 1200 弱演示（成功率 9.58%），过滤 min_score01 ≥ 0.1
- **算法**: Behavior Cloning（行为克隆）
- **损失**: CE(primitive) + 0.7×CE(finger) + MSE(force) + 1.5×(1-cos_sim(direction)) + 0.1×CE(task_aux)
- **优化器**: AdamW (lr=3e-4, weight_decay=1e-5)
- **调度器**: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **训练轮数**: 60 epochs
- **最佳 val_loss**: 0.1164 (epoch 42)
- **混合精度**: AMP (torch.amp)

### Agent 混合策略

采用**规则+模型混合**架构：规则保证安全底线，模型优化操作质量。

**规则层**（优先级从高到低）:
1. `finish` — 距离 < 0.085 且角度偏差 < 0.34
2. `brace` — 接触丢失 / 滑移风险 > 0.92 / 损伤风险 > 0.98
3. `tap` — tool_use / resource_sequence 的周期性工具点击
4. `pivot` — 角度偏差 > 0.22 时纠正方向

**模型层**: 预测 push/drag/roll 操作的 direction、force、finger

**防御性设计**:
- 模型加载失败自动回退到纯规则控制器
- 脆性物体（fragility > 0.58）自动限力
- 障碍物避让

## 文件结构

```
submission/
├── agent.py                 # Agent 类 (reset/act + reset_batch/act_batch)
├── model/
│   ├── __init__.py          # 模块导出
│   ├── config.py            # 超参数配置
│   ├── encoders.py          # 9 个模态编码器
│   ├── policy.py            # DexPolicy 策略网络 + FiLMBlock
│   ├── best_model.pt        # 训练权重 (1.84M 参数, 7.2MB)
│   └── model_manifest.json  # 模型清单
├── training/
│   ├── data_loader.py       # 演示数据解析 + PyTorch Dataset
│   ├── train.py             # BC 训练脚本
│   └── losses.py            # 多头加权损失函数
├── package.py               # 提交打包脚本
└── README.md
```

## 使用方式

```bash
# 格式检查
python tools/check_format.py submission/

# 本地评测
python tools/run_public_eval.py submission/ --tasks tasks/valid_tasks.jsonl
python tools/run_public_eval.py submission/ --tasks tasks/test_tasks.jsonl

# 单回合调试
python tools/render_replay.py submission/ --tasks tasks/valid_tasks.jsonl --index 0

# 打包提交
python package.py --root submission --output NS-2026-07-answer.zip
```

## 本地评测结果

| 指标 | 数值 |
|------|------|
| Valid score | 263.36 / 1500 |
| Test score | 293.10 / 1500 |
| 成功率 | 6.5% |
| 动作错误 | 0 |

### 分项得分 (Valid)

| 任务类型 | score01 |
|----------|---------|
| nonprehensile_relocation | 0.167 |
| resource_sequence | 0.166 |
| tool_use | 0.250 |

| 环境 | score01 |
|------|---------|
| matte_table | 0.207 |
| soft_damping_pad | 0.204 |
| micro_ridge_pad | 0.192 |
| split_texture_pad | 0.191 |
| low_dust_glass | 0.188 |
| low_rim_fixture | 0.164 |

## 局限性

- BC 模型从弱演示学习，无法显著超越手工规则
- resource_sequence 任务得分较低（需更复杂的多阶段规划）
- tactile_slip_recovery 能力短板（0.137），需专门针对滑移场景训练
- 模型推理在 CPU 上较慢，依赖 GPU 加速
