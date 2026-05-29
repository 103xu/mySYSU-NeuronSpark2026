# NS-2026-09 观测之环：隐变量世界模型反事实预测

## 赛题简介

环形观测站正在排查一组异常网格世界。在只能看到 5×5 局部视野和粗粒度传感器摘要的条件下，根据有限的 probe 轨迹还原隐藏动力学变量，并预测同一隐藏世界中多个反事实 query 的后果。

- **类型**：探索型 Hard 题
- **满分**：1500
- **数据**：720 train / 160 valid / 320 test contexts，每个 context 含 5 个 probe episode + 2 个 query

## 方法概述

采用 **物理模拟器 + Probe 统计校准** 的混合方案：

### 1. 物理模拟器

基于对训练数据的分析，实现了网格世界的主要物理机制：

| 元素 | 符号 | 行为 |
|------|------|------|
| Agent | A | 执行 U/D/L/R/WAIT 动作，受冰面和传送门影响 |
| 墙壁 | # | 阻挡所有实体 |
| 冰面 | I | Agent 踩上后沿移动方向滑动 1 步 |
| 方向场 | ^ v < > | 推动站在上面的 Box(B) 和 Orb(O) |
| 传送门 | P | 最近邻配对，Agent 踩上后传送到配对门（3 步冷却） |
| 钥匙 | K | Agent 踩上后收集，用于开门 |
| 门 | D | 锁定状态阻挡移动，消耗钥匙解锁 |
| 箱子 | B | Agent 可推动，推上 Orb 触发 box_on_goal |
| 球体 | O | 站在方向场上时沿场方向移动 |
| 目标 | G | Agent 到达后触发 goal_reached，终局 goal |
| 危险 | H | Agent 到达后触发 hazard，终局 hazard |

### 2. Probe 校准

对每个 context 利用 5 个 probe episode 的已知结果：

- **终局类型推断**：统计 probe 中最常见的 terminal
- **事件频率统计**：用于校正模拟器的 false positive/negative
- **时间线回退**：当模拟器给 never 时，用 probe 统计推断 early/mid/late
- **事件顺序补全**：结合模拟器输出和 probe 顺序

### 3. 事件判断

6 个事件键的判定逻辑：

| 事件 | 触发条件 |
|------|---------|
| goal_reached | Agent 踩上 G |
| collision | Agent 撞墙/锁门/无法推动的箱子 |
| hazard | Agent 踩上 H |
| box_on_goal | Box 被推到 Orb 上 |
| key_collected | Agent 踩上 K |
| portal_used | Agent 踩上 P 并传送 |

## 文件结构

```
09/
├── main.py                 # Python 版主预测脚本
├── gen_results.js          # Node.js 版预测脚本（可直接运行）
├── validate_sim.js         # Node.js 版物理模拟验证
├── readme.md               # 本文件
├── results.json            # 生成的预测结果
├── run_predict.bat         # Windows 一键运行脚本
│
├── 4c6a5a9d-3aaa-4341-b4e5-63cf28710f16/   # 赛题数据
│   ├── train.jsonl          # 训练集（720 contexts, 1440 queries）
│   ├── valid.jsonl          # 验证集（160 contexts, 320 queries）
│   ├── test.jsonl           # 测试集（320 contexts, 640 queries）
│   ├── label_schema.json    # 标签格式定义
│   ├── frames/              # 观测帧 PNG
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── tools/
│   │   ├── baseline.py      # 官方基线
│   │   ├── check_format.py  # 提交格式检查
│   │   └── visualize_rollout.py
│   └── example_submission/
│       └── results.json
│
└── adbe8cb5-9a70-4c45-91cd-73fbc95024f8/   # 示例提交
    └── results.json
```

## 运行方式

### Node.js（推荐，已测试）

```bash
cd 4c6a5a9d-3aaa-4341-b4e5-63cf28710f16
node ../gen_results.js
```

### Python

```bash
cd 4c6a5a9d-3aaa-4341-b4e5-63cf28710f16
python ../main.py --tasks test.jsonl --out results.json
```

### 验证模式（在 train/valid 上评估）

```bash
python ../main.py --tasks valid.jsonl --out pred_valid.json --mode validate
```

### 格式检查

```bash
cd 4c6a5a9d-3aaa-4341-b4e5-63cf28710f16
python tools/check_format.py results.json --tasks test.jsonl
```

## 提交格式

`results.json` 中每条预测的结构：

```json
{
  "id": "wmli_t_0000_xxxxxxxx_q0",
  "final_grid": ["############", "#..........#", "..."],
  "events": {
    "goal_reached": false,
    "collision": true,
    "hazard": false,
    "box_on_goal": false,
    "key_collected": false,
    "portal_used": false
  },
  "event_timeline": {
    "goal_reached": "never",
    "collision": "early",
    "hazard": "never",
    "box_on_goal": "never",
    "key_collected": "never",
    "portal_used": "never"
  },
  "event_order": ["collision", "none", "none"],
  "terminal": "blocked"
}
```

提交时需将 `results.json` 单独压缩为 `NS-2026-09-answer.zip`。

## 评分项

| 项目 | 分值 |
|------|------|
| 动态网格准确率 | 225 |
| 关键实体位置准确率 | 225 |
| 事件向量准确率 | 250 |
| 事件时序准确率 | 250 |
| 事件顺序准确率 | 250 |
| 终局类型准确率 | 200 |
| 格式合法性 | 100 |
| **总分** | **1500** |

## 验证结果

在 200 个训练 context（1000 个 probe）上的模拟器准确率：

| 指标 | 准确率 |
|------|--------|
| Agent 位置 | 41.6% |
| 事件完全匹配 | 54.2% |
| 终局类型 | 77.2% |

## 改进方向

1. **冰面滑动距离**：当前固定 1 步，实际可能因 context 而异（1~N 步）
2. **方向场对 Agent 的影响**：当前不推 Agent（仅推 Box/Orb），需进一步验证
3. **传送门配对**：当前用最近邻配对，可能有多配对或单向传送
4. **Orb 独立移动**：Orb 在非场地上的移动机制尚未完全理解
5. **多 context 联合学习**：可训练神经网络从 probe 数据中端到端预测 query
