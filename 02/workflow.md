# NS-2026-02 RAG 基线方案 Workflow

## 项目结构

```
C:/Users/32010/ai-competition/3/
├── 1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/   ← 比赛数据
│   ├── corpus.jsonl                          # 知识库（90条）
│   ├── train_qa.jsonl                        # 训练QA对
│   ├── test_questions.jsonl                  # 测试问题（332条）
│   └── schema.json                           # 提交格式
├── models/
│   └── Qwen2.5-3B-Instruct/                  # 本地LLM（5.75GB）
├── outputs/
│   └── results.json                          # 最终提交文件
├── build_index.py                            # 索引构建
├── retrieve.py                               # 混合检索 + Rerank
├── answer.py                                 # 答案生成（方案B: LLM）
├── run.py                                    # 入口
├── utils.py                                  # Prompt清洗
├── requirements.txt                          # Python依赖
├── bm25.pkl                                  # BM25索引（运行时生成）
├── docs.pkl                                  # 文档数据（运行时生成）
└── embeddings.npy                            # 向量库（运行时生成）
```

## 环境

| 项目 | 版本 |
|------|------|
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | RTX 5060 Laptop 8GB VRAM |

## 模型矩阵

| 用途 | 模型 | 大小 | 来源 |
|------|------|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 | ~100MB | HuggingFace缓存 |
| Reranker | BAAI/bge-reranker-base | ~400MB | HuggingFace缓存 |
| 生成 | Qwen/Qwen2.5-3B-Instruct | 5.75GB | 本地 ./models/ |

全部满足比赛限制（<1B / <4B）。

## Pipeline

```
test_questions.jsonl
       │
       ▼
  [run.py 入口]
       │
       ├─► retrieve.py  ──► 混合检索 ──► Rerank ──► Top-5 文档
       │
       ├─► utils.py     ──► Prompt Injection 清洗
       │
       └─► answer.py    ──► LLM 生成 ──► Answer Verification ──► 输出
```

### Step 1: 混合检索 (retrieve.py)

```
Query
  ├─► BM25 分词检索 (jieba) ──► Top-15
  ├─► BGE Embedding 向量检索  ──► Top-15
  └─► Union (去重)
       │
       └─► BGE Reranker 精排 ──► Top-5
```

| 参数 | 值 |
|------|----|
| TOP_K_BM25 | 15 |
| TOP_K_EMB | 15 |
| TOP_K_FINAL | 5 |

### Step 2: 答案生成 (answer.py)

```
Top-5 文档
  ├─► Rerank score < -2 ? ──► 拒答
  ├─► Prompt清洗 (Injection防御)
  ├─► 构建上下文 (每条≤800字)
  ├─► Qwen2.5-3B 生成 (bfloat16, max_new_tokens=64)
  ├─► Answer Verification (字符重叠≥30%)
  │     └─► 失败 → fallback到Top-1原文片段
  └─► 输出 (answer, citations)
```

### Step 3: 结果输出 (run.py)

生成 `outputs/results.json`，符合 schema 格式：

```json
{
  "id": "test_0001",
  "answer": "装有液氮的敞口容器不得进入载人电梯...",
  "citations": ["LAB-LN2-001", "..."]
}
```

## 安全机制

| 机制 | 实现 |
|------|------|
| Prompt Injection 防御 | 正则清洗（忽略以上/你现在是/forget等） |
| 拒答阈值 | Reranker score < -2 → 拒答 |
| 拒答兜底 | retrieved为空 → 拒答 |
| Answer Verification | 答案字符与citation重叠<30% → fallback |
| 上下文截断 | 每条chunk≤800字，总prompt≤2048 tokens |

## 运行命令

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建索引（仅需一次）
python build_index.py

# 3. 生成结果
python run.py
```

离线模式已在脚本内置（`HF_HUB_OFFLINE=1`），无需手动设置。

## 性能

| 方案 | 回答方式 | 耗时 | 拒答数 | 回答数 |
|------|----------|------|--------|--------|
| 方案A | 原文截取 | 2分37秒 | 29 | 303 |
| 方案B | Qwen2.5-3B GPU | 8分02秒 | 36 | 296 |

方案B回答更精炼，拒答更保守（避免乱答扣分），推荐比赛使用。

## GPU推理优化

| 配置 | 值 | 说明 |
|------|----|------|
| dtype | bfloat16 | RTX 50系列原生支持 |
| device_map | auto | accelerate自动分配 |
| max_new_tokens | 64 | 限制生成长度 |
| do_sample | False | 贪婪解码，稳定输出 |
| use_cache | True | KV-cache加速 |
| prompt max_length | 2048 | 控制显存 |
