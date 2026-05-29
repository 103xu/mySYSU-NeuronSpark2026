"""
NS-2026-02 优化版检索模块
- 混合检索 (BM25 + Embedding)
- BGE Reranker 精排
- Entity-aware score boosting
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import pickle
import re
import jieba
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from utils import clean_query

TOP_K_BM25 = 15
TOP_K_EMB = 15
TOP_K_FINAL = 3          # 只用 top-3（引用最多2条）

# ── 实体权重提升 ──────────────────────────────────────
_ENTITY_RE = re.compile(r"\d+年|\d+月|\d+日|\d+小时|\d+分钟|\d+天|\d+个?工作日|\d+:\d+|[A-Za-z0-9一-鿿]{2,}")


def _entity_overlap(text_a: str, text_b: str) -> float:
    """计算两个文本的实体重叠度"""
    ents_a = set(_ENTITY_RE.findall(text_a))
    ents_b = set(_ENTITY_RE.findall(text_b))
    if not ents_a:
        return 0.0
    return len(ents_a & ents_b) / len(ents_a)


# ── 加载资源 ──────────────────────────────────────────
print("Loading resources...")

with open("bm25.pkl", "rb") as f:
    bm25 = pickle.load(f)

with open("docs.pkl", "rb") as f:
    docs = pickle.load(f)

embeddings = np.load("embeddings.npy")

embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
rerank_model.eval()


# ── 检索函数 ──────────────────────────────────────────


def bm25_search(query):
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    idx = np.argsort(scores)[::-1][:TOP_K_BM25]
    return idx.tolist()


def embedding_search(query):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores = np.dot(embeddings, q_emb.T).flatten()
    idx = np.argsort(scores)[::-1][:TOP_K_EMB]
    return idx.tolist()


@torch.no_grad()
def rerank(query, candidate_ids):
    """BGE Reranker + entity-aware boosting"""
    pairs = []
    for cid in candidate_ids:
        doc = docs[cid]
        text = f"{doc['title']} {doc['text']}"
        pairs.append([query, text])

    inputs = rerank_tokenizer(
        pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
    )

    scores = rerank_model(**inputs).logits.view(-1).float().cpu().numpy()

    # Entity-aware boosting: 问题中的实体与chunk中的实体重叠越多，加分
    boosted = []
    for (cid, score) in zip(candidate_ids, scores):
        doc = docs[cid]
        chunk_text = f"{doc['title']} {doc['text']}"
        overlap = _entity_overlap(query, chunk_text)
        boosted_score = score + 0.3 * overlap  # entity boost
        boosted.append((cid, boosted_score))

    ranked = sorted(boosted, key=lambda x: x[1], reverse=True)
    return ranked


def retrieve(query):
    query = clean_query(query)

    bm25_ids = bm25_search(query)
    emb_ids = embedding_search(query)

    candidate_ids = list(set(bm25_ids + emb_ids))
    ranked = rerank(query, candidate_ids)

    results = []
    for idx, score in ranked[:TOP_K_FINAL]:
        doc = docs[idx]
        results.append(
            {
                "score": float(score),
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "title": doc["title"],
            }
        )

    return results
